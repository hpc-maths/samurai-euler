// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cassert>
#include <cstddef>

#include <xtensor/containers/xfixed.hpp>

#include "eos.hpp"

// =============================================================================
//  Variables of the five-equation model
// -----------------------------------------------------------------------------
//  Two phases in pressure and velocity equilibrium, four conservation laws and
//  one advection equation:
//
//      d_t (alpha_i rho_i) + div(alpha_i rho_i u) = 0        i = 0, 1
//      d_t (rho u)         + div(rho u @ u + p I) = 0
//      d_t E               + div((E + p) u)       = 0
//      d_t alpha_0         + u . grad(alpha_0)    = 0
//
//  so dim + 4 components per cell. The last equation is written in the code as
//  div(alpha_0 u) - alpha_0 div(u): a flux, plus a term that is not one.
//
//  Everything here lives in namespace two_phase. The monofluid solver has
//  functions of the same names on its own state, the two models are never
//  compiled into the same executable, and the namespace is what says which one
//  a header is written against.
//
//  The layout keeps the momentum at 2 + d, exactly where the monofluid one puts
//  it. That is a convention the two models share on purpose, not a coincidence
//  to be discovered later: the solid wall boundary condition mirrors the normal
//  momentum and is written once for both, against `bc::momentum(d)`. The
//  static_assert below is what holds this layout to it.
// =============================================================================

namespace two_phase
{
    template <std::size_t Dim>
    struct Layout
    {
        // Partial densities alpha_i rho_i: the mass of each phase per unit
        // volume of mixture. They are what is conserved; rho_i alone is not.
        static constexpr std::size_t partial_rho(std::size_t i)
        {
            assert(i < 2);
            return i;
        }

        static constexpr std::size_t mom(std::size_t d)
        {
            assert(d < Dim);
            return 2 + d;
        }

        static constexpr std::size_t rhoE  = 2 + Dim;
        static constexpr std::size_t alpha = 3 + Dim;

        static constexpr std::size_t size = 4 + Dim;
    };

    template <std::size_t Dim>
    using ConsArray = xt::xtensor_fixed<double, xt::xshape<Layout<Dim>::size>>;

    // The primitive state the reconstruction works on: the volume fraction, the
    // two partial densities, the velocity and the pressure.
    //
    // The partial densities rather than the phasic ones: alpha_i rho_i is what
    // the scheme transports and what stays bounded as a phase vanishes, where
    // rho_i is 0/0 there. It is also the set the article reconstructs.
    template <std::size_t Dim>
    struct PrimState
    {
        double alpha;
        double partial_rho[2];
        double p;
        xt::xtensor_fixed<double, xt::xshape<Dim>> v;

        double rho() const
        {
            return partial_rho[0] + partial_rho[1];
        }
    };

    // The primitive state flattened onto the conservative layout, so that slopes
    // and face values can be added and scaled as plain arrays. Pressure sits
    // where the total energy is, velocity where the momentum is, and the partial
    // densities and the volume fraction stay in their own slots.
    template <std::size_t Dim>
    ConsArray<Dim> pack(const PrimState<Dim>& prim)
    {
        using Var = Layout<Dim>;

        ConsArray<Dim> w;
        w[Var::partial_rho(0)] = prim.partial_rho[0];
        w[Var::partial_rho(1)] = prim.partial_rho[1];
        w[Var::rhoE]           = prim.p;
        w[Var::alpha]          = prim.alpha;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            w[Var::mom(d)] = prim.v[d];
        }
        return w;
    }

    template <std::size_t Dim, class Array>
    PrimState<Dim> unpack(const Array& w)
    {
        using Var = Layout<Dim>;

        PrimState<Dim> prim;
        prim.partial_rho[0] = w[Var::partial_rho(0)];
        prim.partial_rho[1] = w[Var::partial_rho(1)];
        prim.p              = w[Var::rhoE];
        prim.alpha          = w[Var::alpha];
        for (std::size_t d = 0; d < Dim; ++d)
        {
            prim.v[d] = w[Var::mom(d)];
        }
        return prim;
    }

    template <std::size_t Dim, class Array>
    PrimState<Dim> cons2prim(const Array& q, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        PrimState<Dim> prim;
        prim.partial_rho[0] = q[Var::partial_rho(0)];
        prim.partial_rho[1] = q[Var::partial_rho(1)];
        prim.alpha          = q[Var::alpha];

        const double rho = prim.rho();
        double kinetic   = 0.;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            prim.v[d] = q[Var::mom(d)] / rho;
            kinetic += 0.5 * rho * prim.v[d] * prim.v[d];
        }
        prim.p = eos.p(prim.alpha, q[Var::rhoE] - kinetic);
        return prim;
    }

    template <std::size_t Dim>
    ConsArray<Dim> prim2cons(const PrimState<Dim>& prim, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        ConsArray<Dim> q;
        q[Var::partial_rho(0)] = prim.partial_rho[0];
        q[Var::partial_rho(1)] = prim.partial_rho[1];
        q[Var::alpha]          = prim.alpha;

        const double rho = prim.rho();
        q[Var::rhoE]     = eos.rho_e(prim.alpha, prim.p);
        for (std::size_t d = 0; d < Dim; ++d)
        {
            q[Var::mom(d)] = rho * prim.v[d];
            q[Var::rhoE] += 0.5 * rho * prim.v[d] * prim.v[d];
        }
        return q;
    }

    // A state the way a test case states it: each fluid at its own density, the
    // two sharing the cell in proportion alpha, at rest.
    template <std::size_t Dim>
    PrimState<Dim> mixture_state(double alpha, double rho0, double rho1, double p)
    {
        PrimState<Dim> prim;
        prim.alpha          = alpha;
        prim.partial_rho[0] = alpha * rho0;
        prim.partial_rho[1] = (1. - alpha) * rho1;
        prim.p              = p;
        prim.v.fill(0.);
        return prim;
    }

    // The two models agree on where the momentum sits; euler/bc.hpp relies on it.
    static_assert(Layout<1>::mom(0) == 2 && Layout<2>::mom(1) == 3 && Layout<3>::mom(2) == 4);
}
