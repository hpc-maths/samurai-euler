// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "variables.hpp"

// =============================================================================
//  Riemann solvers for the five-equation model
// -----------------------------------------------------------------------------
//  Four of the five equations are conservation laws and are solved the way the
//  monofluid ones are: wave speeds from the mixture sound speed, and a flux
//  built from the states on either side. The mixture behaves as one stiffened
//  gas whose coefficients depend on alpha, so the algebra is the monofluid one
//  with rho = alpha_0 rho_0 + alpha_1 rho_1 and the two partial densities
//  carried along as passive masses through the contact.
//
//  The fifth is not a conservation law, and that is the whole difficulty:
//
//      d_t alpha + u . grad(alpha) = 0
//
//  Written as div(alpha u) - alpha div(u), it is discretized the way the article
//  does it, with the contact velocity u* the Riemann solver already computes:
//
//      alpha_i^{n+1} = alpha_i - dt/dx [ (u* alpha*)_{i+1/2} - (u* alpha*)_{i-1/2}
//                                        - alpha_i (u*_{i+1/2} - u*_{i-1/2}) ]
//
//  where alpha* is upwinded by the sign of u*. The term in alpha_i is why the
//  interface contribution differs between the two cells it separates: samurai
//  calls that a non-conservative flux and takes a pair of values, one per cell,
//  which is exactly what `solve` returns here. The cell value alpha_i is used,
//  not the reconstructed face value: what the term stands for is alpha at the
//  centre times the divergence of u, and a face value there would cost the
//  second order rather than buy anything.
//
//  A note on what the alpha equation may NOT be: solved as the conservation law
//  d_t alpha + div(alpha u) = 0. That form is only equivalent for a divergence
//  free velocity, and compressing a mixture would then create volume fraction
//  out of nothing.
// =============================================================================

namespace two_phase::riemann
{
    enum class Solver
    {
        rusanov,
        hll,
        hllc
    };

    inline Solver from_name(const std::string& name)
    {
        if (name == "rusanov")
        {
            return Solver::rusanov;
        }
        if (name == "hll")
        {
            return Solver::hll;
        }
        if (name == "hllc")
        {
            return Solver::hllc;
        }
        throw std::runtime_error("Unknown scheme: " + name);
    }

    // What an interface hands back: the conservative flux of the four
    // conservation laws, and the contact velocity the volume fraction is
    // advected with. The alpha slot of `flux` holds u* alpha*, the flux part of
    // the advection; the cell-dependent part is added by the scheme.
    template <std::size_t Dim>
    struct Interface
    {
        ConsArray<Dim> flux;
        double u_star;
    };

    // The physical flux in direction d.
    template <std::size_t d, std::size_t Dim>
    ConsArray<Dim> physical_flux(const PrimState<Dim>& prim, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        const double rho = prim.rho();
        const double u_d = prim.v[d];

        ConsArray<Dim> flux;
        flux[Var::partial_rho(0)] = prim.partial_rho[0] * u_d;
        flux[Var::partial_rho(1)] = prim.partial_rho[1] * u_d;

        double kinetic = 0.;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            flux[Var::mom(i)] = rho * prim.v[i] * u_d;
            kinetic += 0.5 * rho * prim.v[i] * prim.v[i];
        }
        flux[Var::mom(d)] += prim.p;

        const double E  = eos.rho_e(prim.alpha, prim.p) + kinetic;
        flux[Var::rhoE] = (E + prim.p) * u_d;

        // The flux part of the advection equation; the rest is not a flux.
        flux[Var::alpha] = prim.alpha * u_d;
        return flux;
    }

    // Wave speeds, bounding every wave of both sides (Davis).
    template <std::size_t d, std::size_t Dim>
    std::pair<double, double> wave_speeds(const PrimState<Dim>& primL, const PrimState<Dim>& primR, const EOS::Mixture& eos)
    {
        const double cL = eos.c(primL.alpha, primL.rho(), primL.p);
        const double cR = eos.c(primR.alpha, primR.rho(), primR.p);

        return {std::min(primL.v[d] - cL, primR.v[d] - cR), std::max(primL.v[d] + cL, primR.v[d] + cR)};
    }

    // The contact speed, from the two states across the interface.
    template <std::size_t d, std::size_t Dim>
    double contact_speed(const PrimState<Dim>& primL, const PrimState<Dim>& primR, double sL, double sR)
    {
        const double rhoL = primL.rho();
        const double rhoR = primR.rho();

        return (rhoL * primL.v[d] * (sL - primL.v[d]) - primL.p - rhoR * primR.v[d] * (sR - primR.v[d]) + primR.p)
             / (rhoL * (sL - primL.v[d]) - rhoR * (sR - primR.v[d]));
    }

    template <std::size_t d, std::size_t Dim, class ConsK>
    Interface<Dim>
    rusanov(const PrimState<Dim>& primL, const ConsK& qL, const PrimState<Dim>& primR, const ConsK& qR, const EOS::Mixture& eos)
    {
        const auto [sL, sR] = wave_speeds<d>(primL, primR, eos);
        const double lambda = std::max(std::abs(sL), std::abs(sR));

        Interface<Dim> face;
        face.flux   = 0.5 * (physical_flux<d>(primL, eos) + physical_flux<d>(primR, eos) - lambda * (qR - qL));
        face.u_star = 0.5 * (primL.v[d] + primR.v[d]);
        return face;
    }

    template <std::size_t d, std::size_t Dim, class ConsK>
    Interface<Dim> hll(const PrimState<Dim>& primL, const ConsK& qL, const PrimState<Dim>& primR, const ConsK& qR, const EOS::Mixture& eos)
    {
        const auto [sL, sR] = wave_speeds<d>(primL, primR, eos);

        Interface<Dim> face;
        face.u_star = contact_speed<d>(primL, primR, sL, sR);

        if (sL >= 0.)
        {
            face.flux = physical_flux<d>(primL, eos);
        }
        else if (sR <= 0.)
        {
            face.flux = physical_flux<d>(primR, eos);
        }
        else
        {
            face.flux = (sR * physical_flux<d>(primL, eos) - sL * physical_flux<d>(primR, eos) + sL * sR * (qR - qL)) / (sR - sL);
        }
        return face;
    }

    // The state between the wave of speed s and the contact. The two partial
    // densities go through as the mixture density does: the contact carries
    // them, it does not mix them.
    template <std::size_t d, std::size_t Dim>
    ConsArray<Dim> star_state(const PrimState<Dim>& prim, double s, double s_star, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        const double rho    = prim.rho();
        const double factor = (s - prim.v[d]) / (s - s_star);

        ConsArray<Dim> q_star;
        q_star[Var::partial_rho(0)] = prim.partial_rho[0] * factor;
        q_star[Var::partial_rho(1)] = prim.partial_rho[1] * factor;
        q_star[Var::alpha]          = prim.alpha;

        const double rho_star = rho * factor;
        double kinetic        = 0.;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            q_star[Var::mom(i)] = rho_star * prim.v[i];
            kinetic += 0.5 * prim.v[i] * prim.v[i];
        }
        q_star[Var::mom(d)] = rho_star * s_star;

        const double e    = (eos.rho_e(prim.alpha, prim.p) / rho) + kinetic;
        q_star[Var::rhoE] = rho_star * (e + (s_star - prim.v[d]) * (s_star + prim.p / (rho * (s - prim.v[d]))));

        return q_star;
    }

    template <std::size_t d, std::size_t Dim, class ConsK>
    Interface<Dim> hllc(const PrimState<Dim>& primL, const ConsK& qL, const PrimState<Dim>& primR, const ConsK& qR, const EOS::Mixture& eos)
    {
        const auto [sL, sR] = wave_speeds<d>(primL, primR, eos);
        const double sM     = contact_speed<d>(primL, primR, sL, sR);

        Interface<Dim> face;
        face.u_star = sM;

        if (sL >= 0.)
        {
            face.flux = physical_flux<d>(primL, eos);
        }
        else if (sM >= 0.)
        {
            face.flux = physical_flux<d>(primL, eos) + sL * (star_state<d>(primL, sL, sM, eos) - qL);
        }
        else if (sR >= 0.)
        {
            face.flux = physical_flux<d>(primR, eos) + sR * (star_state<d>(primR, sR, sM, eos) - qR);
        }
        else
        {
            face.flux = physical_flux<d>(primR, eos);
        }
        return face;
    }

    template <Solver s, std::size_t d, std::size_t Dim, class ConsK>
    Interface<Dim> solve(const PrimState<Dim>& primL, const ConsK& qL, const PrimState<Dim>& primR, const ConsK& qR, const EOS::Mixture& eos)
    {
        Interface<Dim> face;
        if constexpr (s == Solver::rusanov)
        {
            face = rusanov<d>(primL, qL, primR, qR, eos);
        }
        else if constexpr (s == Solver::hll)
        {
            face = hll<d>(primL, qL, primR, qR, eos);
        }
        else
        {
            face = hllc<d>(primL, qL, primR, qR, eos);
        }

        // The volume fraction crossing the interface is the one the contact
        // brings, whatever the conservative flux above did with its slot.
        const double alpha_up         = face.u_star >= 0. ? primL.alpha : primR.alpha;
        face.flux[Layout<Dim>::alpha] = face.u_star * alpha_up;

        return face;
    }
}
