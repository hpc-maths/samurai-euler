// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include <samurai/schemes/fv.hpp>

#include "../reconstruction.hpp"
#include "riemann.hpp"
#include "variables.hpp"

// =============================================================================
//  The two-phase scheme, at both orders
// -----------------------------------------------------------------------------
//  Same shape as the monofluid one: a first-order Godunov scheme on a stencil of
//  two, and a MUSCL reconstruction with a Hancock predictor on a stencil of
//  four. What differs is the fifth equation.
//
//  Every flux function here is a `flux_function`, not a `cons_flux_function`:
//  samurai's conservative form applies +F to the cell on the left of an
//  interface and -F to the one on the right, and the volume fraction does not
//  obey that. Its contribution is
//
//      left cell    + u* (alpha* - alpha_L)
//      right cell   - u* (alpha* - alpha_R)
//
//  which is the same u* alpha* flux on both sides, each corrected by the cell's
//  own volume fraction. The four conservation laws take the conservative pair,
//  so mass, momentum and energy are conserved to the last bit; alpha is not
//  conserved and must not be.
//
//  The Hancock predictor is the monofluid one with two more equations:
//
//      d_t alpha + u_d d_x alpha                     = 0
//      d_t m_k   + u_d d_x m_k   + m_k d_x u_d       = 0
//      d_t u_i   + u_d d_x u_i   + [i == d] d_x p / rho = 0
//      d_t p     + u_d d_x p     + rho c^2 d_x u_d   = 0
//
//  where m_k = alpha_k rho_k. Only the normal direction appears, for the same
//  reason as in the monofluid case, and Strang splitting is what makes that
//  exact rather than approximate.
// =============================================================================

namespace two_phase
{
    struct SchemeOptions
    {
        SlopeLimiter limiter = SlopeLimiter::moncen;
        bool hancock         = true;
        std::shared_ptr<const double> dt;
        int direction = -1;
    };

    // The primitive form of the system applied to a slope, i.e. A_d(W) dW.
    template <std::size_t d, std::size_t Dim, class Array>
    ConsArray<Dim> primitive_jacobian_times(const PrimState<Dim>& prim, const Array& slope, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        const double rho = prim.rho();
        const double c   = eos.c(prim.alpha, rho, prim.p);
        const double u_d = prim.v[d];

        const double dp  = slope[Var::rhoE];
        const double dud = slope[Var::mom(d)];

        ConsArray<Dim> out;
        out[Var::alpha] = u_d * slope[Var::alpha];
        for (std::size_t k = 0; k < 2; ++k)
        {
            out[Var::partial_rho(k)] = u_d * slope[Var::partial_rho(k)] + prim.partial_rho[k] * dud;
        }
        out[Var::rhoE] = u_d * dp + rho * c * c * dud;
        for (std::size_t i = 0; i < Dim; ++i)
        {
            out[Var::mom(i)] = u_d * slope[Var::mom(i)];
        }
        out[Var::mom(d)] += dp / rho;

        return out;
    }

    // A reconstructed state is usable only if both partial densities are
    // positive, the volume fraction stays in [0, 1] and the pressure is above
    // the vacuum of the mixture. Otherwise that face falls back to the cell
    // average and is first order there.
    template <std::size_t Dim>
    bool admissible(const ConsArray<Dim>& w, const EOS::Mixture& eos)
    {
        using Var = Layout<Dim>;

        const double alpha = w[Var::alpha];
        if (alpha < 0. || alpha > 1.)
        {
            return false;
        }
        if (w[Var::partial_rho(0)] < 0. || w[Var::partial_rho(1)] < 0.)
        {
            return false;
        }
        return w[Var::rhoE] + eos.pi_inf(alpha) > 0.;
    }

    // The interface contribution of the two cells it separates: conservative for
    // the four conservation laws, corrected by the cell's own volume fraction
    // for the fifth.
    template <std::size_t Dim, class FluxPair, class Interface>
    void split_contribution(FluxPair& fluxes, const Interface& face, double alpha_left, double alpha_right)
    {
        fluxes[0] = face.flux;
        fluxes[1] = -face.flux;

        fluxes[0][Layout<Dim>::alpha] = face.flux[Layout<Dim>::alpha] - alpha_left * face.u_star;
        fluxes[1][Layout<Dim>::alpha] = -(face.flux[Layout<Dim>::alpha] - alpha_right * face.u_star);
    }

    template <riemann::Solver solver, class Field>
    auto make_godunov_scheme(const EOS::Mixture& eos, int direction = -1)
    {
        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t stencil_size = 2;

        using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

        samurai::FluxDefinition<cfg> godunov;

        samurai::static_for<0, dim>::apply(
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                if (direction >= 0 && d != static_cast<std::size_t>(direction))
                {
                    return;
                }

                godunov[d].flux_function = [eos](samurai::FluxValuePair<cfg>& fluxes,
                                                 const samurai::StencilData<cfg>& /*data*/,
                                                 const samurai::StencilValues<cfg>& field)
                {
                    const auto& qL = field[0];
                    const auto& qR = field[1];

                    const auto face = riemann::solve<solver, d>(cons2prim<dim>(qL, eos), qL, cons2prim<dim>(qR, eos), qR, eos);

                    split_contribution<dim>(fluxes, face, qL[Layout<dim>::alpha], qR[Layout<dim>::alpha]);
                };
            });

        auto scheme = make_flux_based_scheme(godunov);
        scheme.set_name("two-phase godunov");

        return scheme;
    }

    template <riemann::Solver solver, class Field>
    auto make_muscl_scheme(const EOS::Mixture& eos, const SchemeOptions& options)
    {
        assert((!options.hancock || options.dt) && "the Hancock predictor needs a time step to read");

        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t stencil_size = 4;
        static constexpr std::size_t n_comp       = Layout<dim>::size;

        using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

        samurai::FluxDefinition<cfg> muscl;

        samurai::static_for<0, dim>::apply(
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                if (options.direction >= 0 && d != static_cast<std::size_t>(options.direction))
                {
                    return;
                }

                muscl[d].flux_function = [eos, options](samurai::FluxValuePair<cfg>& fluxes,
                                                        const samurai::StencilData<cfg>& data,
                                                        const samurai::StencilValues<cfg>& field)
                {
                    const auto w0 = pack<dim>(cons2prim<dim>(field[0], eos));
                    const auto w1 = pack<dim>(cons2prim<dim>(field[1], eos));
                    const auto w2 = pack<dim>(cons2prim<dim>(field[2], eos));
                    const auto w3 = pack<dim>(cons2prim<dim>(field[3], eos));

                    const auto slopeL = limited_slope<n_comp>(w1 - w0, w2 - w1, options.limiter);
                    const auto slopeR = limited_slope<n_comp>(w2 - w1, w3 - w2, options.limiter);

                    ConsArray<dim> wL = w1 + 0.5 * slopeL;
                    ConsArray<dim> wR = w2 - 0.5 * slopeR;

                    if (options.hancock)
                    {
                        const double half = 0.5 * (*options.dt) / data.cell_length;
                        wL -= half * primitive_jacobian_times<d>(unpack<dim>(w1), slopeL, eos);
                        wR -= half * primitive_jacobian_times<d>(unpack<dim>(w2), slopeR, eos);
                    }

                    const auto primL = unpack<dim>(admissible<dim>(wL, eos) ? wL : w1);
                    const auto primR = unpack<dim>(admissible<dim>(wR, eos) ? wR : w2);

                    const auto face = riemann::solve<solver, d>(primL, prim2cons<dim>(primL, eos), primR, prim2cons<dim>(primR, eos), eos);

                    // The non-conservative term is alpha at the cell centre times
                    // the divergence of the velocity, so it is the cell average
                    // that appears here and not the reconstructed face value.
                    split_contribution<dim>(fluxes, face, field[1][Layout<dim>::alpha], field[2][Layout<dim>::alpha]);
                };
            });

        auto scheme = make_flux_based_scheme(muscl);
        scheme.set_name("two-phase muscl");

        return scheme;
    }

    template <class Field>
    auto make_first_order_scheme(const std::string& name, const EOS::Mixture& eos, int direction = -1)
    {
        switch (riemann::from_name(name))
        {
            case riemann::Solver::rusanov:
                return make_godunov_scheme<riemann::Solver::rusanov, Field>(eos, direction);
            case riemann::Solver::hll:
                return make_godunov_scheme<riemann::Solver::hll, Field>(eos, direction);
            default:
                return make_godunov_scheme<riemann::Solver::hllc, Field>(eos, direction);
        }
    }

    template <class Field>
    auto make_second_order_scheme(const std::string& name, const EOS::Mixture& eos, const SchemeOptions& options)
    {
        switch (riemann::from_name(name))
        {
            case riemann::Solver::rusanov:
                return make_muscl_scheme<riemann::Solver::rusanov, Field>(eos, options);
            case riemann::Solver::hll:
                return make_muscl_scheme<riemann::Solver::hll, Field>(eos, options);
            default:
                return make_muscl_scheme<riemann::Solver::hllc, Field>(eos, options);
        }
    }
}
