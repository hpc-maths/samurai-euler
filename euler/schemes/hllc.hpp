// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "flux.hpp"

template <std::size_t d, std::size_t Dim, class Eos>
auto compute_star_state(const PrimState<Dim>& prim, double s, double s_star, const Eos& eos)
{
    using EulerConsVar = EulerLayout<Dim>;
    xt::xtensor_fixed<double, xt::xshape<EulerConsVar::size>> q_star;

    auto rho_star = prim.rho * (s - prim.v[d]) / (s - s_star);

    q_star[EulerConsVar::rho] = rho_star;

    auto e                     = eos.e(prim.rho, prim.p);
    q_star[EulerConsVar::rhoE] = rho_star * (e + (s_star - prim.v[d]) * (s_star + prim.p / (prim.rho * (s - prim.v[d]))));

    for (std::size_t i = 0; i < Dim; ++i)
    {
        q_star[EulerConsVar::mom(i)] = rho_star * prim.v[i];
        q_star[EulerConsVar::rhoE] += 0.5 * rho_star * prim.v[i] * prim.v[i];
    }
    q_star[EulerConsVar::mom(d)] = rho_star * s_star;

    return q_star;
}

template <class Field, class Eos>
auto make_euler_hllc(const Eos& eos)
{
    static constexpr std::size_t dim          = Field::dim;
    static constexpr std::size_t stencil_size = 2;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> hllc;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            hllc[d].cons_flux_function =
                [eos](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& /*data*/, const samurai::StencilValues<cfg>& field)
            {
                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                const auto& qL = field[left];
                auto primL     = cons2prim<dim>(qL, eos);
                auto cL        = eos.c(primL.rho, primL.p);

                const auto& qR = field[right];
                auto primR     = cons2prim<dim>(qR, eos);
                auto cR        = eos.c(primR.rho, primR.p);

                double sL = std::min(primL.v[d] - cL, primR.v[d] - cR);
                double sR = std::max(primL.v[d] + cL, primR.v[d] + cR);
                double sM = (primL.rho * primL.v[d] * (sL - primL.v[d]) - primL.p - primR.rho * primR.v[d] * (sR - primR.v[d]) + primR.p)
                          / (primL.rho * (sL - primL.v[d]) - primR.rho * (sR - primR.v[d]));

                if (sL >= 0)
                {
                    flux = compute_flux<d>(primL, eos);
                }
                else if (sL < 0 && sM >= 0)
                {
                    flux = compute_flux<d>(primL, eos) + sL * (compute_star_state<d>(primL, sL, sM, eos) - qL);
                }
                else if (sM < 0 && sR >= 0)
                {
                    flux = compute_flux<d>(primR, eos) + sR * (compute_star_state<d>(primR, sR, sM, eos) - qR);
                }
                else if (sR < 0)
                {
                    flux = compute_flux<d>(primR, eos);
                }
            };
        });

    auto scheme = make_flux_based_scheme(hllc);
    scheme.set_name("hllc");

    return scheme;
}
