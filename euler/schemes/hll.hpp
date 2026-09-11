// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "flux.hpp"

template <class Field, class Eos>
auto make_euler_hll(Eos eos)
{
    static constexpr std::size_t dim          = Field::dim;
    static constexpr std::size_t stencil_size = 2;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> hll;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            hll[d].cons_flux_function =
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

                if (sL >= 0)
                {
                    flux = compute_flux<d>(primL, eos);
                }
                else if (sL < 0 && sR > 0)
                {
                    flux = (sR * compute_flux<d>(primL, eos) - sL * compute_flux<d>(primR, eos) + sL * sR * (qR - qL)) / (sR - sL);
                }
                else if (sR <= 0)
                {
                    flux = compute_flux<d>(primR, eos);
                }
            };
        });
    auto scheme = make_flux_based_scheme(hll);
    scheme.set_name("hll");

    return scheme;
}