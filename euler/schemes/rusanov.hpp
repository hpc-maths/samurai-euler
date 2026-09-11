// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "flux.hpp"

template <class Field, class Eos>
auto make_euler_rusanov(const Eos& eos)
{
    static constexpr std::size_t dim          = Field::dim;
    static constexpr std::size_t stencil_size = 2;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> rusanov;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            rusanov[d].cons_flux_function =
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

                const auto lambda = std::max(std::abs(primL.v[d]) + cL, std::abs(primR.v[d]) + cR);

                flux = 0.5 * (compute_flux<d>(primL, eos) + compute_flux<d>(primR, eos) - lambda * (qR - qL));
            };
        });
    auto scheme = make_flux_based_scheme(rusanov);
    scheme.set_name("rusanov");

    return scheme;
}