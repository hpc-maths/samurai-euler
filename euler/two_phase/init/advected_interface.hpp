// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../registry.hpp"
#include "../variables.hpp"

// =============================================================================
//  An interface carried by a uniform flow
// -----------------------------------------------------------------------------
//  A slab of water in air, everything at one atmosphere and moving at the same
//  speed. The exact solution is the initial state translated: the pressure and
//  the velocity stay uniform for ever, and only the volume fraction moves.
//
//  This is the test the five-equation model is designed to pass and the reason
//  it exists. A model that advects the volume fraction inconsistently with the
//  masses and the energy produces a pressure spike at the interface out of
//  nothing -- the classic failure of a conservative two-material scheme, where
//  averaging two gases in one cell gives a mixture whose pressure is not the
//  pressure of either. The interface here is between fluids whose stiffened-gas
//  coefficients could hardly be further apart, so nothing hides.
//
//      P. Kapila et al. / R. Abgrall, "How to prevent pressure oscillations in
//      multicomponent flow calculations: a quasi conservative approach",
//      J. Comput. Phys. 125 (1996) 150-160,
//      https://doi.org/10.1006/jcph.1996.0085
//
//  Domain [0,1], periodic, water in [0.3, 0.7], p = 1e5 and u = 100 everywhere.
// =============================================================================

namespace test_case::advected_interface
{
    inline constexpr double left  = 0.3;
    inline constexpr double right = 0.7;

    inline constexpr double pressure = 1e5;
    inline constexpr double speed    = 100.;

    inline constexpr double water_rho = 1e3;
    inline constexpr double air_rho   = 1.;

    inline const EOS::Mixture eos{
        {{4.4, 6e8, 0.}, {1.4, 0., 0.}}
    };

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(0.);
        max_corner.fill(1.);

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::Mixture eos_)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x     = cell.center();
        const bool water = x[0] > left && x[0] < right;

        auto state = water ? two_phase::mixture_state<dim>(1., water_rho, 0., pressure)
                           : two_phase::mixture_state<dim>(0., 0., air_rho, pressure);
        state.v[0] = speed;

        u[cell] = two_phase::prim2cons<dim>(state, eos_);
    }

    template <class Field>
    void bc_fn(Field& /*u*/, double& /*t*/, EOS::Mixture /*eos*/)
    {
    }

    template <class Field>
    two_phase::test_case_t<Field::dim> definition()
    {
        auto definition     = two_phase::test_case_t<Field::dim>{.box  = &box_fn<Field::dim>,
                                                                 .init = &init_fn<Field>,
                                                                 .bc   = &bc_fn<Field>,
                                                                 .eos  = eos};
        definition.periodic = {};
        definition.periodic.fill(true);
        return definition;
    }
}

REGISTER_TWO_PHASE_CASE(advected_interface, test_case::advected_interface, 1, 2)
