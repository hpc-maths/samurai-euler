// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../../bc.hpp"
#include "../registry.hpp"
#include "../variables.hpp"

// =============================================================================
//  Sod's tube, as a two-phase state that is one fluid everywhere
// -----------------------------------------------------------------------------
//  The same problem as `sod_x` in the monofluid solver: the unit tube, the
//  diaphragm at 0.5, the states of Sod, an ideal gas at gamma = 1.4. What makes
//  it worth registering here is the second phase, which is present in the model
//  and absent from the flow: alpha_0 is 1 everywhere and the partial density of
//  phase 1 is exactly zero.
//
//  A five-equation model must then reproduce the monofluid solver, and the test
//  suite holds it to that to twelve digits. It is the sharpest statement
//  available about the two-phase kernel, because it compares it against code
//  that is already held to an exact solution rather than against itself: the
//  mixture equation of state at alpha = 1 has to give back phase 0 alone, the
//  volume fraction has to stay exactly 1 through a shock and a rarefaction, and
//  every wave has to land where the monofluid solver puts it.
//
//  The second phase is water, as far from an ideal gas at 1.4 as this repository
//  has: if any part of the mixture law leaks the absent phase into the answer,
//  it leaks a gigapascal.
// =============================================================================

namespace test_case::sod_x_pure
{
    inline constexpr double diaphragm = 0.5;

    // Phase 0 is the gas the tube is filled with; phase 1 is there and empty.
    inline const EOS::Mixture eos{
        {{1.4, 0., 0.}, {4.4, 6e8, 0.}}
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

        const auto state = cell.center(0) < diaphragm ? two_phase::mixture_state<dim>(1., 1., 0., 1.)
                                                      : two_phase::mixture_state<dim>(1., 0.125, 0., 0.1);
        u[cell]          = two_phase::prim2cons<dim>(state, eos_);
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::Mixture /*eos*/)
    {
        bc::outflow(u);
    }

    template <class Field>
    two_phase::test_case_t<Field::dim> definition()
    {
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = eos};
    }
}

REGISTER_TWO_PHASE_CASE(sod_x_pure, test_case::sod_x_pure, 1, 2)
