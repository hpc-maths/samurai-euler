// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../../bc.hpp"
#include "../registry.hpp"
#include "../variables.hpp"

// =============================================================================
//  Water-air shock tube
// -----------------------------------------------------------------------------
//  Section 6.1.1 of the article this repository reproduces, and the case the
//  five-equation model exists for: a liquid at a gigapascal against air at one
//  atmosphere, the two separated by a contact that the solver has to keep sharp
//  while a rarefaction runs back into the water and a shock runs out into the
//  air.
//
//      water (phase 0)   rho = 1e3, P = 1e9,  gamma = 4.4, pi_inf = 6e8
//      air   (phase 1)   rho = 1,   P = 1e5,  gamma = 1.4, pi_inf = 0
//
//  Both at rest, outflow on both sides, t_f = 9e-4.
//
//  The article gives neither the domain nor the position of the diaphragm; its
//  fig. 11 gives both. The tube is [-2, 2] and the diaphragm is at x = 0.7, the
//  value this problem carries in the literature it comes from. The exact
//  solution below confirms it: at t_f the contact stands at 1.14, the head of
//  the rarefaction at -1.69 and its tail at -0.49, which is what the figure
//  shows.
//
//  It is a one-dimensional problem, run in two dimensions in the article, hence
//  the domain of the 2D case: [-2, 2] x [-0.4, 0.4], where level 6 is exactly
//  the 320 x 64 equivalent resolution it quotes. In one dimension the box is
//  [-2, 2] alone and samurai scales the cell length on its whole length, so
//  level 8 is 256 cells and level 9 is 512.
//
//  The exact solution is a stiffened-gas Riemann problem and is in
//  python/exact_two_phase_riemann.py: p* = 4.7969e5, u* = 491.97 m/s,
//  rho*_water = 800.3, rho*_air = 2.758. That is what the case is held to, not a
//  figure read by eye.
// =============================================================================

namespace test_case::water_air_shock_tube
{
    inline constexpr double diaphragm = 0.7;

    inline constexpr double water_rho = 1e3;
    inline constexpr double water_p   = 1e9;
    inline constexpr double air_rho   = 1.;
    inline constexpr double air_p     = 1e5;

    // Water is phase 0, air is phase 1; alpha is the volume fraction of water.
    inline const EOS::Mixture eos{
        {{4.4, 6e8, 0.}, {1.4, 0., 0.}}
    };

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-0.4);
        max_corner.fill(0.4);
        min_corner[0] = -2.;
        max_corner[0] = 2.;

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::Mixture eos_)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x = cell.center();

        // A pure fluid on each side: alpha is 1 in the water and 0 in the air,
        // and the partial density of the absent phase is exactly zero. Nothing
        // in the model divides by a volume fraction, so the pure states are
        // states like any other and do not need an epsilon.
        u[cell] = x[0] < diaphragm ? two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(1., water_rho, 0., water_p), eos_)
                                   : two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(0., 0., air_rho, air_p), eos_);
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

REGISTER_TWO_PHASE_CASE(water_air_shock_tube, test_case::water_air_shock_tube, 1, 2)
