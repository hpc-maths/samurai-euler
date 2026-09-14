// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>

#include <samurai/box.hpp>

#include "../../bc.hpp"
#include "../registry.hpp"
#include "../variables.hpp"

// =============================================================================
//  Air-helium shock-bubble interaction
// -----------------------------------------------------------------------------
//  Section 6.1.3 of the article this repository reproduces, and the one case in
//  it that is validated against an experiment rather than against another code:
//
//      J.-F. Haas, B. Sturtevant, "Interaction of weak shock waves with
//      cylindrical and spherical gas inhomogeneities", J. Fluid Mech. 181 (1987)
//      41-76, https://doi.org/10.1017/S0022112087002003
//
//  A Mach 1.22 shock runs leftwards down a 445 x 89 mm tube into a 25 mm bubble
//  of helium. The bubble is lighter than the air around it, so the shock crosses
//  it faster than it crosses the air, the refracted wave outruns the incident
//  one, the bubble collapses on itself and throws a jet downstream. Five wave
//  fronts come out of it, and their speeds are what the experiment measured and
//  what a solver can be held to.
//
//  Geometry, from fig. 13 of the article: the tube is [0, 0.445] x [0, 0.089] m,
//  the bubble is centred at x = 0.225 m on the axis with r = 25 mm, and the
//  shock starts at x = 0.275 m, which is 25 mm ahead of the bubble. Reflecting
//  walls top and bottom, the post-shock state imposed on the right, outflow on
//  the left.
//
//  The states are equation (8) of the article, which takes them from Jolgam et
//  al. The bubble is not pure helium: the experiment fills it with 72% helium
//  and 28% air by mass, and the mixture is modelled as one perfect gas of
//  gamma = 1.645, which is the second phase here. The first is air at 1.4.
//
//      (alpha_0 rho_0, alpha_1 rho_1, u, v, P, alpha_0) =
//          (1.6571, 0,      -114.51, 0, 159060, 1)     post-shock
//          (1.1839, 0,       0,      0, 101325, 1)     pre-shock
//          (0,      0.2193,  0,      0, 101325, 0)     bubble
//
//  NOTE that the post-shock density is not the one the Rankine-Hugoniot
//  relations give. At M = 1.22 the pressure ratio 159060 / 101325 = 1.5698 is
//  exact, and the velocity -114.51 is within a metre per second of it, but the
//  density behind the shock should be 1.6295 and not 1.6571. Taken literally the
//  initial discontinuity is therefore not a shock, and the speed it implies is
//  401 m/s where the pressure jump calls for 419. The solver settles that in the
//  first microseconds -- the discontinuity resolves into the shock the pressure
//  ratio asks for -- which is why the article's own 58 microseconds to impact
//  and its measured shock speed of 423 m/s are consistent with each other and
//  not with 401. The numbers are left exactly as the article gives them.
//
//  python/shock_bubble_waves.py extracts the five wave speeds from a run and
//  compares them with table 3 of the article and with the experiment.
// =============================================================================

namespace test_case::shock_bubble
{
    inline constexpr double length = 0.445;
    inline constexpr double height = 0.089;

    inline constexpr double bubble_x = 0.225;
    inline constexpr double bubble_r = 0.025;
    inline constexpr double shock_x  = 0.275;

    // Phase 0 is air, phase 1 the helium-air mixture of the bubble.
    inline const EOS::Mixture eos = EOS::two_ideal_gases(1.4, 1.645);

    inline constexpr double post_shock_rho = 1.6571;
    inline constexpr double post_shock_u   = -114.51;
    inline constexpr double post_shock_p   = 159060.;

    inline constexpr double air_rho    = 1.1839;
    inline constexpr double bubble_rho = 0.2193;
    inline constexpr double ambient_p  = 101325.;

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {length, height};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    auto post_shock_state()
    {
        auto state = two_phase::mixture_state<Field::dim>(1., post_shock_rho, 0., post_shock_p);
        state.v[0] = post_shock_u;
        return state;
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::Mixture eos_)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x = cell.center();

        const double dx      = x[0] - bubble_x;
        const double dy      = x[1] - height / 2.;
        const bool in_bubble = dx * dx + dy * dy < bubble_r * bubble_r;

        if (in_bubble)
        {
            u[cell] = two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(0., 0., bubble_rho, ambient_p), eos_);
        }
        else if (x[0] > shock_x)
        {
            u[cell] = two_phase::prim2cons<dim>(post_shock_state<Field>(), eos_);
        }
        else
        {
            u[cell] = two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(1., air_rho, 0., ambient_p), eos_);
        }
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::Mixture eos_)
    {
        static constexpr std::size_t dim = Field::dim;

        const xt::xtensor_fixed<int, xt::xshape<dim>> left   = {-1, 0};
        const xt::xtensor_fixed<int, xt::xshape<dim>> right  = {1, 0};
        const xt::xtensor_fixed<int, xt::xshape<dim>> bottom = {0, -1};
        const xt::xtensor_fixed<int, xt::xshape<dim>> top    = {0, 1};

        // Walls above and below, which is what keeps the interaction the
        // two-dimensional one the experiment photographs; the shock keeps
        // arriving from the right, and everything leaves through the left.
        //
        // Each side is named: samurai attaches boundary conditions rather than
        // replacing them, so a wall on every side plus an inflow on one would
        // leave that side carrying both, and which of the two wins would be the
        // order they happened to be attached in.
        bc::wall(u)->on(bottom, top);
        bc::imposed_state(u, two_phase::prim2cons<dim>(post_shock_state<Field>(), eos_))->on(right);
        bc::outflow(u)->on(left);
    }

    template <class Field>
    two_phase::test_case_t<Field::dim> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = eos};
    }
}

REGISTER_TWO_PHASE_CASE(shock_bubble, test_case::shock_bubble, 2)
