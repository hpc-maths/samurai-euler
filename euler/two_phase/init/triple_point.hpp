// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../../bc.hpp"
#include "../registry.hpp"
#include "../variables.hpp"

// =============================================================================
//  Triple point, two gases
// -----------------------------------------------------------------------------
//  Section 6.1.2 of the article this repository reproduces, this time as the
//  article poses it: three regions carrying two gases, gamma = 1.5 in regions 1
//  and 3 and gamma = 1.4 in region 2, under the five-equation model.
//
//      region 1  [0,1[ x [0,3[      alpha_0 = 1, rho = 1,     P = 1
//      region 2  [1,7[ x [0,1.5[    alpha_0 = 0, rho = 1,     P = 0.1
//      region 3  [1,7[ x [1.5,3[    alpha_0 = 1, rho = 0.125, P = 0.1
//
//  Domain [0,7] x [0,3], t_f = 2.0, walls all around, everything at rest. The
//  left region drives a shock into the two others; they carry it at different
//  speeds, the contact between them shears, and a Kelvin-Helmholtz roll-up grows
//  around the point where the three meet.
//
//  `triple_point_single_gamma` in the monofluid solver is the same geometry with
//  one gas, and says at length that it is not this case. This one is.
// =============================================================================

namespace test_case::triple_point
{
    inline constexpr double x_wall = 1.0; // between region 1 and the other two
    inline constexpr double y_wall = 1.5; // between regions 2 and 3

    // Phase 0 is the gas of regions 1 and 3, phase 1 the gas of region 2.
    inline const EOS::Mixture eos = EOS::two_ideal_gases(1.5, 1.4);

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {7., 3.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::Mixture eos_)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x = cell.center();

        if (x[0] < x_wall)
        {
            u[cell] = two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(1., 1., 0., 1.), eos_);
        }
        else if (x[1] < y_wall)
        {
            u[cell] = two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(0., 0., 1., 0.1), eos_);
        }
        else
        {
            u[cell] = two_phase::prim2cons<dim>(two_phase::mixture_state<dim>(1., 0.125, 0., 0.1), eos_);
        }
    }

    // Solid walls: the problem is posed in a closed box, the shock reflects off
    // the right-hand wall well after t_f, and the top and bottom walls are what
    // keeps the roll-up two-dimensional.
    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::Mixture /*eos*/)
    {
        bc::wall(u);
    }

    template <class Field>
    two_phase::test_case_t<Field::dim> definition()
    {
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = eos};
    }
}

REGISTER_TWO_PHASE_CASE(triple_point, test_case::triple_point, 2)
