// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/bc.hpp>
#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Double rarefaction (the "123 problem")
// -----------------------------------------------------------------------------
//  Two halves of the same gas move apart at the same speed. No shock forms: two
//  rarefactions run outwards and leave a near-vacuum region behind them, where
//  the internal energy is a small difference of two large numbers.
//
//  That is what the case is for. A scheme that is not positivity preserving
//  produces a negative pressure in the middle and the sound speed turns NaN, so
//  it fails loudly rather than quietly. It is the initial state euler_1d used
//  before the test cases were shared, kept here as its default.
//
//  Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", test 2.
// =============================================================================

namespace test_case::double_rarefaction
{
    using field_t = config<1>::field_t;

    inline constexpr double x0 = 0.5; // where the two halves meet

    inline const PrimState<1> left_state{1., 0.4, xt::xtensor_fixed<double, xt::xshape<1>>{-2.}};

    inline const PrimState<1> right_state{1., 0.4, xt::xtensor_fixed<double, xt::xshape<1>>{2.}};

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        u[cell] = prim2cons<1>(x[0] < x0 ? left_state : right_state, eos);
    }

    inline void bc_fn(field_t& u, double& /*t*/, EOS::IdealGas eos)
    {
        const xt::xtensor_fixed<int, xt::xshape<1>> left  = {-1};
        const xt::xtensor_fixed<int, xt::xshape<1>> right = {1};

        // Dirichlet rather than an imposed ghost state: the rarefactions never
        // reach the boundaries, so holding the initial state there is exact.
        using EulerConsVar = EulerLayout<1>;

        const auto cons_left  = prim2cons<1>(left_state, eos);
        const auto cons_right = prim2cons<1>(right_state, eos);

        samurai::make_bc<samurai::Dirichlet<1>>(u, cons_left[EulerConsVar::rho], cons_left[EulerConsVar::rhoE], cons_left[EulerConsVar::mom(0)])
            ->on(left);
        samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                cons_right[EulerConsVar::rho],
                                                cons_right[EulerConsVar::rhoE],
                                                cons_right[EulerConsVar::mom(0)])
            ->on(right);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 1, "this test case is one-dimensional");
        return {.box = &box_fn<1>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(double_rarefaction, test_case::double_rarefaction, 1)
