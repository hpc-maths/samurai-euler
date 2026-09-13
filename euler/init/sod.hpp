// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Sod shock tube, rotated by 45 degrees
// -----------------------------------------------------------------------------
//  The classical 1D Riemann problem, laid out along a diagonal of a 2D domain so
//  that the solution is not aligned with the mesh. Any directional bias in the
//  scheme or in the adaptation shows up as a distortion of what should stay a
//  planar wave.
//
//  The states are those of
//
//      G.A. Sod, "A survey of several finite difference methods for systems of
//      nonlinear hyperbolic conservation laws", J. Comput. Phys. 27 (1) (1978)
//      1-31, https://doi.org/10.1016/0021-9991(78)90023-2
//
//  reprinted as test 1 of Table 4.1 of
//
//      E.F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics.
//      A Practical Introduction", 3rd ed., Springer, 2009,
//      https://doi.org/10.1007/b79761
//
//  whose exact solution python/exact_riemann.py computes: p* = 0.30313,
//  u* = 0.92745. The 45 degree rotation is ours, not Sod's; the article this
//  repository reproduces uses the axis-aligned tube.
// =============================================================================

namespace test_case::sod
{
    using field_t = config<2>::field_t;

    inline const double theta = std::numbers::pi / 4.;

    inline const double Rdx = std::sin(theta);
    inline const double Rdy = std::cos(theta);
    inline const double k   = 0.5 / Rdy;
    inline const double x0  = 0.5 + k * Rdx;

    inline const PrimState<2> left_state{
        1.,
        1.,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline const PrimState<2> right_state{
        0.125,
        0.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        const double y_theta = (x0 - x[0]) * Rdy / Rdx;

        if (x[1] < y_theta)
        {
            u[cell] = prim2cons<2>(left_state, eos);
        }
        else
        {
            u[cell] = prim2cons<2>(right_state, eos);
        }
    }

    inline void bc_fn(field_t& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(sod, test_case::sod, 2)
