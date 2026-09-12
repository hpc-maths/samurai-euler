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
//  Double Mach reflection
// -----------------------------------------------------------------------------
//  A Mach 10 shock hits a 30 degree ramp. The ramp is represented by a reflecting
//  wall along the bottom boundary starting at x0, the shock entering upstream of
//  it. The top boundary follows the analytic shock position, which is why the
//  boundary condition needs the current time.
// =============================================================================

namespace test_case::double_mach_reflection
{
    using field_t = config<2>::field_t;

    inline const double alpha = std::numbers::pi / 3.;
    inline const double x0    = 1. / 6;

    inline const PrimState<2> left_state{
        8.,
        116.5,
        xt::xtensor_fixed<double, xt::xshape<2>>{8.25 * std::sin(alpha), -8.25 * std::cos(alpha)}
    };

    inline const PrimState<2> right_state{
        1.4,
        1.,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        if (x[0] < x0 + x[1] / std::tan(alpha))
        {
            u[cell] = prim2cons<2>(left_state, eos);
        }
        else
        {
            u[cell] = prim2cons<2>(right_state, eos);
        }
    }

    inline void bc_fn(field_t& u, double& t, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = field_t::dim;
        using EulerConsVar               = EulerLayout<dim>;

        const xt::xtensor_fixed<int, xt::xshape<dim>> bottom = {0, -1};
        const xt::xtensor_fixed<int, xt::xshape<dim>> top    = {0, 1};
        const xt::xtensor_fixed<int, xt::xshape<dim>> right  = {1, 0};
        const xt::xtensor_fixed<int, xt::xshape<dim>> left   = {-1, 0};

        // Bottom: post-shock state upstream of the ramp foot, reflecting wall
        // downstream of it. This one stays a value function: expressing it as two
        // conditions restricted by coordinates is what samurai's CoordsRegion is
        // for, but that path throws on this non-square domain.
        bc::imposed(u,
                    [&u, eos](const auto&, const auto& cell, const auto&)
                    {
                        if (cell.center(0) < x0)
                        {
                            return prim2cons<2>(left_state, eos);
                        }
                        else
                        {
                            return xt::xtensor_fixed<double, xt::xshape<dim + 2>>{u[cell][EulerConsVar::rho],
                                                                                  u[cell][EulerConsVar::rhoE],
                                                                                  u[cell][EulerConsVar::mom(0)],
                                                                                  -u[cell][EulerConsVar::mom(1)]};
                        }
                    })
            ->on(bottom);

        // Top: follows the analytic shock position, hence the dependence on t.
        bc::imposed(u,
                    [&t, eos](const auto&, const auto& cell, const auto&)
                    {
                        const double x1 = x0 + 10 * t / std::sin(alpha) + 1 / std::tan(alpha);
                        if (cell.center(0) < x1)
                        {
                            return prim2cons<2>(left_state, eos);
                        }
                        else
                        {
                            return prim2cons<2>(right_state, eos);
                        }
                    })
            ->on(top);

        // Right: outflow. Left: the incoming post-shock state.
        bc::outflow(u)->on(right);

        bc::imposed(u, left_state, eos)->on(left);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {4., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(double_mach_reflection, test_case::double_mach_reflection, 2)
