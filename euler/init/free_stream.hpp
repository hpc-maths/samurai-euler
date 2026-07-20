// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/bc.hpp>
#include <samurai/box.hpp>

#include "../user_bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Free-stream preservation (mesh adaptation sanity check)
// -----------------------------------------------------------------------------
//  A uniform flow (constant density, pressure and velocity) is an exact, trivial
//  solution of the Euler equations: it must stay strictly uniform for all time.
//
//  On an adapted (multiresolution) mesh this is NOT automatic: any inconsistency
//  in the prediction / projection operators or in the flux reconstruction at
//  level interfaces breaks the uniform state and shows up immediately as a
//  non-zero L1/Linf error. This makes it the cheapest, sharpest test to validate
//  the AMR machinery (free-stream preservation).
//
//  The exact solution at any time is the same constant state, so the error is
//  obtained by comparing to the initial uniform values.
// =============================================================================

namespace test_case::free_stream
{
    inline const PrimState<2> uniform_state{
        1.,                                                     // density
        1.,                                                     // pressure
        xt::xtensor_fixed<double, xt::xshape<2>>{1., 1.}        // velocity (non-zero on purpose)
    };

    auto init_fn = [](auto& u, auto& cell)
    {
        u[cell] = prim2cons<2>(uniform_state);
    };

    void bc_fn(auto& u, double /*t*/)
    {
        // Impose the exact uniform state on every boundary.
        auto cons = prim2cons<2>(uniform_state);
        using EulerConsVar = EulerLayout<2>;
        samurai::make_bc<Imposed>(u,
                                  cons[EulerConsVar::rho],
                                  cons[EulerConsVar::rhoE],
                                  cons[EulerConsVar::mom(0)],
                                  cons[EulerConsVar::mom(1)]);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }
}

REGISTER_TEST_CASE(free_stream, test_case::free_stream::box_fn, test_case::free_stream::init_fn, test_case::free_stream::bc_fn)
