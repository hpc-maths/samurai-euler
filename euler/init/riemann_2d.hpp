// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Two-dimensional Riemann problems (four quadrants)
// -----------------------------------------------------------------------------
//  The unit square is split into four quadrants, each holding a uniform state.
//  The nineteen admissible combinations are classified in
//
//      P.D. Lax, X.-D. Liu, "Solution of two-dimensional Riemann problems of gas
//      dynamics by positive schemes", SIAM J. Sci. Comput. 19 (1998) 319-340.
//
//  Quadrants are numbered counter-clockwise from the upper right:
//      q[0] : x >= x0, y >= y0        q[1] : x <  x0, y >= y0
//      q[2] : x <  x0, y <  y0        q[3] : x >= x0, y <  y0
//
//  The interfaces sit at x0 = y0 = 0.8 on [0,1]^2, as in Lax & Liu and in the
//  article this repository reproduces: with t_f = 0.8 the waves then fill
//  the domain without reaching its boundary, where the outflow condition
//  would pollute them.
// =============================================================================

namespace test_case::riemann_2d
{
    using field_t = config<2>::field_t;

    struct Config
    {
        double x0;
        double y0;
        std::array<PrimState<2>, 4> q;
    };

    inline void init_from(const Config& c, field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        if (x[0] >= c.x0 && x[1] >= c.y0)
        {
            u[cell] = prim2cons<2>(c.q[0], eos);
        }
        else if (x[0] < c.x0 && x[1] >= c.y0)
        {
            u[cell] = prim2cons<2>(c.q[1], eos);
        }
        else if (x[0] < c.x0 && x[1] < c.y0)
        {
            u[cell] = prim2cons<2>(c.q[2], eos);
        }
        else // (x[0] >= x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(c.q[3], eos);
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

    // --- configuration 3 -----------------------------------------------------
    inline const Config config_3{
        0.8,
        0.8,
        {PrimState<2>{1.5, 1.5, {0., 0.}},
          PrimState<2>{0.5323, 0.3, {1.206, 0.}},
          PrimState<2>{0.138, 0.029, {1.206, 1.206}},
          PrimState<2>{0.5323, 0.3, {0., 1.206}}}
    };

    // --- configuration 4 -----------------------------------------------------
    inline const Config config_4{
        0.8,
        0.8,
        {PrimState<2>{1.1, 1.1, {0., 0.}},
          PrimState<2>{0.5065, 0.35, {0.8939, 0.}},
          PrimState<2>{1.1, 1.1, {0.8939, 0.89396}},
          PrimState<2>{0.5065, 0.35, {0., 0.89396}}}
    };

    // --- configuration 12 ----------------------------------------------------
    inline const Config config_12{
        0.8,
        0.8,
        {PrimState<2>{0.5197, 0.4, {0., 0.}},
          PrimState<2>{1., 1., {-0.6259, 0.}},
          PrimState<2>{0.8, 1., {-0.6259, -0.6259}},
          PrimState<2>{1., 1., {0., -0.6259}}}
    };

    // One definition per configuration; `--riemann-config` will replace these
    // three entries by a single parameterised case once the constants are fixed.
    template <class Field, const Config& c>
    test_case::TestCase<Field> definition_for()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>,
                .init =
                    [](Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
                {
                    init_from(c, u, cell, eos);
                },
                .bc  = &bc_fn,
                .eos = EOS::ideal_gas(1.4)};
    }

    template <class Field>
    test_case::TestCase<Field> definition_config3()
    {
        return definition_for<Field, config_3>();
    }

    template <class Field>
    test_case::TestCase<Field> definition_config4()
    {
        return definition_for<Field, config_4>();
    }

    template <class Field>
    test_case::TestCase<Field> definition_config12()
    {
        return definition_for<Field, config_12>();
    }
}

// The macro expects a `definition` in the namespace it is given, so each
// configuration gets a thin namespace of its own.
namespace test_case::riemann_2d_config3
{
    template <class Field>
    auto definition()
    {
        return riemann_2d::definition_config3<Field>();
    }
}

namespace test_case::riemann_2d_config4
{
    template <class Field>
    auto definition()
    {
        return riemann_2d::definition_config4<Field>();
    }
}

namespace test_case::riemann_2d_config12
{
    template <class Field>
    auto definition()
    {
        return riemann_2d::definition_config12<Field>();
    }
}

REGISTER_TEST_CASE(riemann2d_config3, test_case::riemann_2d_config3, 2)
REGISTER_TEST_CASE(riemann2d_config4, test_case::riemann_2d_config4, 2)
REGISTER_TEST_CASE(riemann2d_config12, test_case::riemann_2d_config12, 2)
