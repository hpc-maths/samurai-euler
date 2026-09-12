// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../eos.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Sedov blast wave
// -----------------------------------------------------------------------------
//  A fixed amount of energy is deposited in a small ball at the centre of an
//  ambient gas at rest, driving a strong self-similar shock. The exact solution
//  is radially symmetric, which makes it the canonical stress test for an AMR
//  solver: any anisotropy in adaptation, prediction or flux reconstruction shows
//  up immediately as a broken symmetry.
//
//  The definition is dimension agnostic — only the volume over which the energy
//  is spread changes — so the same case serves euler_2d and euler_3d.
// =============================================================================

namespace test_case::sedov_blast
{
    inline constexpr double rho_ambient = 1.0;  // ambient density
    inline constexpr double p_ambient   = 1e-5; // ambient pressure (very small)
    inline constexpr double r_blast     = 0.1;  // blast radius

    // Blast energy. These are the values that put the shock at r = 1 at t = 1 for
    // gamma = 1.4 in the Kamm & Timmes verification suite (planar 0.0673185,
    // spherical 0.851072; the same spherical value is used by clawpack and by
    // lanl/HARD).
    //
    // NOTE the two-dimensional value below is NOT that suite's cylindrical one,
    // which is 0.311357. It predates this file and is left as it was, since
    // changing it changes results; worth settling separately.
    template <std::size_t dim>
    constexpr double blast_energy()
    {
        static_assert(dim >= 1 && dim <= 3, "no blast energy tabulated for this dimension");

        if constexpr (dim == 1)
        {
            return 0.0673185;
        }
        else if constexpr (dim == 2)
        {
            return 0.244816;
        }
        else
        {
            return 0.851072;
        }
    }

    // Measure of the region the energy is deposited in: a segment in 1D, a disk
    // in 2D, a ball in 3D. Spelled out for every dimension rather than left to a
    // trailing else, which would silently hand a new dimension the 3D answer.
    template <std::size_t dim>
    constexpr double blast_volume()
    {
        static_assert(dim >= 1 && dim <= 3, "no blast volume for this dimension");

        if constexpr (dim == 1)
        {
            return 2. * r_blast;
        }
        else if constexpr (dim == 2)
        {
            return std::numbers::pi * r_blast * r_blast;
        }
        else
        {
            return 4. / 3. * std::numbers::pi * r_blast * r_blast * r_blast;
        }
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;
        using EulerConsVar               = EulerLayout<dim>;

        const auto x = cell.center();

        // Domain is centred on the origin, so the blast centre is the origin.
        double r2 = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            r2 += x[d] * x[d];
        }

        const double p = (r2 < r_blast * r_blast) ? (eos.gamma - 1.0) * blast_energy<dim>() / blast_volume<dim>() // blast zone
                                                  : p_ambient;                                                    // ambient zone

        u[cell][EulerConsVar::rho]  = rho_ambient;
        u[cell][EulerConsVar::rhoE] = rho_ambient * eos.e(rho_ambient, p);
        for (std::size_t d = 0; d < dim; ++d)
        {
            u[cell][EulerConsVar::mom(d)] = 0.; // gas initially at rest
        }
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1.);
        max_corner.fill(1.);

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(sedov_blast, test_case::sedov_blast, 1, 2, 3)
