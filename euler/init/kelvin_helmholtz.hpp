// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/bc.hpp>

#include "../variables.hpp"
#include "registry.hpp"

// Kelvin-Helmholtz instability: two horizontal layers in shear. A small
// vertical-velocity perturbation localized at the two interfaces grows into
// the characteristic rolled-up billows.
namespace test_case::kelvin_helmholtz
{
    constexpr double pi = std::numbers::pi;

    // Layer states
    double rho_in  = 2.0;  // inner layer (0.25 < y < 0.75)
    double rho_out = 1.0;  // outer layers
    double v_shear = 0.5;  // horizontal shear velocity (+/-)
    double p0      = 2.5;  // uniform pressure

    // Perturbation
    double amp   = 0.1;    // amplitude of the seeded vertical velocity
    double sigma = 0.05;   // interface thickness of the seed
    int    mode  = 2;      // number of billows (wavenumber = 2*mode)

    auto init_fn = [](auto& u, auto& cell)
    {
        auto c        = cell.center();
        const double x = c[0];
        const double y = c[1];

        const bool inner = (y > 0.25 && y < 0.75);
        const double rho = inner ? rho_in : rho_out;
        const double vx  = inner ? v_shear : -v_shear;

        const double seed = std::exp(-(y - 0.25) * (y - 0.25) / (2 * sigma * sigma))
                          + std::exp(-(y - 0.75) * (y - 0.75) / (2 * sigma * sigma));
        const double vy = amp * std::sin(2 * mode * pi * x) * seed;

        PrimState<2> state{
            rho,
            p0,
            xt::xtensor_fixed<double, xt::xshape<2>>{vx, vy}
        };
        u[cell] = prim2cons<2>(state);
    };

    void bc_fn(auto& u, double /*t*/)
    {
        samurai::make_bc<samurai::Neumann<1>>(u, 0., 0., 0., 0.);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
        return samurai::Box<double, dim>(min_corner, max_corner);
    }
}

REGISTER_TEST_CASE(kelvin_helmholtz,
                   test_case::kelvin_helmholtz::box_fn,
                   test_case::kelvin_helmholtz::init_fn,
                   test_case::kelvin_helmholtz::bc_fn)
