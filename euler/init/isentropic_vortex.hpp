// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>

#include "../user_bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Isentropic Euler vortex (smooth, analytic test case)
// -----------------------------------------------------------------------------
//  An isentropic vortex superimposed on a uniform mean flow. It is an exact
//  solution of the Euler equations: the vortex is advected by the mean flow
//  without deformation, so the exact solution at time t is the initial condition
//  translated by the mean velocity. Being C-infinity (no shock, no contact), the
//  L1/L2/Linf errors decay at the *design* order of the scheme, which makes this
//  case the reference for measuring convergence rates; the moving structure also
//  exercises the mesh adaptation (the refined region must track the vortex).
//
//  Notation and formulas follow:
//      S. C. Spiegel, H. T. Huynh, J. R. DeBonis, "A Survey of the Isentropic
//      Euler Vortex Problem using High-Order Methods", AIAA Paper 2015-2444,
//      NASA Glenn Research Center, 2015 (NTRS 20150018403).
//  Equation/section numbers below refer to that paper. The parameter set is the
//  "Shu" row of Table 1, in the paper's (sound-speed) non-dimensionalization:
//  rho_inf = 1, a_inf = 1, T_inf = 1, R_gas = 1, so p_inf = 1/gamma.
//
//  Periodicity is emulated by imposing the exact (time-dependent) solution at the
//  boundaries, so no change to the mesh / BC framework is required.
// =============================================================================

namespace test_case::isentropic_vortex
{
    using std::numbers::pi;

    inline const double gamma = EOS::stiffened_gas::gamma;       // ratio of specific heats (eq. 18-19)

    // --- "Shu" parameter row of Table 1 ------------------------------------
    inline const double     alpha   = pi / 4.;                   // angle of attack (mean-flow direction)
    inline const double     M_inf   = std::sqrt(2. / gamma);     // free-stream Mach number
    inline constexpr double rho_inf = 1.;                        // free-stream density
    inline constexpr double R       = 1.;                        // characteristic length scale (eq. 21)
    inline constexpr double sigma   = 1.;                        // Gaussian standard deviation (eq. 21)
    inline const double     beta    = M_inf * 5. * std::sqrt(2.) / (4. * pi) * std::exp(0.5); // vortex strength (eq. 20)
    inline constexpr double L       = 5.;                        // half domain length: domain [-L, L]^2 (sec. IV.A.2)
    inline constexpr double x0      = 0.;                        // initial vortex center x (eq. 24)
    inline constexpr double y0      = 0.;                        // initial vortex center y (eq. 24)

    // free-stream velocity components (eq. 23): M_inf * (cos alpha, sin alpha)
    inline const double v_x_inf = M_inf * std::cos(alpha);
    inline const double v_y_inf = M_inf * std::sin(alpha);

    // Exact primitive state at point (x, y) and time t. Implements eqs. (20)-(24)
    // of the reference paper (periodic, nearest-image evaluation).
    inline PrimState<2> exact_state(double x, double y, double t)
    {
        constexpr double domain_length = 2. * L; // periodic length in each direction

        // moving vortex center (eq. 24), wrapped into [-L, L]
        const double x_c = x0 + v_x_inf * t;
        const double y_c = y0 + v_y_inf * t;

        double x_bar = x - x_c;
        double y_bar = y - y_c;
        x_bar -= domain_length * std::round(x_bar / domain_length);
        y_bar -= domain_length * std::round(y_bar / domain_length);

        // Gaussian (eqs. 20-21) and perturbations (eq. 22)
        const double f     = -1. / (2. * sigma * sigma) * ((x_bar / R) * (x_bar / R) + (y_bar / R) * (y_bar / R));
        const double Omega = beta * std::exp(f);

        const double delta_v_x = -(y_bar / R) * Omega;
        const double delta_v_y = +(x_bar / R) * Omega;
        const double delta_T   = -(gamma - 1.) / 2. * Omega * Omega;

        // initial primitive variables (eq. 23), with the isentropic relations
        const double rho = rho_inf * std::pow(1. + delta_T, 1. / (gamma - 1.));
        const double p   = 1. / gamma * std::pow(1. + delta_T, gamma / (gamma - 1.));
        const double v_x = v_x_inf + delta_v_x;
        const double v_y = v_y_inf + delta_v_y;

        return PrimState<2>{
            rho,
            p,
            xt::xtensor_fixed<double, xt::xshape<2>>{v_x, v_y}
        };
    }

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x  = cell.center();
        u[cell] = prim2cons<2>(exact_state(x[0], x[1], 0.));
    };

    void bc_fn(auto& u, double& t)
    {
        // Impose the exact (time-dependent) solution on every boundary.
        auto exact_bc = [&t](const auto&, const auto& cell, const auto&)
        {
            auto x = cell.center();
            return prim2cons<2>(exact_state(x[0], x[1], t));
        };

        for (const auto& dir : {xt::xtensor_fixed<int, xt::xshape<2>>{-1, 0},
                                xt::xtensor_fixed<int, xt::xshape<2>>{1, 0},
                                xt::xtensor_fixed<int, xt::xshape<2>>{0, -1},
                                xt::xtensor_fixed<int, xt::xshape<2>>{0, 1}})
        {
            samurai::make_bc<Imposed>(u, exact_bc)->on(dir);
        }
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {-L, -L};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {L, L};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }
}

REGISTER_TEST_CASE(isentropic_vortex,
                   test_case::isentropic_vortex::box_fn,
                   test_case::isentropic_vortex::init_fn,
                   test_case::isentropic_vortex::bc_fn)
