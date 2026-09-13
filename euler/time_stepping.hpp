// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <stdexcept>
#include <string>

#include <samurai/field.hpp>

// =============================================================================
//  Time integration
// -----------------------------------------------------------------------------
//  Two integrators:
//
//      euler   one explicit step. First order on its own; with the Hancock
//              predictor in the flux it is the MUSCL-Hancock scheme, second
//              order and one flux evaluation per step.
//      ssprk2  two stages averaged (Heun). Second order in time whatever the
//              flux does, at two flux evaluations per step.
//
//  Measured: on the advected pulse, in one dimension, MUSCL-Hancock converges
//  at order 2.01 and is about twice as accurate as SSP-RK2 at equal
//  resolution, for half the work. On the isentropic vortex, in two dimensions,
//  it drops to order 1.05 while SSP-RK2 holds 2.13.
//
//  The Hancock predictor advances a face value with the equations taken normal
//  to that face, and in more than one dimension the transverse terms it leaves
//  out are of the same order as the ones it keeps. Recovering them needs the
//  cells across the face, and a flux stencil is a line. Lowering the CFL
//  shrinks the missing term and the order climbs back towards 2, reaching 1.77
//  at CFL 0.05, so the reconstruction itself is sound.
//
//  Hence `auto`: explicit Euler at order 1, SSP-RK2 at order 2. In one
//  dimension `--time-integrator euler` at order 2 is the better scheme.
// =============================================================================

enum class TimeIntegrator
{
    euler,
    ssprk2
};

// What `--time-integrator auto` means: the cheapest integrator that gives the
// scheme the order it is being asked for.
inline std::string default_time_integrator(std::size_t order)
{
    return order == 1 ? "euler" : "ssprk2";
}

inline TimeIntegrator time_integrator_from_name(const std::string& name)
{
    if (name == "euler")
    {
        return TimeIntegrator::euler;
    }
    if (name == "ssprk2")
    {
        return TimeIntegrator::ssprk2;
    }
    throw std::runtime_error("Unknown time integrator: " + name);
}

// One step, from u to u. `work1` and `work2` are scratch fields on the same
// mesh, carrying the same boundary conditions as u: the second stage of SSP-RK2
// evaluates the scheme on an intermediate state, and a scheme evaluation fills
// ghosts, which needs the boundary conditions to be there.
template <class Field, class Scheme>
void advance(Field& u, Field& work1, Field& work2, Scheme& scheme, double dt, TimeIntegrator integrator)
{
    work1.resize();

    if (integrator == TimeIntegrator::euler)
    {
        work1 = u - dt * scheme(u);
        samurai::swap(u, work1);
        return;
    }

    work2.resize();

    work1 = u - dt * scheme(u);          // u^(1)
    work2 = work1 - dt * scheme(work1);  // u^(2), one more step from there
    work1 = 0.5 * (u + work2);           // the average of the two is second order

    samurai::swap(u, work1);
}
