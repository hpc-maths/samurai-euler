// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

#include <samurai/bc.hpp>

#include "variables.hpp"

// =============================================================================
//  Boundary conditions for the Euler system
// -----------------------------------------------------------------------------
//  samurai ships Dirichlet, Neumann and polynomial extrapolation. The Euler
//  equations need two more that depend on the *meaning* of the components, so
//  they live here rather than in the library:
//
//    - Imposed    a state written straight into the ghost cell, either constant
//                 or given by a function of the cell. This is how a supersonic
//                 inflow and a time-dependent boundary are expressed.
//    - Reflective a solid wall: the ghost cell mirrors its interior neighbour
//                 with the normal momentum reversed, so the wall-normal mass
//                 flux vanishes.
//
//  The free helpers at the bottom build the boundary conditions the test cases
//  actually ask for, in any dimension, so that a `bc_fn` is one line. They apply
//  to every boundary by default; chain `->on(direction)` to restrict one to a
//  single face, spelling the direction as a vector at the call site:
//
//      bc::outflow(u)->on(right);
//      bc::imposed(u, inflow_state, eos)->on(left);
//
//  NOTE  Both conditions fill a SINGLE layer of ghost cells (stencil size 2),
//  which is all a first-order flux needs. A MUSCL reconstruction reads two
//  layers: these will have to grow to stencil size 4 in lot 1.
// =============================================================================

template <class Field>
struct Imposed : public samurai::Bc<Field>
{
    INIT_BC(Imposed, 2)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t& value)
        {
            u[cells[1]] = value;
        };
    }
};

template <class Field>
struct Reflective : public samurai::Bc<Field>
{
    INIT_BC(Reflective, 2)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t& direction) const override
    {
        // The stencil is aligned with the boundary normal, so exactly one
        // component of `direction` is non-zero: that is the axis to mirror.
        std::size_t normal = 0;
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (direction[d] != 0)
            {
                normal = d;
                break;
            }
        }

        return [normal](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            static constexpr std::size_t in  = 0; // interior cell
            static constexpr std::size_t out = 1; // ghost cell

            u[cells[out]]                                       = u[cells[in]];
            u[cells[out]][EulerLayout<Field::dim>::mom(normal)] = -u[cells[in]][EulerLayout<Field::dim>::mom(normal)];
        };
    }
};

namespace bc
{
    // Outflow (non-reflecting to first order): zero normal derivative on every
    // component. `make_bc` wants exactly n_comp values, hence the pack.
    template <class Field>
    auto outflow(Field& u)
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return samurai::make_bc<samurai::Neumann<1>>(u, (static_cast<void>(I), 0.)...);
        }(std::make_index_sequence<Field::n_comp>{});
    }

    // Solid wall on every boundary.
    template <class Field>
    auto wall(Field& u)
    {
        return samurai::make_bc<Reflective>(u);
    }

    // An imposed state given by a function of the boundary cell. This is how a
    // time-dependent boundary is written: the case captures whatever it needs,
    // typically the current simulation time, by reference.
    template <class Field>
    auto imposed(Field& u, const typename samurai::FunctionBc<Field>::function_t& f)
    {
        return samurai::make_bc<Imposed>(u, f);
    }

    // A uniform state imposed on every boundary.
    template <class Field, class Eos>
    auto imposed(Field& u, const PrimState<Field::dim>& state, Eos eos)
    {
        const auto cons = prim2cons<Field::dim>(state, eos);

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return samurai::make_bc<Imposed>(u, cons[I]...);
        }(std::make_index_sequence<Field::n_comp>{});
    }
}
