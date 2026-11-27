// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

#include <samurai/bc.hpp>

#include "variables.hpp"
#include "../geometry/grid.hpp"

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
//  Both conditions are written for any even stencil size: a stencil of 2 fills
//  one layer of ghost cells, which is what the first-order flux reads, and a
//  stencil of 4 fills two, which is what the MUSCL reconstruction reads. The
//  helpers below pick the width from `bc::ghost_layers`, set once at startup
//  from the order of the scheme.
// =============================================================================

namespace detail
{
    // In a boundary stencil of size N the cells run from the inside out: the
    // last N/2 are the ghosts to fill, cells[N/2 - 1] is the cell just inside
    // the boundary, and cells[N/2 - 1 - k] is k cells further in.
    inline constexpr std::size_t first_ghost(std::size_t stencil_size)
    {
        return stencil_size / 2;
    }
}

template <std::size_t StencilSize, class Field>
struct ImposedImpl : public samurai::Bc<Field>
{
    INIT_BC(ImposedImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t& value)
        {
            for (std::size_t i = detail::first_ghost(StencilSize); i < StencilSize; ++i)
            {
                u[cells[i]] = value;
            }
        };
    }
};

template <std::size_t StencilSize, class Field>
struct ReflectiveImpl : public samurai::Bc<Field>
{
    INIT_BC(ReflectiveImpl, StencilSize)

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
            // The wall is a mirror: the k-th ghost copies the k-th cell inside,
            // with the normal momentum reversed, so the wall-normal mass flux
            // vanishes.
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);

            for (std::size_t k = 0; k < ghost0; ++k)
            {
                const auto& inside = cells[ghost0 - 1 - k];
                const auto& ghost  = cells[ghost0 + k];

                u[ghost]                                       = u[inside];
                u[ghost][EulerLayout<Field::dim>::mom(normal)] = -u[inside][EulerLayout<Field::dim>::mom(normal)];
            }
        };
    }
};

// Zeroth-order extrapolation: every ghost repeats the boundary cell. This is
// what samurai's Neumann<1> does with a zero derivative, widened to more than
// one layer.
template <std::size_t StencilSize, class Field>
struct OutflowImpl : public samurai::Bc<Field>
{
    INIT_BC(OutflowImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);

            for (std::size_t i = ghost0; i < StencilSize; ++i)
            {
                u[cells[i]] = u[cells[ghost0 - 1]];
            }
        };
    }
};


// What the walls of a mapped domain need to know: the map, the gas and
// whether to correct the pressure for the curvature of the wall. Set once by
// the program before the boundary conditions are applied.
struct MappedWallContext
{
    const PolysquareMap* map = nullptr;
    EOS::IdealGas eos        = EOS::ideal_gas(1.4);
    bool curvature           = true;
};

inline MappedWallContext& mapped_wall_context()
{
    static MappedWallContext context;
    return context;
}

// The state u (conservative, per physical volume) carried a signed distance s
// along the outward normal n of a wall of curvature kappa. At a fixed wall
// v . n = 0 at all times, and the normal momentum equation gives
//     dp/dn = rho kappa v_t^2,
// with kappa > 0 when the wall is concave seen from the fluid. The pressure
// moves by rho kappa v_t^2 s, the density follows the isentrope, and the
// velocity is kept. A mirror without this term imposes dp/dn = 0, a first
// order error in the pressure at a curved wall.
template <class State>
State curvature_corrected(const State& u, const point_t& n, double kappa, double s, const EOS::IdealGas& eos)
{
    auto prim         = cons2prim<2>(u, eos);
    const double vn   = prim.v[0] * n[0] + prim.v[1] * n[1];
    const double vt2  = prim.v[0] * prim.v[0] + prim.v[1] * prim.v[1] - vn * vn;
    const double p_new = prim.p + prim.rho * kappa * vt2 * s;
    if (!(p_new > 0))
    {
        return u;
    }
    prim.rho = prim.rho * std::pow(p_new / prim.p, 1. / eos.gamma);
    prim.p   = p_new;
    return prim2cons<2>(prim, eos);
}

// A solid wall of a mapped domain, filled by position. The ghost takes the
// state of the cell of the domain it mirrors (PolysquareMap::mirror_box, the
// cell whose isometric image it is, not its neighbour in the stencil), with
// the momentum reflected across n = (x_g - x_m) / |x_g - x_m|, the direction
// from the mirror centre to the ghost centre: the normal of the wall face for
// a reflection, the direction to the corner for a point reflection. With the
// curvature correction, the pressure is moved by rho kappa v_t^2 |x_g - x_m|.
//
// The value depends on the ghost only, never on the direction from which it
// is reached: the ghost at the corner of the hole, the +x neighbour of one
// cell and the +y neighbour of another, gets the same value from both.
//
// PerParametricArea: the field holds q = J u (true) or u (false). The ghost
// has the Jacobian of its mirror, so q_g = J_m u_g.
//
// The flux at the wall does not read these ghosts: make_mapped_upwind builds
// its own mirror state, so that the mass flux through the wall vanishes
// exactly. The ghosts serve the prediction of the multiresolution and the
// stencils of higher order schemes.
template <std::size_t StencilSize, class Field, bool PerParametricArea>
struct MappedWallImpl : public samurai::Bc<Field>
{
    INIT_BC(MappedWallImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);
            using layout                        = EulerLayout<Field::dim>;
            const auto& ctx                     = mapped_wall_context();
            if (ctx.map == nullptr)
            {
                throw std::runtime_error("MappedWall needs mapped_wall_context().map to be set");
            }
            const auto& map  = *ctx.map;
            const auto& mesh = u.mesh();

            for (std::size_t k = 0; k < ghost0; ++k)
            {
                const auto& ghost = cells[ghost0 + k];
                const auto box    = cell_box(map, ghost);
                const auto mbox   = map.mirror_box(box);
                // indices of the mirror cell at the level of the ghost
                const long offset_i = box.i0 - static_cast<long>(ghost.indices[0]) * box.n;
                const long offset_j = box.j0 - static_cast<long>(ghost.indices[1]) * box.n;
                const long mi       = (mbox.i0 - offset_i) / box.n;
                const long mj       = (mbox.j0 - offset_j) / box.n;
                // the mirror cell exists at this level in the reference mesh
                // unless the mesh is too coarse there: then the cell inside
                // the stencil, which is the mirror for a single reflection
                auto u_m = xt::eval(u[cells[ghost0 - 1 - k]]);
                try
                {
                    const auto mcell = mesh.get_cell(ghost.level, static_cast<int>(mi), static_cast<int>(mj));
                    u_m              = u[mcell];
                }
                catch (const std::out_of_range&)
                {
                }
                const double jac = PerParametricArea ? std::abs(polygon_area(map.polygon(mbox))) / (ghost.length * ghost.length) : 1.;
                ConsArray<2> state;
                for (std::size_t c = 0; c < Field::n_comp; ++c)
                {
                    state[c] = u_m[c] / jac;
                }

                const point_t xg = polygon_centroid(map.physical_polygon(box));
                const point_t xm = polygon_centroid(map.polygon(mbox));
                const point_t d  = xg - xm;
                const double dist = std::hypot(d[0], d[1]);
                const point_t n  = d / dist;
                if (ctx.curvature)
                {
                    state = curvature_corrected(state, n, map.ghost_curvature(box), dist, ctx.eos);
                }
                const double mx = state[layout::mom(0)];
                const double my = state[layout::mom(1)];
                const double mn = mx * n[0] + my * n[1];
                state[layout::mom(0)] = mx - 2 * mn * n[0];
                state[layout::mom(1)] = my - 2 * mn * n[1];
                for (std::size_t c = 0; c < Field::n_comp; ++c)
                {
                    u[ghost][c] = jac * state[c];
                }
            }
        };
    }
};

template <std::size_t StencilSize = 2, bool PerParametricArea = false>
struct MappedWall
{
    template <class Field>
    using impl_t = MappedWallImpl<StencilSize, Field, PerParametricArea>;
};

// A smooth extension of the solution beyond the boundary, for the details of
// the multiresolution and not for the flux. A mirror makes the extension
// even, with a kink at the wall wherever the normal derivative does not
// vanish (the tangential velocity of a shear flow, the pressure along a
// curved wall where dp/dn = rho kappa v_t^2), and the multiresolution refines
// that kink although the flow is smooth there.
//
// The extension is linear and, on a mapped domain, depends on the position of
// the ghost only (PolysquareMap::ghost_image):
//  - across one wall, along the column of the ghost: u_b + (k + 1)(u_b - u_b2),
//    from the boundary cell b of that column and the next one inside, for the
//    ghost of layer k + 1;
//  - at a corner of the polysquare, the plane through the three cells of the
//    domain around the corner, u = u_row + u_col - u_corner: the cell of the
//    row of the ghost and the cell of its column along the walls, and the
//    corner cell.
// The ghost at the corner of the hole then gets one value, whether it is the
// +x neighbour of one cell or the +y neighbour of another. Without a map, or
// where a cell is missing at this level, the line of the stencil is used.
template <std::size_t StencilSize, class Field>
struct SmoothExtensionImpl : public samurai::Bc<Field>
{
    INIT_BC(SmoothExtensionImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);
            static_assert(ghost0 >= 2, "the linear extension needs two cells inside");
            const auto* map  = mapped_wall_context().map;
            const auto& mesh = u.mesh();
            for (std::size_t k = 0; k < ghost0; ++k)
            {
                const auto& ghost = cells[ghost0 + k];
                // the line of the stencil, by default
                auto value = xt::eval(u[cells[ghost0 - 1]] + static_cast<double>(k + 1) * (u[cells[ghost0 - 1]] - u[cells[ghost0 - 2]]));
                if (map != nullptr)
                {
                    try
                    {
                        const auto box      = cell_box(*map, ghost);
                        const auto img      = map->ghost_image(box);
                        const auto mbox     = map->mirror_box(box);
                        const long n        = box.n;
                        const long offset_i = box.i0 - static_cast<long>(ghost.indices[0]) * n;
                        const long offset_j = box.j0 - static_cast<long>(ghost.indices[1]) * n;
                        auto at             = [&](long gi, long gj)
                        {
                            return xt::eval(u[mesh.get_cell(ghost.level, static_cast<int>((gi - offset_i) / n), static_cast<int>((gj - offset_j) / n))]);
                        };
                        // for an axis reflected across `wall`: geometry index of the
                        // boundary cell on the side of the domain, its step inward,
                        // and the layer of the ghost (1 for the first)
                        auto axis = [&](long g0, long m0, long wall)
                        {
                            const bool below  = m0 < wall;
                            const long b      = below ? wall - n : wall;
                            const long step   = below ? -n : n;
                            const long layer  = (g0 >= wall ? (g0 - wall) / n : (wall - g0 - n) / n) + 1;
                            return std::tuple<long, long, long>{b, step, layer};
                        };
                        if (img.reflect_i && img.reflect_j)
                        {
                            const long bi = std::get<0>(axis(box.i0, mbox.i0, img.wall_i));
                            const long bj = std::get<0>(axis(box.j0, mbox.j0, img.wall_j));
                            value         = xt::eval(at(bi, box.j0) + at(box.i0, bj) - at(bi, bj));
                        }
                        else if (img.reflect_i)
                        {
                            const auto [bi, si, li] = axis(box.i0, mbox.i0, img.wall_i);
                            const auto ub           = at(bi, box.j0);
                            value                   = xt::eval(ub + static_cast<double>(li) * (ub - at(bi + si, box.j0)));
                        }
                        else
                        {
                            const auto [bj, sj, lj] = axis(box.j0, mbox.j0, img.wall_j);
                            const auto ub           = at(box.i0, bj);
                            value                   = xt::eval(ub + static_cast<double>(lj) * (ub - at(box.i0, bj + sj)));
                        }
                    }
                    catch (const std::out_of_range&)
                    {
                    }
                }
                u[ghost] = value;
            }
        };
    }
};

template <std::size_t StencilSize = 4>
struct SmoothExtension
{
    template <class Field>
    using impl_t = SmoothExtensionImpl<StencilSize, Field>;
};

// samurai's make_bc takes a type exposing impl_t<Field>, so each condition gets
// the same thin wrapper its Neumann does.
template <std::size_t StencilSize = 2>
struct Imposed
{
    template <class Field>
    using impl_t = ImposedImpl<StencilSize, Field>;
};

template <std::size_t StencilSize = 2>
struct Reflective
{
    template <class Field>
    using impl_t = ReflectiveImpl<StencilSize, Field>;
};

template <std::size_t StencilSize = 2>
struct Outflow
{
    template <class Field>
    using impl_t = OutflowImpl<StencilSize, Field>;
};

namespace bc
{
    // How many layers of ghost cells the boundary conditions have to fill: one
    // for the first-order flux, two for the MUSCL reconstruction. Set once at
    // startup from the order of the scheme, and read by every helper below, so
    // that the `bc_fn` of a test case states what the boundary is and not how
    // wide the stencil reading it will be.
    inline std::size_t& ghost_layers()
    {
        static std::size_t layers = 1;
        return layers;
    }

    inline bool wide()
    {
        return ghost_layers() > 1;
    }

    // Outflow (non-reflecting to first order): the boundary cell repeated into
    // every ghost. The one-layer case still goes through samurai's Neumann,
    // which leaves a first-order run unchanged to the last bit.
    template <class Field>
    auto outflow(Field& u)
    {
        if (wide())
        {
            return samurai::make_bc<Outflow<4>>(u);
        }

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return samurai::make_bc<samurai::Neumann<1>>(u, (static_cast<void>(I), 0.)...);
        }(std::make_index_sequence<Field::n_comp>{});
    }

    // Solid wall on every boundary.
    template <class Field>
    auto wall(Field& u)
    {
        return wide() ? samurai::make_bc<Reflective<4>>(u) : samurai::make_bc<Reflective<2>>(u);
    }

    // Solid wall on every boundary of a mapped domain (see MappedWall), for a
    // field of q = J u (per_parametric_area) or of u. Needs
    // mapped_wall_context().
    template <bool per_parametric_area, class Field>
    auto mapped_wall(Field& u)
    {
        return wide() ? samurai::make_bc<MappedWall<4, per_parametric_area>>(u) : samurai::make_bc<MappedWall<2, per_parametric_area>>(u);
    }

    // An imposed state given by a function of the boundary cell. This is how a
    // time-dependent boundary is written: the case captures whatever it needs,
    // typically the current simulation time, by reference.
    //
    // NOTE that time is the one at the start of the iteration. A stage of
    // SSP-RK2 and a sweep of Strang both re-read it, so both see t^n where the
    // one wants t^n + dt and the other t^n + dt/2. Nothing measurable comes of
    // it today, the only moving boundary in the repository being the analytic
    // shock of the double Mach, whose position over one step moves by less than
    // the error the scheme makes anyway. It is a ceiling on the order that a
    // genuinely unsteady boundary would hit.
    template <class Field>
    auto imposed(Field& u, const typename samurai::FunctionBc<Field>::function_t& f)
    {
        return wide() ? samurai::make_bc<Imposed<4>>(u, f) : samurai::make_bc<Imposed<2>>(u, f);
    }

    // A uniform state imposed on every boundary.
    template <class Field, class Eos>
    auto imposed(Field& u, const PrimState<Field::dim>& state, Eos eos)
    {
        const auto cons = prim2cons<Field::dim>(state, eos);

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return wide() ? samurai::make_bc<Imposed<4>>(u, cons[I]...) : samurai::make_bc<Imposed<2>>(u, cons[I]...);
        }(std::make_index_sequence<Field::n_comp>{});
    }
}
