// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Prototype: finite volumes on a mapped polysquare.
//
// samurai works on the parametric domain [-2, 2]^2 \ (-1, 1)^2 (a polysquare,
// built with DomainBuilder), mapped onto the circular annulus 1 < |x| < 2.
// We solve du/dt + div(a u) = 0 with the rigid rotation a = (-x_2, x_1),
// which is tangent to both walls.
//
// The map is piecewise bilinear on the parametric grid of the geometry level
// (max_level) and is stored by its vertices. They come either from the
// analytic map x = rho(xi) xi / |xi| with rho = max(|xi_1|, |xi_2|), which
// folds the grid along the diagonals, or from the minimization of Garanzha's
// foldover-free energy started from it, with the boundary vertices sliding
// on the circles.
//
// The unknown stored in samurai is q = |K| u / h^2, the mass per unit of
// parametric area, so that the arithmetic mean used by the field projection
// conserves the mass without any change in the library.
//
// The geometry of a cell at level l is the polygon traced by its boundary
// at the geometry level. The physical area of a cell is then exactly the sum
// of the areas of its children, and the sum of the face normals of any cell
// is zero: the scheme is free-stream preserving on a uniform mesh and the
// volumes are nested across levels.

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <bit>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>

#include "euler/variables.hpp"
#include "euler/reconstruction.hpp"
#include "euler/schemes/riemann.hpp"

#include "geometry/grid.hpp"
#include "geometry/save.hpp"
#include "euler/bc.hpp"


namespace fs = std::filesystem;

static constexpr std::size_t dim = 2;


template<typename ConsArray>
ConsArray rotate(ConsArray u, const point_t& n) {
  using EulerConsVar = EulerLayout<2>;
  auto mu = u[EulerConsVar::mom(0)];
  auto mv = u[EulerConsVar::mom(1)];
  u[EulerConsVar::mom(0)] =  mu*n[0] + mv*n[1];
  u[EulerConsVar::mom(1)] = -mu*n[1] + mv*n[0];
  return u;
}

template<typename FluxArray>
FluxArray rotate_back(FluxArray u, const point_t& n) {
  using EulerConsVar = EulerLayout<2>;
  auto fu = u[EulerConsVar::mom(0)];
  auto fv = u[EulerConsVar::mom(1)];
  u[EulerConsVar::mom(0)] = fu*n[0] - fv*n[1];
  u[EulerConsVar::mom(1)] = fu*n[1] + fv*n[0];
  return u;
}



// Physical flux of a through the face on the low side of 'cell' in
// direction d, oriented towards +d, summed over the segments of the
// geometry grid.
template <class Cell>
point_t mean_oriented_surface(const PolysquareMap& map, const Cell& cell, std::size_t d)
{
    const auto box = cell_box(map, cell);
    point_t Sf; 
    Sf.fill(0);

    // En 2D : cette somme est équivalente à la surface orienté des deux vertex extrémaux de la face.
    for (long k = 0; k < box.n; ++k)
    {
        // the normal on the right of a -> b points towards +d
        if (d == 0)
        {
            Sf += oriented_surface(map.vertex(box.i0, box.j0 + k), map.vertex(box.i0, box.j0 + k + 1));
        }
        else
        {
            Sf += oriented_surface(map.vertex(box.i0 + k + 1, box.j0), map.vertex(box.i0 + k, box.j0));
        }
    }
    return Sf;
}

// Largest stable time step of the upwind scheme on the geometry grid: the
// outgoing flux of a cell during dt must not exceed its area. Coarser cells
// are less constrained, since their area grows like h^2 and their fluxes
// like h.
// Largest stable time step of the first order scheme on the physical cells:
// a cell must not send out more than it holds during dt,
//     dt sum_f |S_f| lambda_f <= |K|,  lambda_f <= |v| + c.
// The perimeter of the cell bounds sum_f |S_f|, so dt = min |K| / (P (|v| + c))
// is stable with a CFL number up to 1. Computed on the leaves, every step.
template <class Field, class Eos>
double local_time_step(const PolysquareMap& map, const Field& u_phys, const Eos& eos)
{
    double dt = std::numeric_limits<double>::max();
    samurai::for_each_cell(u_phys.mesh(),
                           [&](const auto& cell)
                           {
                               const auto poly    = cell_polygon(map, cell);
                               const double area  = std::abs(polygon_area(poly));
                               double perimeter   = 0;
                               for (std::size_t k = 0; k < poly.size(); ++k)
                               {
                                   const point_t e = poly[(k + 1) % poly.size()] - poly[k];
                                   perimeter += std::hypot(e[0], e[1]);
                               }
                               const auto prim     = cons2prim<dim>(ConsArray<dim>(u_phys[cell]), eos);
                               const double lambda = std::hypot(prim.v[0], prim.v[1]) + eos.c(prim.rho, prim.p);
                               dt                  = std::min(dt, area / (perimeter * lambda));
                           });
    return dt;
}

template<typename Field>
double stable_time_step(const PolysquareMap& map, Field& u_phys)
{
    return map.step();

    // TODO: time_step un peu plus propre
    // double dt = std::numeric_limits<double>::max;
    // samurai::for_each_cell(q.mesh(),
    //                        [&](const auto& cell)
    //                        {
    //                            const double area = std::sqrt(physical_area(map, cell));
    //                            const double rho  = u_phys[cell][EulerLayout<dim>::rho];
    //                            const double u    = u_phys[cell][EulerLayout<dim>::mom(0)] / rho;
    //                            const double v    = u_phys[cell][EulerLayout<dim>::mom(1)] / rho;
    //                            dt = std::min(dt, (cs + std::hypot(u,v)));
    //                        });
    // return dt;

    // double dt = std::numeric_limits<double>::max();
    // for (long j = 0; j < map.size(); ++j)
    // {
    //     for (long i = 0; i < map.size(); ++i)
    //     {
    //         if (!map.quad_in_domain(i, j))
    //         {
    //             continue;
    //         }
    //         const auto poly = map.polygon({i, j, 1});
    //         double out      = 0;
    //         for (std::size_t k = 0; k < 4; ++k)
    //         {
    //             out += std::max(0., mean_oriented_surface(poly[k], poly[(k + 1) % 4]));
    //         }
    //         dt = std::min(dt, polygon_area(poly) / out);
    //     }
    // }
    // return dt;
}

// The scheme takes u and returns d(q)/dt. Its ghosts are then predicted
// from u, which keeps a constant state exact at level jumps; predicting q
// would not, since the prediction does not know the Jacobian.
// With wall_flux, a face of the wall (one of its two cells is a ghost) does
// not read the ghost: its flux is the Riemann problem between the state of
// the cell inside, carried to the wall (curvature_corrected, over the
// distance from the cell centroid to the face), and its mirror across the
// face. The problem is symmetric, so the mass and energy fluxes through the
// wall vanish exactly whatever the ghost holds.
template <class Field, class Eos>
auto make_mapped_upwind(const PolysquareMap& map, Eos eos, bool wall_flux)
{
    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, Field, Field>;

    samurai::FluxDefinition<cfg> upwind;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            upwind[d].cons_flux_function =
                [&map, eos, wall_flux](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u_stencil)
            {
                const auto [A_f, n_f] = get_norm_unit(mean_oriented_surface(map, data.cells[1], d)); // left interface of the right cell

                const bool in0 = map.quad_in_domain(cell_box(map, data.cells[0]).i0, cell_box(map, data.cells[0]).j0);
                const bool in1 = map.quad_in_domain(cell_box(map, data.cells[1]).i0, cell_box(map, data.cells[1]).j0);
                if (wall_flux && in0 != in1)
                {
                    using layout        = EulerLayout<dim>;
                    const std::size_t k = in0 ? 0 : 1;
                    const point_t n_out = in0 ? n_f : point_t(-n_f);
                    ConsArray<dim> u_in = u_stencil[k];
                    const auto& ctx     = mapped_wall_context();
                    if (ctx.curvature)
                    {
                        // the wall face: low face of the right cell in direction d
                        const auto box     = cell_box(map, data.cells[1]);
                        const double kappa = map.face_curvature(box.i0, box.j0, box.n, d);
                        const point_t& p0  = map.vertex(box.i0, box.j0);
                        const point_t xc   = polygon_centroid(cell_polygon(map, data.cells[k]));
                        const double dist  = (p0[0] - xc[0]) * n_out[0] + (p0[1] - xc[1]) * n_out[1];
                        u_in               = curvature_corrected(u_in, n_out, kappa, dist, ctx.eos);
                    }
                    auto u_wall                 = rotate(u_in, n_f);
                    auto u_mirror               = u_wall;
                    u_mirror[layout::mom(0)]    = -u_wall[layout::mom(0)];
                    const auto& uL              = in0 ? u_wall : u_mirror;
                    const auto& uR              = in0 ? u_mirror : u_wall;
                    flux = rotate_back(riemann::solve<riemann::Solver::hllc, 0>(cons2prim<dim>(uL, eos), uL, cons2prim<dim>(uR, eos), uR, eos), n_f)
                         * A_f / data.cell_length;
                    return;
                }
                
                // Rotate the conservative state
                const auto uL = rotate(u_stencil[0], n_f);
                const auto uR = rotate(u_stencil[1], n_f);

                // compute Riemann problem, rotate back the flux
                flux = rotate_back(
                            riemann::solve<riemann::Solver::hllc, 0>(cons2prim<dim>(uL, eos), uL, cons2prim<dim>(uR, eos), uR, eos), 
                            n_f) 
                            * A_f / data.cell_length;
            };
        });
    return samurai::make_flux_based_scheme(upwind);
}

// Second order in space: MUSCL reconstruction of the primitive variables
// (density, Cartesian velocity, pressure) in the parametric space, where the
// cells of a level are uniform, and the flux of make_mapped_upwind between the
// two face values. Four cells per face, numbered 0 to 3, the face between 1
// and 2. The slopes are limited (reconstruction.hpp); a face value that is not
// admissible falls back to the cell average.
//
// At a face of the wall (cell 1 or 2 is a ghost), the state reconstructed on
// the face from the cell inside is reflected across the face, and the flux is
// the Riemann problem between the two: the mass and energy fluxes through the
// wall vanish exactly, whatever the ghost holds. The ghost enters through the
// slope of the cell inside only. It is filled by MappedWall with the curvature
// correction, so that the slope of the pressure there carries
// dp/dn = rho kappa v_t^2 rather than 0.
//
// The time integration (SSP-RK2) is in the main loop.
template <class Field, class Eos>
auto make_mapped_muscl(const PolysquareMap& map, Eos eos, SlopeLimiter limiter)
{
    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 4, Field, Field>;

    samurai::FluxDefinition<cfg> muscl;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            muscl[d].cons_flux_function =
                [&map, eos, limiter](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u)
            {
                using layout = EulerLayout<dim>;
                // left interface of cell 2
                const auto [A_f, n_f] = get_norm_unit(mean_oriented_surface(map, data.cells[2], d));

                auto in_domain = [&](std::size_t k)
                {
                    const auto box = cell_box(map, data.cells[k]);
                    return map.quad_in_domain(box.i0, box.j0);
                };
                const bool in1 = in_domain(1);
                const bool in2 = in_domain(2);

                const auto w0 = pack<dim>(cons2prim<dim>(u[0], eos));
                const auto w1 = pack<dim>(cons2prim<dim>(u[1], eos));
                const auto w2 = pack<dim>(cons2prim<dim>(u[2], eos));
                const auto w3 = pack<dim>(cons2prim<dim>(u[3], eos));

                auto admissible = [](const ConsArray<dim>& reconstructed, const ConsArray<dim>& average) -> ConsArray<dim>
                {
                    const bool ok = reconstructed[layout::rho] > 0. && reconstructed[layout::rhoE] > 0.;
                    return ok ? reconstructed : average;
                };
                const ConsArray<dim> wL = admissible(w1 + 0.5 * limited_slope<dim>(w1 - w0, w2 - w1, limiter), w1);
                const ConsArray<dim> wR = admissible(w2 - 0.5 * limited_slope<dim>(w2 - w1, w3 - w2, limiter), w2);

                ConsArray<dim> uL = rotate(prim2cons<dim>(unpack<dim>(wL), eos), n_f);
                ConsArray<dim> uR = rotate(prim2cons<dim>(unpack<dim>(wR), eos), n_f);
                if (in1 != in2)
                {
                    // the wall: the face state from inside and its mirror
                    auto& inside  = in1 ? uL : uR;
                    auto& outside = in1 ? uR : uL;
                    outside                 = inside;
                    outside[layout::mom(0)] = -inside[layout::mom(0)];
                }
                flux = rotate_back(riemann::solve<riemann::Solver::hllc, 0>(cons2prim<dim>(uL, eos), uL, cons2prim<dim>(uR, eos), uR, eos), n_f)
                     * A_f / data.cell_length;
            };
        });
    return samurai::make_flux_based_scheme(muscl);
}

struct Diagnostics
{
    double mass         = 0;
    double l1_error     = 0;
    double max_error    = 0;
    std::size_t n_cells = 0;
};

// Parametric cell given by its level and integer indices
struct CellGeometry
{
    std::size_t level;
    double length;
    std::array<double, 2> lower;

    double corner(std::size_t d) const
    {
        return lower[d];
    }
};

template <class Mesh>
CellGeometry cell_geometry(const Mesh& mesh, std::size_t level, long long i, long long j)
{
    const double h = mesh.cell_length(level);
    const auto& o  = mesh.origin_point();
    return {
        level,
        h,
        {o[0] + h * static_cast<double>(i), o[1] + h * static_cast<double>(j)}
    };
}

// Prediction used by the mesh adaptation to fill the cells created by
// refinement. For q = |K| u / h^2, it predicts u with samurai's
// coefficients, multiplies it by the mean Jacobian of each child and adds
// the correction that keeps the mean of the children equal to the parent.
// The mass is conserved and a constant u stays constant, because the
// Jacobian of the parent is the mean of the Jacobians of its children.
// Other fields use the standard prediction.
// When u_source is given, the coarse values of u are read from it instead of
// q / J: its ghosts are predicted in u, not in q.
template <class Field>
auto mapped_prediction_fn(const PolysquareMap& map, bool enabled, const Field* u_source)
{
    return [&map, enabled, u_source](auto& new_field, const auto& old_field)
    {
        auto standard     = samurai::default_config::default_prediction_fn(new_field, old_field);
        const bool mapped = enabled && old_field.name() == "q";
        return [&new_field, &old_field, standard, mapped, &map, u_source](std::size_t level, const auto& interval, const auto& index) mutable
        {
            if (!mapped)
            {
                standard(level, interval, index);
                return;
            }
            using interval_t  = std::decay_t<decltype(interval)>;
            const auto& even  = samurai::prediction_coefficients<1>(0, 0).c;
            const auto& odd   = samurai::prediction_coefficients<1>(1, 0).c;
            const auto& mesh  = old_field.mesh();
            const auto j      = index[0];
            auto coarse_index = index;
            auto fine_index   = index;
            auto coeff        = [&](std::size_t parity, std::size_t s)
            {
                return parity == 0 ? even[s] : odd[s];
            };

            for (auto i = interval.start; i < interval.end; ++i)
            {
                std::array<std::array<double, 3>, 3> u;
                for (std::size_t a = 0; a < 3; ++a)
                {
                    for (std::size_t b = 0; b < 3; ++b)
                    {
                        const auto ia   = i + static_cast<int>(a) - 1;
                        coarse_index[0] = j + static_cast<int>(b) - 1;
                        u[a][b]         = u_source ? (*u_source)(level, interval_t{ia, ia + 1}, coarse_index)[0]
                                                   : old_field(level, interval_t{ia, ia + 1}, coarse_index)[0]
                                                 / mean_jacobian(map, cell_geometry(mesh, level, ia, coarse_index[0]));
                    }
                }
                coarse_index[0]       = j;
                const double q_parent = old_field(level, interval_t{i, i + 1}, coarse_index)[0];

                std::array<std::array<double, 2>, 2> q_child;
                double mean = 0;
                for (std::size_t dx = 0; dx < 2; ++dx)
                {
                    for (std::size_t dy = 0; dy < 2; ++dy)
                    {
                        double u_hat = 0;
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            for (std::size_t b = 0; b < 3; ++b)
                            {
                                u_hat += coeff(dx, a) * coeff(dy, b) * u[a][b];
                            }
                        }
                        const auto ic   = 2 * i + static_cast<int>(dx);
                        const auto jc   = 2 * j + static_cast<int>(dy);
                        q_child[dx][dy] = mean_jacobian(map, cell_geometry(mesh, level + 1, ic, jc)) * u_hat;
                        mean += 0.25 * q_child[dx][dy];
                    }
                }
                for (std::size_t dx = 0; dx < 2; ++dx)
                {
                    for (std::size_t dy = 0; dy < 2; ++dy)
                    {
                        const auto ic                                            = 2 * i + static_cast<int>(dx);
                        fine_index[0]                                            = 2 * j + static_cast<int>(dy);
                        new_field(level + 1, interval_t{ic, ic + 1}, fine_index) = q_child[dx][dy] + q_parent - mean;
                    }
                }
            }
        };
    };
}

template <class Field>
Diagnostics compute_diagnostics(const Field& q, const PolysquareMap& map)
{
    Diagnostics diag;
    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               auto c            = cell.center();
                               auto x            = map(c(0), c(1));
                               diag.mass += q[cell][0] * cell.length * cell.length; // rho
                               ++diag.n_cells;
                           });
    return diag;
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Linear advection on a polysquare mapped onto an annulus", argc, argv);

    double Tf                  = std::numbers::pi / 2;
    bool disc                  = false;
    double cfl                 = 0.9;
    bool mr                    = true;
    bool bump                  = true;
    std::size_t nfiles         = 1;
    bool on_u                  = true;
    double base                = 0;
    bool mapped                = true;
    bool u_source              = false;
    bool optimize_map          = true;
    bool at_rest               = false;
    bool rotation              = false;
    bool curvature             = true;
    bool local_dt              = true;
    int order                  = 2;
    std::string limiter_name   = "moncen";
    bool save_ghosts           = false;
    std::string wall           = "normal";
    bool smooth_details        = true;
    double theta               = 0.8;
    std::size_t map_iterations = 5000;
    double gamma               = 5./3.;

    std::size_t min_level = 3;
    std::size_t max_level = 7;

    fs::path path        = fs::current_path();
    std::string filename = "mapped_advection_annulus";

    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--disc", disc, "Disc grid with no hole.")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--mr,!--no-mr", mr, "Adapt the mesh with multiresolution")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--bump,!--constant", bump, "Initial condition: Gaussian bump or constant state")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--background", base, "Constant state added to the bump")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--mapped-prediction,!--standard-prediction", mapped, "Predict u rather than q when refining")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--u-source", u_source, "With --detail-on-u, read the coarse u of the prediction from u instead of q / J")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--detail-on-u,!--detail-on-q", on_u, "Compute the details on u, or on q = |K| u / h^2, which also sees the geometry")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--optimized-map,!--analytic-map", optimize_map, "Minimize Garanzha's energy of the map, starting from the analytic one")
        ->capture_default_str()
        ->group("Geometry");
    app.add_option("--map-theta", theta, "Weight of the volume term in the energy of the map")->capture_default_str()->group("Geometry");
    app.add_option("--map-iterations", map_iterations, "Maximum number of L-BFGS iterations per level of the map")
        ->capture_default_str()
        ->group("Geometry");
    app.add_option("--gamma", gamma, "Ratio of specific heats (defaults to the value of the test case)")
        ->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_flag("--at-rest", at_rest, "Initial condition: gas at rest, rho = 1, v = 0, p = 1")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--wall", wall, "Wall condition: 'normal' reflects the momentum across the physical wall, 'cartesian' reverses the Cartesian component of the parametric direction (the former Reflective)")
        ->check(CLI::IsMember({"normal", "cartesian"}))
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--rotation", rotation, "Initial condition: steady rigid rotation, rho = 1, v = 0.5 (-y, x), p = 1 + rho 0.25 (r^2 - 1) / 2")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--curvature,!--no-curvature", curvature, "With --wall normal, correct the wall pressure by dp/dn = rho kappa v_t^2")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--local-dt,!--grid-dt",
                 local_dt,
                 "Time step from the physical cells and the waves, dt = cfl min |K| / (P (|v| + c)), every step; or cfl times the step of the geometry grid")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--order", order, "Order of the scheme: 1 (first order flux, explicit Euler) or 2 (MUSCL, SSP-RK2)")
        ->check(CLI::IsMember({1, 2}))
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--limiter", limiter_name, "Slope limiter at order 2: none, minmod, vanleer, moncen")
        ->check(CLI::IsMember({"none", "minmod", "vanleer", "moncen"}))
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--smooth-detail-ghosts,!--wall-detail-ghosts",
                 smooth_details,
                 "With --detail-on-u, fill the ghosts read by the details with a linear extension of u, or with the wall condition")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--save-ghosts", save_ghosts, "Also save the cells and their ghosts with the extended map in <filename>_ghosts_<k>.vtu")
        ->capture_default_str()
        ->group("Output");
    app.add_option("--nfiles", nfiles, "Number of saved time steps after the initial one (0: no output)")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    // The test case carries the gas it was designed for; --gamma overrides it.
    EOS::IdealGas eos = EOS::ideal_gas(gamma);
    std::cout << "Using gamma = " << eos.gamma << std::endl;

    samurai::DomainBuilder<dim> domain({-2., -2.}, {2., 2.});
    if (!disc) {
        domain.remove({-1., -1.}, {1., 1.});
    }

    // Two layers of ghosts at order 2, for the four cell stencil of MUSCL
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level).max_stencil_size(4);
    auto mesh   = samurai::mra::make_mesh(domain, config);

    // --min-level and --max-level override the values above
    min_level      = mesh.min_level();
    max_level      = mesh.max_level();
    const auto map = build_map(max_level, mesh.cell_length(max_level), !disc, optimize_map, theta, map_iterations, radial_map);
    const auto mq  = map_quality(map);
    fmt::print(
        "map ({}): largest angle = {:.2f} deg, det J in [{:.3f}, {:.3f}], singular values in [{:.3f}, {:.3f}], largest aspect = {:.2f}\n",
        optimize_map ? "optimized" : "analytic",
        mq.max_angle,
        mq.min_det,
        mq.max_det,
        mq.min_sigma,
        mq.max_sigma,
        mq.max_aspect);

    auto u0 = [&eos, at_rest, rotation](const point_t& x) -> ConsArray<dim>
    {
        if (rotation)
        {
            constexpr double omega = 0.5;
            PrimState<dim> prim;
            prim.rho  = 1.;
            prim.v[0] = -omega * x[1];
            prim.v[1] = omega * x[0];
            prim.p    = 1. + 0.5 * prim.rho * omega * omega * (x[0] * x[0] + x[1] * x[1] - 1.);
            return prim2cons<dim>(prim, eos);
        }
        if (at_rest)
        {
            PrimState<dim> prim;
            prim.rho  = 1.;
            prim.v[0] = 0.;
            prim.v[1] = 0.;
            prim.p    = 1.;
            return prim2cons<dim>(prim, eos);
        }
        // Kelvin-Helmholtz cylindrique
        constexpr double rmin = 1.;
        constexpr double rsep = 1.5;
        constexpr double rho_in = 1, rho_out = 2;
        constexpr double omega = 0.1;
        constexpr double p0 = 1;

        PrimState<dim> prim;
        const double r = std::hypot(x[0], x[1]);

        prim.rho = (r < rsep) ? rho_in : rho_out;
        const double v_th = (r < rsep) ? -omega : omega;
        prim.v[0] = -v_th * x[1];
        prim.v[1] =  v_th * x[0];

        if (r < rsep) {
          prim.p = p0 + 0.5 * omega*omega * rho_in * (r*r - rmin*rmin);
        }
        else {
          prim.p = p0 + 0.5 * omega*omega * (rho_in * (rsep*rsep - rmin*rmin) + rho_out * (r*r - rsep*rsep));
        }

        return prim2cons<dim>(prim, eos);
    };

    auto q    = samurai::make_vector_field<double, 2 + dim>("euler", mesh);
    // auto q    = samurai::make_scalar_field<double>("q", mesh);
    auto init = [&]()
    {
        q.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   auto c  = cell.center();
                                   q[cell] = xt::eval(mean_jacobian(map, cell) * u0(map(c(0), c(1))));
                               });
    };
    init();

    // samurai::make_bc<samurai::Neumann<1>>(q, 0.); 
// TODO appliquer les bonne bc pour Euler

    auto u_phys = samurai::make_vector_field<double, 2 + dim>("u", mesh);
    // samurai::make_bc<samurai::Neumann<1>>(u_phys, 0.);
    // u = q / J on the cells of a field of q (the stages of SSP-RK2)
    auto update_u_from = [&](const auto& q_field)
    {
        u_phys.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   u_phys[cell] = q_field[cell] / mean_jacobian(map, cell);
                               });
    };
    auto update_u = [&]()
    {
        u_phys.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   u_phys[cell] = q[cell] / mean_jacobian(map, cell);
                               });
    };

    auto qnp1   = samurai::make_vector_field<double, 2 + dim>("qnp1", mesh);
    auto scheme  = make_mapped_upwind<decltype(q)>(map, eos, wall == "normal");
    auto scheme2 = make_mapped_muscl<decltype(q)>(map, eos, slope_limiter_from_name(limiter_name));
    auto q_stage = samurai::make_vector_field<double, 2 + dim>("q_stage", mesh);
    if (order == 2)
    {
        if (wall != "normal")
        {
            throw std::runtime_error("--order 2 needs --wall normal");
        }
        bc::ghost_layers() = 2; // MappedWall<4>: two layers of ghosts
    }

    mapped_wall_context() = {&map, eos, curvature};
    if (wall == "normal")
    {
        bc::mapped_wall<true>(q);
        bc::mapped_wall<false>(u_phys);
    }
    else
    {
        bc::wall(q);
        u_phys.copy_bc_from(q);
    }
    qnp1.copy_bc_from(q);
    q_stage.copy_bc_from(q);
    
    // With --detail-on-u, the details are computed on u so that the
    // geometry does not drive the refinement, and q follows the new mesh.
    auto adapt_on_q = samurai::make_MRAdapt(mapped_prediction_fn<decltype(u_phys)>(map, mapped, nullptr), q);
    auto adapt_on_u = samurai::make_MRAdapt(mapped_prediction_fn(map, mapped, u_source ? &u_phys : nullptr), u_phys);
    // u with ghosts that extend it smoothly (SmoothExtension), only to compute
    // the details: the wall condition stays on u_phys, which the scheme reads.
    auto u_detail = samurai::make_vector_field<double, 2 + dim>("u_detail", mesh);
    samurai::make_bc<SmoothExtension<4>>(u_detail);
    auto adapt_on_smooth_u = samurai::make_MRAdapt(mapped_prediction_fn<decltype(u_phys)>(map, mapped, nullptr), u_detail);
    auto mra_config = samurai::mra_config().epsilon(1e-3); // --mr-eps overrides it
    auto adapt      = [&]()
    {
        if (on_u && smooth_details)
        {
            update_u();
            u_detail.resize();
            samurai::for_each_cell(mesh,
                                   [&](const auto& cell)
                                   {
                                       u_detail[cell] = u_phys[cell];
                                   });
            adapt_on_smooth_u(mra_config, q);
        }
        else if (on_u)
        {
            update_u();
            adapt_on_u(mra_config, q);
        }
        else
        {
            adapt_on_q(mra_config);
        }
    };
    if (mr)
    {
        // Adapt from the exact initial data rather than from projected values
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            adapt();
            init();
        }
    }

    auto diag0 = compute_diagnostics(q, map);
    fmt::print("initial: cells = {}, mass = {:.15e}, max |u - u0| = {:.3e}\n", diag0.n_cells, diag0.mass, diag0.max_error);
    // Physical cells in <filename>_<k>.vtu, gathered with their time in
    // <filename>.pvd; parametric mesh in <filename>_parametric_<k>.h5.
    std::vector<std::pair<double, std::string>> saved_steps;
    auto save = [&](double time)
    {
        const auto k    = saved_steps.size();
        const auto name = fmt::format("{}_{:04}.vtu", filename, k);
        save_physical(path / name, q, map, eos);
        if (save_ghosts)
        {
            update_u();
            samurai::update_ghost_mr(u_phys);
            save_cells_and_ghosts(path / fmt::format("{}_ghosts_{:04}.vtu", filename, k), u_phys, map, eos);
        }
        saved_steps.emplace_back(time, name);
        save_collection(path / fmt::format("{}.pvd", filename), saved_steps);

        // update_u();
        // auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
        // samurai::for_each_cell(mesh,
        //                        [&](const auto& cell)
        //                        {
        //                            level[cell] = cell.level;
        //                        });
        // samurai::save(path, fmt::format("{}_parametric_{:04}", filename, k), mesh, u_phys, level);
    };
    if (nfiles > 0)
    {
        fs::create_directories(path);
        save(0);
    }

    const double grid_dt = cfl * stable_time_step(map, u_phys);
    double dt_min        = std::numeric_limits<double>::max();
    double dt_max        = 0;

    double t              = 0;
    std::size_t nt        = 0;
    std::size_t max_cells = diag0.n_cells;
    while (t < Tf)
    {
        if (mr)
        {
            adapt();
        }
        qnp1.resize();
        update_u();
        const double dt   = local_dt ? cfl * local_time_step(map, u_phys, eos) : grid_dt;
        dt_min            = std::min(dt_min, dt);
        dt_max            = std::max(dt_max, dt);
        const double dt_n = std::min(dt, Tf - t);
        if (order == 1)
        {
            qnp1 = q - dt_n * scheme(u_phys);
        }
        else
        {
            // SSP-RK2: q1 = q - dt L(q), q2 = q1 - dt L(q1), q^{n+1} = (q + q2) / 2
            q_stage.resize();
            q_stage = q - dt_n * scheme2(u_phys);
            update_u_from(q_stage);
            qnp1 = q_stage - dt_n * scheme2(u_phys);
            qnp1 = 0.5 * (q + qnp1);
        }
        samurai::swap(q, qnp1);
        t += dt_n;
        ++nt;
        max_cells = std::max(max_cells, mesh.nb_cells(decltype(mesh)::mesh_id_t::cells));

        if (nfiles > 0 && (t >= static_cast<double>(saved_steps.size()) * Tf / static_cast<double>(nfiles) || t >= Tf))
        {
            save(t);
        }
    }

    auto diag = compute_diagnostics(q, map);
    fmt::print("final:   t = {:.4f}, iterations = {}, cells = {} (max {}), levels [{}, {}]\n",
               t,
               nt,
               diag.n_cells,
               max_cells,
               mesh[decltype(mesh)::mesh_id_t::cells].min_level(),
               mesh[decltype(mesh)::mesh_id_t::cells].max_level());
    fmt::print("         relative mass drift = {:.3e}\n", (diag.mass - diag0.mass) / diag0.mass);
    fmt::print("         time step in [{:.3e}, {:.3e}] ({})\n", dt_min, dt_max, local_dt ? "local" : "grid");
    // The steady cases have the initial state as exact solution: errors at
    // the centroids, weighted by the physical areas.
    if (at_rest || rotation)
    {
        update_u();
        double area = 0, e_p = 0, e_v = 0, max_vr = 0, max_p = 0, wall_p = 0, wall_area = 0;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const auto poly  = cell_polygon(map, cell);
                                   const double a   = std::abs(polygon_area(poly));
                                   const point_t xc = polygon_centroid(poly);
                                   const auto num   = cons2prim<dim>(ConsArray<dim>(u_phys[cell]), eos);
                                   const auto ex    = cons2prim<dim>(u0(xc), eos);
                                   const double r   = std::hypot(xc[0], xc[1]);
                                   const double dp  = std::abs(num.p - ex.p);
                                   area += a;
                                   e_p += a * dp;
                                   e_v += a * std::hypot(num.v[0] - ex.v[0], num.v[1] - ex.v[1]);
                                   max_p  = std::max(max_p, dp);
                                   max_vr = std::max(max_vr, std::abs(num.v[0] * xc[0] + num.v[1] * xc[1]) / r);
                                   // the cells along the walls
                                   if (std::abs(r - 2.) < 2 * cell.length || std::abs(r - 1.) < 2 * cell.length)
                                   {
                                       wall_p += a * dp;
                                       wall_area += a;
                                   }
                               });
        fmt::print("         steady case: L1 |p - p_ex| = {:.3e} (near the walls {:.3e}), max |p - p_ex| = {:.3e}, L1 |v - v_ex| = {:.3e}, max |v_r| = {:.3e}\n",
                   e_p / area,
                   wall_area > 0 ? wall_p / wall_area : 0.,
                   max_p,
                   e_v / area,
                   max_vr);
    }

    samurai::finalize();
    return 0;
}
