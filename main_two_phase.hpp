// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <memory>
#include <string>

#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>

#include "euler/bc.hpp"
#include "euler/metrics.hpp"
#include "euler/time_stepping.hpp"
#include "euler/two_phase/eos.hpp"
#include "euler/two_phase/init/cases.hpp"
#include "euler/two_phase/registry.hpp"
#include "euler/two_phase/scheme.hpp"
#include "euler/two_phase/thinc.hpp"
#include "euler/two_phase/utils.hpp"
#include "euler/two_phase/variables.hpp"

// =============================================================================
//  The two-phase solver
// -----------------------------------------------------------------------------
//  One time loop, written once for every dimension, where the monofluid solver
//  has one main per dimension. The two differ in the time loop by exactly two
//  things -- the state has dim + 4 components and the thermodynamics takes a
//  volume fraction -- and everything else here is the same machinery: the same
//  registry, the same boundary conditions, the same time integrators, the same
//  performance metrics.
//
//  What is deliberately NOT here yet:
//
//    - The positivity-preserving multiresolution prediction of the monofluid
//      solver, which keys on the Euler layout. The default prediction plus the
//      admissibility floor is what an adapted two-phase run has for now.
// =============================================================================

template <std::size_t dim>
int run_two_phase(int argc, char* argv[])
{
    using field_t = typename two_phase::config<dim>::field_t;

    auto& app = samurai::initialize(fmt::format("Five-equation two-phase solver ({}D)", dim), argc, argv);

    double Tf  = 9e-4;
    double cfl = 0.4;
    double t   = 0.;
    std::string restart_file;
    std::size_t order           = 2;
    std::string slope_limiter   = "moncen";
    std::string time_integrator = "auto";
    std::string scheme          = "hllc";
    std::string test_case       = "water_air_shock_tube";

    two_phase::ThincOptions thinc;

    fs::path path = "results";
    std::string filename;
    std::size_t nfiles = 1;
    std::string metrics_file;

    auto available = two_phase::registry_t<dim>::instance().available_test_cases();

    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--scheme", scheme, "Finite volume scheme")
        ->capture_default_str()
        ->check(CLI::IsMember({"rusanov", "hll", "hllc"}))
        ->group("Simulation parameters");
    app.add_option("--order", order, "Order of the scheme in space: 1 for cell averages, 2 for a MUSCL reconstruction")
        ->capture_default_str()
        ->check(CLI::IsMember({1, 2}))
        ->group("Simulation parameters");
    app.add_option("--slope-limiter", slope_limiter, "Slope limiter of the MUSCL reconstruction")
        ->capture_default_str()
        ->check(CLI::IsMember({"none", "minmod", "vanleer", "moncen"}))
        ->group("Simulation parameters");
    app.add_option("--time-integrator", time_integrator, "Time integration")
        ->capture_default_str()
        ->check(CLI::IsMember({"auto", "euler", "ssprk2", "strang"}))
        ->group("Simulation parameters");
    app.add_option("--test-case", test_case, "Test case")->capture_default_str()->check(CLI::IsMember(available))->group("Simulation parameters");
    app.add_flag("--thinc", thinc.enabled, "Sharpen the interface: THINC reconstruction of the volume fraction in mixed cells")
        ->group("Simulation parameters");
    app.add_option("--thinc-beta", thinc.beta, "Steepness of the THINC profile")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix (defaults to <test-case>_<scheme>)")->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.add_option("--metrics-file", metrics_file, "Write the performance metrics of the run, as JSON, to this file")->group("Output");

    two_phase::registry_t<dim>::instance().add_options(app);

    SAMURAI_PARSE(argc, argv);

    const auto& selected   = two_phase::registry_t<dim>::instance().get(test_case);
    const EOS::Mixture eos = selected.eos;

    if (time_integrator == "auto")
    {
        time_integrator = default_time_integrator(order);
    }
    const auto integrator = time_integrator_from_name(time_integrator);

    if (filename.empty())
    {
        filename = fmt::format("{}_{}", test_case, scheme);
    }

    auto box = selected.box();

    auto config = samurai::mesh_config<dim>().min_level(8).max_level(8).max_stencil_size(4).disable_minimal_ghost_width();
    config.periodic(selected.periodic);
    config.parse_args();
    config.disable_args_parse();

    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_vector_field<double, dim + 4>("two_phase", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().relative_detail(true);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        u.resize();
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   selected.init(u, cell, eos);
                               });
        MRadaptation(mra_config);
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   selected.init(u, cell, eos);
                               });
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    bc::ghost_layers() = order;
    selected.bc(u, t, eos);

    auto unp1 = samurai::make_vector_field<double, dim + 4>("two_phase", mesh);
    auto unp2 = samurai::make_vector_field<double, dim + 4>("two_phase", mesh);
    unp1.copy_bc_from(u);
    unp2.copy_bc_from(u);

    const double dx      = mesh.cell_length(config.max_level());
    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 0;
    std::size_t nt       = 0;

    two_phase::save(path.string(), fmt::format("{}_init", filename), u, eos);
    samurai::dump(path, fmt::format("{}_restart_init", filename), mesh, u);

    std::cout << fmt::format("Using scheme: {}, order {}, {} in time", scheme, order, time_integrator) << std::endl;

    auto dt_for_flux = std::make_shared<double>(0.);

    two_phase::SchemeOptions options{.limiter   = slope_limiter_from_name(slope_limiter),
                                     .hancock   = integrator != TimeIntegrator::ssprk2,
                                     .dt        = dt_for_flux,
                                     .direction = -1,
                                     .thinc     = thinc};

    // |n_d| per cell and per direction, which is what weights the steepness of
    // the THINC profile. The flux function reads it through a pointer and this
    // is the field it points at, recomputed at the top of every time step.
    auto normals = samurai::make_vector_field<double, dim>("interface_normal", mesh);

    auto first_order  = two_phase::make_first_order_scheme<field_t>(scheme, eos);
    auto second_order = two_phase::make_second_order_scheme<field_t>(scheme, eos, options, &normals);

    auto directional = [&](auto&& make_one)
    {
        return [&]<std::size_t... D>(std::index_sequence<D...>)
        {
            return std::array{make_one(static_cast<int>(D))...};
        }(std::make_index_sequence<dim>{});
    };

    auto first_order_sweeps = directional(
        [&](int d)
        {
            return two_phase::make_first_order_scheme<field_t>(scheme, eos, d);
        });
    auto second_order_sweeps = directional(
        [&](int d)
        {
            auto sweep_options      = options;
            sweep_options.direction = d;
            return two_phase::make_second_order_scheme<field_t>(scheme, eos, sweep_options, &normals);
        });

    Metrics metrics(mesh);

    samurai::times::timers.start("TimeLoop");
    metrics.start(mesh);
    std::size_t limited_cells = 0;
    bool done                 = false;
    while (!done)
    {
        double dt = cfl * dx / two_phase::get_max_lambda(u, eos);

        metrics.adapt(
            [&]
            {
                MRadaptation(mra_config);
            });

        // Twice per step, and for two different reasons. Here because the
        // multiresolution prediction is the default one, not the positivity
        // preserving one the monofluid solver uses, so a refined cell can come
        // out of the adaptation inadmissible; and again after the update, for
        // the same reason the monofluid solver does it there.
        limited_cells += two_phase::limit_positivity(u, eos);

        if (std::isnan(t))
        {
            std::cerr << "Error: Time became NaN, stopping simulation" << std::endl;
            break;
        }

        if (t + dt > Tf)
        {
            dt   = Tf - t;
            done = true;
        }
        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << "\r";

        metrics.step(mesh);

        if (thinc.enabled)
        {
            // The interface normal is a property of the state at the start of
            // the step, and the ghosts have to be filled before it can be read
            // across a level jump. One evaluation per step, not one per stage:
            // the interface moves by less than a cell in a step, and a normal
            // that is one stage old is a smaller error than the one the
            // sharpening is correcting.
            samurai::update_ghost_mr(u);
            two_phase::compute_interface_normals(u, normals);
        }

        if (order == 1)
        {
            advance(u, unp1, unp2, first_order, first_order_sweeps, dt_for_flux, dt, integrator);
        }
        else
        {
            advance(u, unp1, unp2, second_order, second_order_sweeps, dt_for_flux, dt, integrator);
        }

        limited_cells += two_phase::limit_positivity(u, eos);

        t += dt;

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            metrics.output(
                [&]
                {
                    two_phase::save(path.string(), fmt::format("{}{}", filename, suffix), u, eos);
                    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
                });
        }
    }
    metrics.stop(mesh);
    samurai::times::timers.stop("TimeLoop");

    std::cout << std::endl
              << fmt::format("admissibility floor applied to {} cell updates over {} iterations", limited_cells, nt) << std::endl;

    metrics.report(metrics_file);

    samurai::finalize();
    return 0;
}
