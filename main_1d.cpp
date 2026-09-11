// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#include <cassert>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "euler/config.hpp"
#include "euler/eos.hpp"
#include "euler/init/cases.hpp"
#include "euler/save.hpp"
#include "euler/schemes.hpp"
#include "euler/utils.hpp"
#include "euler/variables.hpp"

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;

    using field_t = config<dim>::field_t;

    auto& app = samurai::initialize("Euler equations solver (1D)", argc, argv);

    double Tf  = .15;
    double cfl = 0.4;
    double t   = 0.;
    std::string restart_file;
    std::string scheme    = "hll";
    std::string test_case = "double_rarefaction";
    double gamma          = 0.; // only used when --gamma is given

    // Output parameters
    fs::path path = "results";
    std::string filename;
    std::size_t nfiles = 1;

    auto available = test_case::TestCaseRegistry<field_t>::instance().available_test_cases();

    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--scheme", scheme, "Finite volume scheme")
        ->capture_default_str()
        ->check(CLI::IsMember({"rusanov", "hll", "hllc"}))
        ->group("Simulation parameters");
    app.add_option("--test-case", test_case, "Test case")->capture_default_str()->check(CLI::IsMember(available))->group("Simulation parameters");
    auto* gamma_opt = app.add_option("--gamma", gamma, "Ratio of specific heats (defaults to the value of the test case)")
                          ->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix (defaults to <test-case>_<scheme>)")->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    std::cout << "Samurai version: " << SAMURAI_VERSION << std::endl; // Print Samurai version info

    const auto& selected = test_case::TestCaseRegistry<field_t>::instance().get(test_case);

    // The test case carries the gas it was designed for; --gamma overrides it.
    EOS::IdealGas eos = selected.eos;
    if (gamma_opt->count() > 0)
    {
        eos.gamma = gamma;
    }
    std::cout << "Using gamma = " << eos.gamma << std::endl;

    if (filename.empty())
    {
        filename = fmt::format("{}_{}", test_case, scheme);
    }

    // Initialize the mesh
    auto box = selected.box();

    auto config = samurai::mesh_config<dim>().min_level(8).max_level(8).max_stencil_size(2).disable_minimal_ghost_width();
    config.periodic(selected.periodic);
    config.parse_args();

    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        u.resize();
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

    std::cout << config.min_level() << " " << config.max_level() << std::endl;

    selected.bc(u, t, eos);

    auto unp1 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    double dx            = mesh.cell_length(config.max_level());
    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 1;
    std::size_t nt       = 0;

    save(path.string(), fmt::format("{}_init", filename), u, eos);

    std::cout << "Using scheme: " << scheme << std::endl;
    auto fv_scheme = get_fv_scheme<decltype(u)>(scheme, eos);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().relative_detail(true);

    while (t != Tf)
    {
        MRadaptation(mra_config);

        double dt = cfl * dx / get_max_lambda(u, eos);
        t += dt;

        if (std::isnan(t))
        {
            std::cerr << "Error: Time became NaN, stopping simulation" << std::endl;
            break;
        }

        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        unp1.resize();
        unp1 = u - dt * fv_scheme(u);

        samurai::swap(u, unp1);

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path.string(), fmt::format("{}{}", filename, suffix), u, eos);
        }
    }

    samurai::finalize();
    return 0;
}
