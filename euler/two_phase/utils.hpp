// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <string>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>

#include "variables.hpp"

// =============================================================================
//  What the two-phase time loop needs around the scheme
// -----------------------------------------------------------------------------
//  The time step, the admissibility floor and the output. The monofluid versions
//  of the three read rho and rhoE straight out of the state and cannot be reused
//  as they are: here the density is a sum of two, and the equation of state
//  takes the volume fraction.
// =============================================================================

namespace two_phase
{
    // The largest wave speed over the mesh and over the directions, which is
    // what the time step is built on.
    template <class Field>
    double get_max_lambda(const Field& u, const EOS::Mixture& eos)
    {
        static constexpr std::size_t dim = Field::dim;

        double res = 0.;
        samurai::for_each_cell(u.mesh(),
                               [&](const auto& cell)
                               {
                                   const auto prim = cons2prim<dim>(u[cell], eos);
                                   const double c  = eos.c(prim.alpha, prim.rho(), prim.p);
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       res = std::max(std::abs(prim.v[d]) + c, res);
                                   }
                               });
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        return mpi::all_reduce(world, res, mpi::maximum<double>());
#else
        return res;
#endif
    }

    // The admissible set of the five-equation model: a volume fraction in
    // [0, 1], two non-negative partial densities, a positive mixture density and
    // a pressure above the vacuum of the mixture, p + pi_inf(alpha) > 0, which is
    // what the sound speed needs.
    //
    // A pure fluid sits exactly at alpha = 0 or 1 with one partial density
    // exactly zero, and that is legitimate: the floor clamps what falls outside
    // and leaves the boundary of the set alone. Like the monofluid one it
    // returns how many cells it touched, and the projection is not conservative.
    template <class Field>
    std::size_t limit_positivity(Field& u, const EOS::Mixture& eos, double rho_min = 1e-12, double p_min = 1e-12)
    {
        static constexpr std::size_t dim = Field::dim;
        using Var                        = Layout<dim>;

        std::size_t limited = 0;

        samurai::for_each_cell(u.mesh(),
                               [&](const auto& cell)
                               {
                                   bool touched = false;

                                   double alpha = u[cell][Var::alpha];
                                   if (alpha < 0. || alpha > 1.)
                                   {
                                       alpha               = std::clamp(alpha, 0., 1.);
                                       u[cell][Var::alpha] = alpha;
                                       touched             = true;
                                   }

                                   for (std::size_t k = 0; k < 2; ++k)
                                   {
                                       if (u[cell][Var::partial_rho(k)] < 0.)
                                       {
                                           u[cell][Var::partial_rho(k)] = 0.;
                                           touched                      = true;
                                       }
                                   }

                                   double rho = u[cell][Var::partial_rho(0)] + u[cell][Var::partial_rho(1)];
                                   if (rho < rho_min)
                                   {
                                       // Give the missing mass to the phase the
                                       // cell is made of, so that alpha is left
                                       // saying what it said.
                                       const std::size_t k = alpha >= 0.5 ? 0 : 1;
                                       u[cell][Var::partial_rho(k)] += rho_min - rho;
                                       rho     = rho_min;
                                       touched = true;
                                   }

                                   double kinetic = 0.;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       const double mom = u[cell][Var::mom(d)];
                                       kinetic += 0.5 * mom * mom / rho;
                                   }

                                   const double p = eos.p(alpha, u[cell][Var::rhoE] - kinetic);
                                   if (p + eos.pi_inf(alpha) < p_min)
                                   {
                                       u[cell][Var::rhoE] = eos.rho_e(alpha, p_min - eos.pi_inf(alpha)) + kinetic;
                                       touched            = true;
                                   }

                                   limited += touched ? 1 : 0;
                               });

        return limited;
    }

    // What a two-phase run writes: the volume fraction the interface is read
    // from, the mixture density and pressure, the velocity, and the two partial
    // densities.
    //
    // The partial densities are in there because they are what the model
    // conserves, and alpha * rho is not one of them: in a mixed cell the mass of
    // each phase is alpha_i rho_i, which the mixture density and the volume
    // fraction together cannot give back. Anything checking conservation needs
    // them, and so does anyone reading a mixed cell.
    template <class Field>
    void save(const std::string& path, const std::string& filename, const Field& u, const EOS::Mixture& eos)
    {
        static constexpr std::size_t dim = Field::dim;

        auto& mesh    = u.mesh();
        auto alpha    = samurai::make_scalar_field<double>("alpha", mesh);
        auto rho      = samurai::make_scalar_field<double>("rho", mesh);
        auto rho_0    = samurai::make_scalar_field<double>("partial_rho_0", mesh);
        auto rho_1    = samurai::make_scalar_field<double>("partial_rho_1", mesh);
        auto pressure = samurai::make_scalar_field<double>("pressure", mesh);
        auto velocity = samurai::make_vector_field<double, dim>("velocity", mesh);

        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   const auto prim = cons2prim<dim>(u[cell], eos);
                                   alpha[cell]     = prim.alpha;
                                   rho[cell]       = prim.rho();
                                   rho_0[cell]     = prim.partial_rho[0];
                                   rho_1[cell]     = prim.partial_rho[1];
                                   pressure[cell]  = prim.p;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       velocity[cell][d] = prim.v[d];
                                   }
                               });

        samurai::save(path, filename, mesh, alpha, rho, rho_0, rho_1, pressure, velocity);
    }
}
