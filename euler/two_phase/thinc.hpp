// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/stencil.hpp>

#include "variables.hpp"

// =============================================================================
//  THINC interface sharpening
// -----------------------------------------------------------------------------
//  Tangent of Hyperbola for INterface Capturing. The diffuse interface of the
//  five-equation model spreads over ten cells and keeps spreading; THINC holds
//  it on two or three by reconstructing the volume fraction inside a mixed cell
//  as a hyperbolic tangent instead of a limited straight line:
//
//      alpha_i(X) = 1/2 [ 1 + tanh( beta (sigma X + x_c) ) ],   X in [0, 1]
//
//  X is the position across the cell, sigma = sign(alpha_{i+1} - alpha_{i-1})
//  says which way the interface faces, beta how steep it is, and x_c where it
//  sits inside the cell. Only x_c is unknown, and it is fixed by the one thing
//  the reconstruction must not break: its average over the cell has to be the
//  cell average it came from. Integrating the profile,
//
//      alpha_bar = 1/2 + 1/(2 beta sigma) ln( cosh(beta(sigma + x_c)) / cosh(beta x_c) )
//
//  and solving for tanh(beta x_c) gives the closed form below -- no iteration,
//  and conservation is exact rather than approached.
//
//      F. Xiao, Y. Honma, T. Kono, "A simple algebraic interface capturing
//      scheme using hyperbolic tangent function", Int. J. Numer. Methods Fluids
//      48 (2005) 1023-1040, https://doi.org/10.1002/fld.975
//
//      K.-M. Shyue, F. Xiao, "An Eulerian interface sharpening algorithm for
//      compressible two-phase flow: the algebraic THINC approach", J. Comput.
//      Phys. 268 (2014) 326-354, https://doi.org/10.1016/j.jcp.2014.03.010
//
//  It applies to the volume fraction alone. Everything else at that face --
//  pressure, velocity, the two partial densities -- keeps its MUSCL
//  reconstruction, and the total energy is rebuilt from the sharpened alpha by
//  `prim2cons`, which is what keeps the interface condition exact: the mixture
//  law is linear in alpha, so a uniform pressure survives whatever alpha does.
//
//  The steepness is not the same in every direction. The article weights it with
//  the interface normal,
//
//      beta_d = beta |n_d| + 0.001,      n = grad(psi) / |grad(psi)|
//
//  with psi = alpha^m / (alpha^m + (1-alpha)^m) and m = 0.1 a smoothing of the
//  volume fraction that makes the gradient usable in nearly pure cells. A face
//  whose normal the interface is parallel to gets beta = 0.001 and no
//  sharpening, which is what stops the scheme from carving steps into an
//  interface that runs across the mesh.
//
//  The normal needs the gradient in every direction, and a flux stencil is a
//  line, so it cannot be computed where it is used. It is computed once per time
//  step over the whole mesh, into the field below, and read from there.
// =============================================================================

namespace two_phase
{
    // How sharp, and how mixed a cell has to be for any of this to apply.
    struct ThincOptions
    {
        bool enabled = false;

        // The steepness. 1.6 to 2.2 in the literature; above that the profile is
        // a step inside one cell and the interface starts to break into pieces.
        double beta = 1.6;

        // Below this distance from a pure fluid the cell is not an interface and
        // is left to the MUSCL reconstruction.
        double alpha_min = 1e-6;
    };

    // The smoothed volume fraction whose gradient gives the interface normal.
    // The exponent is small on purpose: psi saturates to 0 and 1 away from the
    // interface, so the gradient sees the interface and not the tails.
    inline double interface_function(double alpha, double m = 0.1)
    {
        const double a = std::pow(std::clamp(alpha, 0., 1.), m);
        const double b = std::pow(std::clamp(1. - alpha, 0., 1.), m);
        return (a + b) > 0. ? a / (a + b) : 0.5;
    }

    // |n_d| for every cell and every direction, from a centred difference of psi.
    // Ghosts keep the 1 they are filled with, so a mixed cell against a boundary
    // sharpens as if the interface faced it.
    //
    // Call it with up-to-date ghosts: the stencil reads the neighbours of every
    // cell, and at a level jump those are ghosts the multiresolution fills.
    template <class Field, class NormalField>
    void compute_interface_normals(const Field& u, NormalField& normals)
    {
        static constexpr std::size_t dim = Field::dim;

        normals.resize();
        normals.fill(1.);

        // One pass per direction, each on the three cells of a line, so that the
        // component being written is the one the stencil is aligned with. The
        // raw derivative goes in first and the whole vector is normalised after,
        // there being no way to normalise before every component is known.
        samurai::static_for<0, dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = integral_constant_d();

                auto stencil = samurai::make_stencil_analyzer(samurai::line_stencil<dim, d>(-1, 0, 1));

                samurai::for_each_stencil(u.mesh(),
                                          stencil,
                                          [&](const auto& cells)
                                          {
                                              const double behind = interface_function(u[cells[0]][Layout<dim>::alpha]);
                                              const double ahead  = interface_function(u[cells[2]][Layout<dim>::alpha]);

                                              normals[cells[1]][d] = 0.5 * (ahead - behind);
                                          });
            });

        samurai::for_each_cell(u.mesh(),
                               [&](const auto& cell)
                               {
                                   double norm = 0.;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       norm += normals[cell][d] * normals[cell][d];
                                   }
                                   norm = std::sqrt(norm);

                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       // A cell with no gradient at all is not an
                                       // interface; what it gets here is never
                                       // used, the mixed test below rejecting the
                                       // cell first.
                                       normals[cell][d] = norm > 0. ? std::abs(normals[cell][d]) / norm : 1.;
                                   }
                               });
    }

    // The volume fraction at the two faces of a cell, from its average and the
    // side its neighbours put the interface on.
    //
    // Returns false when the cell is not one THINC applies to: a pure fluid, or
    // a cell whose two neighbours do not agree on which way the fraction goes.
    // A non-monotone cell is an extremum, and sharpening an extremum turns it
    // into a step.
    struct ThincProfile
    {
        double left;
        double right;

        double slope() const
        {
            return right - left;
        }
    };

    inline bool thinc_profile(double behind, double average, double ahead, double beta, const ThincOptions& options, ThincProfile& profile)
    {
        if (average < options.alpha_min || average > 1. - options.alpha_min)
        {
            return false;
        }
        if ((ahead - average) * (average - behind) <= 0.)
        {
            return false;
        }

        const double sigma = ahead > behind ? 1. : -1.;

        // tanh(beta x_c) in closed form. The exponential is of beta (2 a - 1),
        // which stays well inside the range of a double for any beta a scheme
        // would use.
        const double t = (std::exp(beta * sigma * (2. * average - 1.)) / std::cosh(beta) - 1.) / (sigma * std::tanh(beta));
        if (!std::isfinite(t) || std::abs(t) >= 1.)
        {
            return false;
        }

        const double x_c = std::atanh(t) / beta;

        profile.left  = 0.5 * (1. + std::tanh(beta * x_c));
        profile.right = 0.5 * (1. + std::tanh(beta * (sigma + x_c)));

        return true;
    }
}
