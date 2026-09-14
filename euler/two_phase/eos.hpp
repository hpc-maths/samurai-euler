// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>

#include "../eos.hpp"

// =============================================================================
//  Mixture equation of state of the five-equation model
// -----------------------------------------------------------------------------
//  Two stiffened gases sharing a cell, in pressure and velocity equilibrium.
//  Their volume fractions alpha_0 and alpha_1 = 1 - alpha_0 say how much of the
//  cell each one occupies, and the mixture behaves as a single stiffened gas
//  whose coefficients depend on alpha:
//
//      1 / (gamma_m - 1) = sum_i alpha_i / (gamma_i - 1)                    (G)
//      gamma_m pi_m / (gamma_m - 1) = sum_i alpha_i gamma_i pi_i / (gamma_i - 1)
//
//  Those two sums are the whole equation of state. Written as G(alpha) and
//  P(alpha) they give the internal energy and the pressure directly,
//
//      rho e = G p + P          p = (rho e - P) / G
//
//  and the sound speed is the stiffened-gas one at the mixture coefficients,
//
//      c^2 = gamma_m (p + pi_m) / rho,   gamma_m = 1 + 1/G,  pi_m = P / (G gamma_m)
//
//  which is the frozen speed the Allaire model is hyperbolic for. A single phase
//  (alpha = 0 or 1) gives back exactly its own gamma and pi_inf, which is worth
//  keeping in mind: a two-phase run on a pure fluid must reproduce the monofluid
//  solver bit for bit up to the scheme, and that is a test.
//
//      G. Allaire, S. Clerc, S. Kokh, "A five-equation model for the simulation
//      of interfaces between compressible fluids", J. Comput. Phys. 181 (2)
//      (2002) 577-616, https://doi.org/10.1006/jcph.2002.7143
// =============================================================================

namespace EOS
{
    struct Mixture
    {
        StiffenedGas phase[2] = {
            {1.4, 0., 0.},
            {1.4, 0., 0.}
        };

        // sum_i alpha_i / (gamma_i - 1), the inverse of gamma_m - 1.
        double G(double alpha0) const
        {
            const double alpha1 = 1. - alpha0;
            return alpha0 / (phase[0].gamma - 1.) + alpha1 / (phase[1].gamma - 1.);
        }

        // sum_i alpha_i gamma_i pi_i / (gamma_i - 1).
        double P(double alpha0) const
        {
            const double alpha1 = 1. - alpha0;
            return alpha0 * phase[0].gamma * phase[0].pi_inf / (phase[0].gamma - 1.)
                 + alpha1 * phase[1].gamma * phase[1].pi_inf / (phase[1].gamma - 1.);
        }

        double gamma(double alpha0) const
        {
            return 1. + 1. / G(alpha0);
        }

        double pi_inf(double alpha0) const
        {
            const double g = G(alpha0);
            return P(alpha0) / (g * (1. + 1. / g));
        }

        // Pressure from the volumic internal energy rho e.
        double p(double alpha0, double rho_e) const
        {
            return (rho_e - P(alpha0)) / G(alpha0);
        }

        // The inverse: the volumic internal energy at that pressure.
        double rho_e(double alpha0, double p) const
        {
            return G(alpha0) * p + P(alpha0);
        }

        double c(double alpha0, double rho, double p) const
        {
            return std::sqrt(gamma(alpha0) * (p + pi_inf(alpha0)) / rho);
        }
    };

    inline constexpr Mixture two_ideal_gases(double gamma0, double gamma1)
    {
        return {
            {{gamma0, 0., 0.}, {gamma1, 0., 0.}}
        };
    }
}
