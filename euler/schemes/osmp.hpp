// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "flux.hpp"
#include "riemann.hpp"

// =============================================================================
//  OSMPo: one-step monotonicity-preserving scheme of order o = 2 or 3
// -----------------------------------------------------------------------------
//  V. Daru, C. Tenaud, High order one-step monotonicity-preserving schemes for
//  unsteady compressible flow calculations, J. Comput. Phys. 193 (2004) 563-594.
//
//  A Lax-Wendroff type scheme: time and space are discretized together, so one
//  update of a sweep is third order in both for a scalar linear equation. The
//  flux is the first-order Roe flux plus, for each characteristic wave k of the
//  Roe matrix of the interface, a correction (eq. 64)
//
//      F = F_roe + 1/2 sum_k Phi_k (1 - |nu_k|) delta|f_k| r_k,
//      delta|f_k| = |lambda_k| delta alpha_k,   nu_k = lambda_k dt / dx,
//
//  where Phi_k is the accuracy function of order o (with nu -> |nu_k| and r ->
//  the ratio of the upwind wave strength to the local one): Lax-Wendroff at
//  order 2, eq. 7 at order 3,
//
//      Phi2 = 1,   Phi3 = 1 - (1 + |nu|) / 3 (1 - r),
//
//  bounded by the monotonicity-preserving constraints written on the flux
//  differences (eq. 67-71, 37-40), with the curvature d^M4 of eq. 40.
//
//  All the constraints are compared in the units of Phi, Phi = 2 delta f /
//  ((1 - |nu|) delta|f|), which removes every division by r or nu.
//
//  Like the article, the system is handled wave by wave in the eigenvector basis
//  of the interface (section 3.1.1): the wave strengths of the neighbouring
//  interfaces are the differences of the conservative variables projected on
//  the left eigenvectors of this interface, and their speeds are the
//  eigenvalues of their own Roe matrix. The scheme is then second order for the
//  system, with a very low error level. The multidimensional case is a Strang
//  splitting (section 3.2), so the scheme is meant to be run sweep by sweep.
//
//  Six cells per interface, numbered 0 to 5, the flux being computed between
//  cells 2 and 3:
//
//      |  0  |  1  |  2  |  3  |  4  |  5  |
//                        ^
//  Interfaces 0 to 4 separate the cells i and i + 1; this one is interface 2.
//  An interface reads the curvature of the upwind interface, which reads one
//  more cell on each side.
//
//  Harten's entropy fix is applied to the acoustic waves.
//
//  Not in the article: a positivity fallback to the first-order HLLC flux, see
//  OsmpOptions::positivity. Without it the scheme fails on the Mach 10 double
//  Mach reflection (a negative pressure near the wall), at order 2 and with the
//  TVD constraints at order 3.
// =============================================================================

struct OsmpOptions
{
    // The time step of the sweep, read through this pointer (see MusclOptions).
    std::shared_ptr<const double> dt;

    // The order o of the accuracy function Phi^o: 1 (Roe alone), 2 (Lax-Wendroff, Phi^2 = 1), 3 (eq. 7)
    // to 7 (eq. 10-11, with a stencil of 8 cells).
    int order = 3;

    // The constraints that bound Phi: the monotonicity-preserving ones (OSMP, eq. 37-40), or the
    // TVD ones only (OSTVD, eq. 66).
    bool tvd = false;

    // No constraint at all: the unlimited OSo scheme of section 2.1.1 (for convergence studies).
    bool unlimited = false;

    // The final step of the limiter as in samurai's demo (demos/FiniteVolume/convection_nonlinear_osmp.hpp):
    // Phi^o is kept when it lies in the MP interval [Phi_min, Phi_max], and is otherwise bounded by the TVD
    // constraints, instead of being brought back into the MP interval (eq. 37).
    bool demo_limiter = false;

    // The wave strengths of the neighbouring interfaces in their own Roe basis (Harten-Yee), instead of
    // projected on the basis of this interface.
    bool own_basis = false;

    // Positivity fallback, which the article does not have: the interface falls back to the
    // first-order HLLC flux when its flux would make the density or the pressure of one of its two
    // cells negative (test of Hu, Adams and Shu: each cell alone against twice this flux).
    // The number of fallbacks is counted in `fallbacks` when it is set.
    bool positivity = true;

    // The first-order flux the correction is added to: Roe (the article) or HLLC. With HLLC the
    // correction removes the HLLC dissipation projected on the Roe waves, q_k, instead of the Roe one:
    //     F = F_hllc + 1/2 sum_k Phi_k (q_k - |nu_k| |lambda_k| delta alpha_k) r_k,
    //     q_k = l_k . 2 (F_centred - F_hllc).
    // Phi = 1 is still Lax-Wendroff, Phi = 0 is HLLC, and q_k = |lambda_k| delta alpha_k gives back the article.
    bool hllc_base = false;
    std::shared_ptr<std::size_t> fallbacks;

    // Harten's entropy fix on the acoustic waves, as a fraction of |u_n| + c.
    double entropy_fix = 0.1;

    // Apply the entropy fix to the linear waves (entropy, shear) as well, against the carbuncle of
    // the Roe flux on strong shocks nearly aligned with the mesh.
    bool entropy_fix_all_waves = false;

    // The scheme is one-step and directional: it has to be restricted to one
    // direction and run inside a Strang splitting.
    int direction = -1;
};

namespace osmp
{
    inline double minmod(double a, double b)
    {
        if (a * b <= 0.)
        {
            return 0.;
        }
        return std::abs(a) < std::abs(b) ? a : b;
    }

    inline double minmod4(double a, double b, double c, double d)
    {
        return minmod(minmod(a, b), minmod(c, d));
    }

    // The curvature d^M4 at an interface from the curvatures of its two cells (eq. 40).
    inline double curvature_m4(double d_left, double d_right)
    {
        return minmod4(4. * d_left - d_right, 4. * d_right - d_left, d_left, d_right);
    }

    // The Roe average of two states, with the eigen-decomposition of the 2D
    // Euler flux in direction d. Waves: 0 u_n - c, 1 entropy, 2 shear, 3 u_n + c.
    template <std::size_t d>
    struct RoeBasis
    {
        static constexpr std::size_t dim = 2;
        static constexpr std::size_t t   = 1 - d; // the tangential direction
        using layout                     = EulerLayout<dim>;

        double gamma;
        double u[dim];
        double H;
        double c;
        std::array<double, 4> lambda;

        template <class Eos>
        RoeBasis(const PrimState<dim>& L, const PrimState<dim>& R, Eos eos)
            : gamma(eos.gamma)
        {
            const double sl = std::sqrt(L.rho);
            const double sr = std::sqrt(R.rho);
            const double HL = (L.rho * eos.e(L.rho, L.p) + L.p) / L.rho + 0.5 * (L.v[0] * L.v[0] + L.v[1] * L.v[1]);
            const double HR = (R.rho * eos.e(R.rho, R.p) + R.p) / R.rho + 0.5 * (R.v[0] * R.v[0] + R.v[1] * R.v[1]);
            for (std::size_t k = 0; k < dim; ++k)
            {
                u[k] = (sl * L.v[k] + sr * R.v[k]) / (sl + sr);
            }
            H                = (sl * HL + sr * HR) / (sl + sr);
            const double ke  = 0.5 * (u[0] * u[0] + u[1] * u[1]);
            c                = std::sqrt(std::max((gamma - 1.) * (H - ke), 1e-300));
            lambda           = {u[d] - c, u[d], u[d], u[d] + c};
        }

        // The wave strengths of the conservative difference dw: the left eigenvectors applied to it.
        std::array<double, 4> strengths(const ConsArray<dim>& dw) const
        {
            const double b1  = (gamma - 1.) / (c * c);
            const double b2  = 0.5 * (u[0] * u[0] + u[1] * u[1]) * b1;
            const double dr  = dw[layout::rho];
            const double dmn = dw[layout::mom(d)];
            const double dmt = dw[layout::mom(t)];
            const double dE  = dw[layout::rhoE];
            const double udm = u[0] * dw[layout::mom(0)] + u[1] * dw[layout::mom(1)];

            return {0.5 * (b2 * dr - b1 * udm + b1 * dE - (dmn - u[d] * dr) / c),
                    (1. - b2) * dr + b1 * udm - b1 * dE,
                    dmt - u[t] * dr,
                    0.5 * (b2 * dr - b1 * udm + b1 * dE + (dmn - u[d] * dr) / c)};
        }

        // The right eigenvector of wave k.
        ConsArray<dim> eigenvector(std::size_t k) const
        {
            ConsArray<dim> r;
            const double ke = 0.5 * (u[0] * u[0] + u[1] * u[1]);
            switch (k)
            {
                case 0:
                    r[layout::rho]    = 1.;
                    r[layout::mom(d)] = u[d] - c;
                    r[layout::mom(t)] = u[t];
                    r[layout::rhoE]   = H - u[d] * c;
                    break;
                case 1:
                    r[layout::rho]    = 1.;
                    r[layout::mom(d)] = u[d];
                    r[layout::mom(t)] = u[t];
                    r[layout::rhoE]   = ke;
                    break;
                case 2:
                    r[layout::rho]    = 0.;
                    r[layout::mom(d)] = 0.;
                    r[layout::mom(t)] = 1.;
                    r[layout::rhoE]   = u[t];
                    break;
                default:
                    r[layout::rho]    = 1.;
                    r[layout::mom(d)] = u[d] + c;
                    r[layout::mom(t)] = u[t];
                    r[layout::rhoE]   = H + u[d] * c;
                    break;
            }
            return r;
        }

        // |lambda_k|, with Harten's entropy fix on the acoustic waves.
        double speed(std::size_t k, double entropy_fix, bool all_waves) const
        {
            const double a = std::abs(lambda[k]);
            if (!all_waves && (k == 1 || k == 2))
            {
                return a;
            }
            const double delta = entropy_fix * (std::abs(u[d]) + c);
            return (a >= delta || delta <= 0.) ? a : 0.5 * (lambda[k] * lambda[k] + delta * delta) / delta;
        }
    };

    // The limited accuracy function Phi of wave k at the interface (eq. 7 and 37-40, written for systems in eq. 66-71).
    //  nu, nu_up: |nu| at this interface and at the upwind one
    //  g:         delta|f| at this interface (same sign as the wave strength)
    //  d_md:      the curvature d^M4 at this interface, d_lc the one at the upwind interface
    //  r:         the upwind wave strength over this one
    /**
     * The accuracy function Phi^o of eq. 7, 10 and 11, o = 1 (Roe alone) to 7, for |nu| = nu.
     * a[m + 3] is the wave strength m interfaces downwind of this one (m = -3 ... 2, a[3] being this one):
     * the ratios r of the article are ratios of consecutive strengths, so each bracket of eq. 10-11 is
     * a finite difference of the strengths divided by a_0.
     */
    inline double accuracy_function(int order, double nu, const std::array<double, 6>& a)
    {
        if (order <= 1)
        {
            return 0.;
        }
        if (order == 2)
        {
            return 1.;
        }
        const double am3 = a[0], am2 = a[1], am1 = a[2], a0 = a[3], a1 = a[4], a2 = a[5];
        const double c3  = (1. + nu) / 3.;
        double phi       = 1. - c3 * (a0 - am1) / a0; // eq. 7: 1 - (1 + nu) / 3 (1 - r)
        const double c4  = c3 * (nu - 2.) / 4.;
        if (order >= 4)
        {
            phi += c4 * (a0 - 2. * am1 + am2) / a0; // eq. 10
        }
        const double c5 = c4 * (nu - 3.) / 5.;
        if (order >= 5)
        {
            phi -= c5 * (a1 - 3. * a0 + 3. * am1 - am2) / a0; // eq. 11
        }
        const double c6 = c5 * (nu + 2.) / 6.;
        if (order >= 6)
        {
            phi += c6 * (a2 - 4. * a1 + 6. * a0 - 4. * am1 + am2) / a0;
        }
        const double c7 = c6 * (nu + 3.) / 7.;
        if (order >= 7)
        {
            phi -= c7 * (a2 - 5. * a1 + 10. * a0 - 10. * am1 + 5. * am2 - am3) / a0;
        }
        return phi;
    }

    inline double limited_phi(double phi_o, bool tvd, bool demo, double nu, double nu_up, double g, double d_md, double d_lc, double r)
    {

        // delta f^ul (eq. 68), delta f^md (eq. 69), delta f^lc (eq. 70)
        const double f_ul = r * (1. - nu_up) * g / nu;
        const double f_md = 0.5 * g - 0.5 * d_md;
        const double f_lc = 0.5 * f_ul + 0.5 * (1. - nu_up) / nu * d_lc;

        // In the units of Phi: Phi = 2 delta f / ((1 - |nu|) delta|f|)
        const double unit   = 2. / ((1. - nu) * g);
        const double phi_1  = 2. / (1. - nu); // the TVD bound F in [f_j, f_j+1]
        const double phi_ul = unit * f_ul;    // the TVD bound F in [f_j, f_ul]
        const double phi_md = unit * f_md;
        const double phi_lc = unit * f_lc;

        if (tvd)
        {
            return std::max(0., std::min({phi_ul, phi_o, phi_1}));
        }

        const double phi_min = std::max(std::min(0., phi_md), std::min({0., phi_ul, phi_lc}));
        const double phi_max = std::min(std::max(phi_1, phi_md), std::max({0., phi_ul, phi_lc}));
        if (demo)
        {
            if (phi_o >= phi_min && phi_o <= phi_max)
            {
                return phi_o;
            }
            return std::max(0., std::min({phi_ul, phi_o, phi_1}));
        }
        return std::max(phi_min, std::min(phi_o, phi_max));
    }
}

// stencil_size: 6 up to order 3, 8 from order 4 (the accuracy functions of order 4 to 7 read three
// interfaces upwind and two downwind).
template <class Field, std::size_t stencil_size = 6, class Eos>
auto make_osmp_scheme(Eos eos, const OsmpOptions& options)
{
    static constexpr std::size_t dim = Field::dim;
    static_assert(stencil_size == 6 || stencil_size == 8);
    assert((stencil_size == 8 || options.order <= 3) && "OSMP of order 4 to 7 needs a stencil of 8 cells");
    // This interface separates the cells c and c + 1; there are stencil_size - 1 interfaces.
    static constexpr std::size_t c  = stencil_size / 2 - 1;
    static constexpr std::size_t ni = stencil_size - 1;
    static_assert(dim == 2, "OSMP3 is written for the 2D Euler equations");

    assert(options.dt && "the one-step scheme needs the time step of the sweep");
    assert(options.direction >= 0 && "OSMP3 is directional: run it in a Strang splitting");

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> osmp_flux;

    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            if (d != static_cast<std::size_t>(options.direction))
            {
                return;
            }

            osmp_flux[d].cons_flux_function =
                [eos, options](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& field)
            {
                const double dt_dx = (*options.dt) / data.cell_length;

                std::array<PrimState<dim>, stencil_size> prim;
                for (std::size_t i = 0; i < stencil_size; ++i)
                {
                    prim[i] = cons2prim<dim>(field[i], eos);
                }

                // The basis of this interface, and the speeds of the waves at the five interfaces.
                const osmp::RoeBasis<d> basis(prim[c], prim[c + 1], eos);
                std::array<std::array<double, 4>, ni> speed;    // |lambda|, entropy-fixed
                std::array<std::array<double, 4>, ni> strength; // delta alpha, in the basis of this interface
                for (std::size_t i = 0; i < ni; ++i)
                {
                    const ConsArray<dim> dw = field[i + 1] - field[i];
                    strength[i]             = basis.strengths(dw);
                    if (i == c)
                    {
                        for (std::size_t k = 0; k < 4; ++k)
                        {
                            speed[i][k] = basis.speed(k, options.entropy_fix, options.entropy_fix_all_waves);
                        }
                    }
                    else
                    {
                        const osmp::RoeBasis<d> other(prim[i], prim[i + 1], eos);
                        if (options.own_basis)
                        {
                            strength[i] = other.strengths(dw);
                        }
                        for (std::size_t k = 0; k < 4; ++k)
                        {
                            speed[i][k] = other.speed(k, options.entropy_fix, options.entropy_fix_all_waves);
                        }
                    }
                }

                // First-order flux: Roe (eq. 65), or HLLC with its dissipation projected on the Roe waves
                const ConsArray<dim> centred = 0.5 * (compute_flux<d>(prim[c], eos) + compute_flux<d>(prim[c + 1], eos));
                ConsArray<dim> F             = centred;
                std::array<double, 4> q{};
                if (options.hllc_base)
                {
                    F = riemann::solve<riemann::Solver::hllc, d>(prim[c], field[c], prim[c + 1], field[c + 1], eos);
                    const ConsArray<dim> twice_dissipation = 2. * (centred - F);
                    q                                      = basis.strengths(twice_dissipation);
                }

                for (std::size_t k = 0; k < 4; ++k)
                {
                    // delta|f| at each interface, and the curvature of each inner cell (eq. 71)
                    std::array<double, ni> g;
                    for (std::size_t i = 0; i < ni; ++i)
                    {
                        g[i] = speed[i][k] * strength[i][k];
                    }
                    std::array<double, stencil_size> curv{};
                    for (std::size_t m = 1; m < ni; ++m)
                    {
                        curv[m] = g[m] - g[m - 1];
                    }

                    const ConsArray<dim> rk = basis.eigenvector(k);
                    if (!options.hllc_base)
                    {
                        F -= 0.5 * g[c] * rk;
                        q[k] = g[c];
                    }

                    const double nu = speed[c][k] * dt_dx;
                    if (g[c] == 0. || nu <= 0. || nu >= 1.)
                    {
                        continue;
                    }

                    // The upwind interface: c - 1 for a right-going wave, c + 1 for a left-going one.
                    const bool right_going = basis.lambda[k] >= 0.;
                    const std::size_t up   = right_going ? c - 1 : c + 1;
                    const double r         = strength[up][k] / strength[c][k];
                    const double nu_up   = std::min(speed[up][k] * dt_dx, 1.);

                    const double d_md = osmp::curvature_m4(curv[c], curv[c + 1]);
                    const double d_lc = osmp::curvature_m4(curv[up], curv[up + 1]);

                    // The strengths from 3 interfaces upwind to 2 downwind, in the direction of the wave
                    std::array<double, 6> a{};
                    for (int m = -3; m <= 2; ++m)
                    {
                        const int i = static_cast<int>(c) + (right_going ? m : -m);
                        if (i >= 0 && i < static_cast<int>(ni))
                        {
                            a[static_cast<std::size_t>(m + 3)] = strength[static_cast<std::size_t>(i)][k];
                        }
                    }
                    const double phi_o = osmp::accuracy_function(options.order, nu, a);

                    const double phi = options.unlimited ? phi_o : osmp::limited_phi(phi_o, options.tvd, options.demo_limiter, nu, nu_up, g[c], d_md, d_lc, r);

                    F += 0.5 * phi * (q[k] - nu * g[c]) * rk;
                }

                if (options.positivity)
                {
                    const double two_dt_dx = 2. * dt_dx;
                    auto admissible        = [&](const ConsArray<dim>& w)
                    {
                        if (!(w[EulerLayout<dim>::rho] > 0.))
                        {
                            return false;
                        }
                        double kinetic = 0.;
                        for (std::size_t k = 0; k < dim; ++k)
                        {
                            kinetic += w[EulerLayout<dim>::mom(k)] * w[EulerLayout<dim>::mom(k)];
                        }
                        return w[EulerLayout<dim>::rhoE] - 0.5 * kinetic / w[EulerLayout<dim>::rho] > 0.;
                    };
                    const ConsArray<dim> left  = field[c] - two_dt_dx * (F - compute_flux<d>(prim[c], eos));
                    const ConsArray<dim> right = field[c + 1] + two_dt_dx * (F - compute_flux<d>(prim[c + 1], eos));
                    if (!admissible(left) || !admissible(right))
                    {
                        F = riemann::solve<riemann::Solver::hllc, d>(prim[c], field[c], prim[c + 1], field[c + 1], eos);
                        if (options.fallbacks)
                        {
                            ++(*options.fallbacks);
                        }
                    }
                }

                flux = F;
            };
        });

    auto scheme = make_flux_based_scheme(osmp_flux);
    scheme.set_name("osmp" + std::to_string(options.order));

    return scheme;
}
