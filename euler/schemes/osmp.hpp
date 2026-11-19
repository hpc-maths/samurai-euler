// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "flux.hpp"

// =============================================================================
//  OSMPo: one-step monotonicity-preserving schemes of order o = 1 to 7
// -----------------------------------------------------------------------------
//  V. Daru, C. Tenaud, High order one-step monotonicity-preserving schemes for
//  unsteady compressible flow calculations, J. Comput. Phys. 193 (2004) 563-594.
//
//  A Lax-Wendroff type scheme: time and space are discretized together, so one
//  update of a sweep has the same order in both for a scalar linear equation.
//  The flux is the first-order Roe flux plus, for each characteristic wave k of
//  the Roe matrix of the interface, a correction (eq. 64-65)
//
//      F = F_roe + sum_k dF_k r_k,     dF_k = 1/2 Phi_k (1 - |nu_k|) delta|f_k|,
//      delta|f_k| = |lambda_k| delta alpha_k,   nu_k = lambda_k dt / dx,
//
//  where Phi_k is the accuracy function of order o (eq. 7, 10, 11), bounded by
//  the monotonicity-preserving constraints (eq. 37-40, 67-71).
//
//  Everything is written on the flux differences, as the article does for
//  systems (eq. 67-70), and never on Phi itself: Phi is a ratio of wave
//  strengths, and at the foot of a wave these are as small as 1e-100, where a
//  ratio is inf or NaN and a NaN bound silently switches the limiter off. So
//  the accuracy function is kept multiplied by delta alpha (a polynomial in the
//  strengths of the stencil), and the bounds are compared after a
//  multiplication by |nu|, which removes the last division of eq. 70.
//
//  The wave strengths and the speeds of an interface are those of its own Roe
//  matrix (Harten and Yee), for this interface and for its neighbours alike.
//  The scheme is second order for the system, with a very low error level. The
//  multidimensional case is a Strang splitting (section 3.2), so the scheme is
//  meant to be run sweep by sweep.
//
//  Cells of the stencil, here for six of them, the flux being computed between
//  cells 2 and 3:
//
//      |  0  |  1  |  2  |  3  |  4  |  5  |
//                        ^
//  Interface i separates the cells i and i + 1; this one is interface 2. The
//  limiter reads the curvature at the upwind interface, hence two interfaces on
//  each side, which is also all the orders up to 6 read. Order 7 reads a third
//  interface upwind, and needs eight cells.
//
//  Entropy correction: the one of Khalfallah and Lerat (C. R. Acad. Sci. Paris
//  308 II, 1989), which the article uses. It adds to |lambda| of every wave of
//  an interface the same speed a >= 0, the least for which the first-order flux
//  satisfies Tadmor's entropy criterion: zero wherever the Roe flux is already
//  dissipative enough, which leaves an expansion through a sonic point. It has
//  no parameter, and it "vanishes with zero flow velocity" as the article says
//  of its dissipation, so it does nothing against the carbuncle of the Roe
//  flux: see OsmpOptions::harten_fix.
// =============================================================================

// What bounds the correction of each wave.
enum class OsmpLimiter
{
    mp,  // monotonicity-preserving constraints: OSMPo (eq. 37-40, 67-71)
    tvd, // TVD constraints only: OSTVDo (eq. 66), which clips the extrema
    none // no constraint: OSo (section 2.1.1), for convergence studies on smooth solutions
};

inline OsmpLimiter osmp_limiter_from_name(const std::string& name)
{
    if (name == "mp")
    {
        return OsmpLimiter::mp;
    }
    if (name == "tvd")
    {
        return OsmpLimiter::tvd;
    }
    if (name == "none")
    {
        return OsmpLimiter::none;
    }
    throw std::runtime_error("Unknown OSMP limiter: " + name);
}

struct OsmpOptions
{
    // The time step of the sweep, read through this pointer (see MusclOptions).
    std::shared_ptr<const double> dt;

    // The order o of the accuracy function Phi^o: 1 (Roe alone), 2 (Lax-Wendroff, Phi^2 = 1), 3 (eq. 7)
    // to 7 (eq. 10-11).
    int order = 3;

    OsmpLimiter limiter = OsmpLimiter::mp;

    // Harten's entropy fix on every wave, on top of the correction of Khalfallah and Lerat, as a
    // fraction of |u_n| + c: below it, |lambda| is replaced by a parabola. Zero switches it off.
    //
    // This is the cure for the carbuncle of the Roe flux, which kinks the Mach stem of the double
    // Mach reflection at order 2 and 3 from 512 x 128 cells on: 0.2 straightens it. It is off by
    // default because it is paid wherever the flow is slow across the sweep,
    // |u_n| < harten_fix (|u_n| + c): there the scheme loses its order.
    double harten_fix = 0.;

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
            H               = (sl * HL + sr * HR) / (sl + sr);
            const double ke = 0.5 * (u[0] * u[0] + u[1] * u[1]);
            // Positive for two admissible states. No clamp: an inadmissible state gives a NaN that
            // shows, where a clamped sound speed would carry on with a meaningless basis.
            c      = std::sqrt((gamma - 1.) * (H - ke));
            lambda = {u[d] - c, u[d], u[d], u[d] + c};
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

        // The largest speed of the interface: no wave needs more dissipation than this (Rusanov).
        double max_speed() const
        {
            return std::abs(u[d]) + c;
        }

        /**
         * The speed a of the entropy correction of Khalfallah and Lerat for the Roe flux between
         * the conservative states wL and wR, dw = wR - wL having the strengths `alpha`.
         *
         * With v the entropy variables of the entropy -rho s / (gamma - 1), s = log(p / rho^gamma),
         * and psi = rho u_n its potential flux, a flux h is entropy stable when dv.h <= d(psi)
         * (Tadmor; eq. 3.5 of the note with its lemma). Adding -a/2 dw to the flux lowers dv.h by
         * a/2 dv.dw, and dv.dw > 0, so the least a that restores the inequality is (eq. 4.7)
         *
         *     a = 2 max(dv.h_roe - d(psi), 0) / (dv.dw).
         */
        template <class Eos>
        double entropy_correction(const PrimState<dim>& L,
                                  const PrimState<dim>& R,
                                  const ConsArray<dim>& wL,
                                  const ConsArray<dim>& wR,
                                  const std::array<double, 4>& alpha,
                                  Eos eos) const
        {
            auto entropy_variables = [&](const PrimState<dim>& q)
            {
                const double s  = std::log(q.p / std::pow(q.rho, gamma));
                const double ke = 0.5 * (q.v[0] * q.v[0] + q.v[1] * q.v[1]);
                ConsArray<dim> v;
                v[layout::rho]    = (gamma - s) / (gamma - 1.) - q.rho * ke / q.p;
                v[layout::rhoE]   = -q.rho / q.p;
                v[layout::mom(0)] = q.rho * q.v[0] / q.p;
                v[layout::mom(1)] = q.rho * q.v[1] / q.p;
                return v;
            };

            const ConsArray<dim> vL = entropy_variables(L);
            const ConsArray<dim> vR = entropy_variables(R);
            const ConsArray<dim> dv = vR - vL;
            const ConsArray<dim> dw = wR - wL;

            ConsArray<dim> h = 0.5 * (compute_flux<d>(L, eos) + compute_flux<d>(R, eos));
            for (std::size_t k = 0; k < 4; ++k)
            {
                h -= 0.5 * std::abs(lambda[k]) * alpha[k] * eigenvector(k);
            }

            double production = -dw[layout::mom(d)]; // dv.h - d(psi)
            double rounding   = std::abs(wL[layout::mom(d)]) + std::abs(wR[layout::mom(d)]);
            double dv_dw      = 0.;
            for (std::size_t i = 0; i < layout::size; ++i)
            {
                production += dv[i] * h[i];
                rounding += (std::abs(vL[i]) + std::abs(vR[i])) * std::abs(h[i]);
                dv_dw += dv[i] * dw[i];
            }

            // The production is of third order in the jump, and is what is left of terms that
            // are of order zero: v.h on each side. Below a relative jump of about 1e-4 it is
            // their rounding error, and the quotient is anything up to 1e16 at the foot of a
            // wave, where the jump is down to 1e-100. No correction there: it would be of the
            // order of 1e-12 anyway.
            const bool significant = dv_dw > 0. && production > 1e-13 * rounding;
            return significant ? 2. * production / dv_dw : 0.;
        }

        /**
         * |lambda_k| with the entropy corrections: Harten's parabola below harten_fix (|u_n| + c)
         * if asked for, then the speed a of Khalfallah and Lerat on top.
         *
         * Bounded by |u_n| + c, the dissipation of the Rusanov flux, which is entropy stable: the
         * quotient that gives a is only as good as its denominator, which is a rounding error
         * between two nearly equal states.
         */
        double speed(std::size_t k, double a, double harten_fix) const
        {
            double modulus     = std::abs(lambda[k]);
            const double delta = harten_fix * max_speed();
            if (modulus < delta)
            {
                modulus = 0.5 * (modulus * modulus + delta * delta) / delta;
            }
            return std::min(modulus + a, max_speed());
        }
    };

    // The wave strengths the accuracy function reads, numbered along the wave: `up[m - 1]` is m
    // interfaces upwind of this one (m = 1, 2, 3), `here` is this one, `down[m - 1]` is m interfaces
    // downwind (m = 1, 2).
    struct Strengths
    {
        std::array<double, 3> up{};
        double here = 0.;
        std::array<double, 2> down{};
    };

    /**
     * The accuracy function Phi^o of eq. 7, 10 and 11 times the strength of this interface, for
     * |nu| = nu and o = 1 (zero: Roe alone) to 7.
     *
     * The ratios r of the article are ratios of consecutive strengths, so each bracket of eq. 10-11
     * times the strength is a finite difference of the strengths: no division is left.
     */
    inline double accuracy_times_strength(int order, double nu, const Strengths& a)
    {
        if (order <= 1)
        {
            return 0.;
        }
        double phi = a.here; // Lax-Wendroff
        if (order == 2)
        {
            return phi;
        }
        const double c3 = (1. + nu) / 3.;
        phi -= c3 * (a.here - a.up[0]); // eq. 7
        if (order == 3)
        {
            return phi;
        }
        const double c4 = c3 * (nu - 2.) / 4.;
        phi += c4 * (a.here - 2. * a.up[0] + a.up[1]); // eq. 10
        if (order == 4)
        {
            return phi;
        }
        const double c5 = c4 * (nu - 3.) / 5.;
        phi -= c5 * (a.down[0] - 3. * a.here + 3. * a.up[0] - a.up[1]); // eq. 11
        if (order == 5)
        {
            return phi;
        }
        const double c6 = c5 * (nu + 2.) / 6.;
        phi += c6 * (a.down[1] - 4. * a.down[0] + 6. * a.here - 4. * a.up[0] + a.up[1]);
        if (order == 6)
        {
            return phi;
        }
        const double c7 = c6 * (nu + 3.) / 7.;
        phi -= c7 * (a.down[1] - 5. * a.down[0] + 10. * a.here - 10. * a.up[0] + 5. * a.up[1] - a.up[2]);
        return phi;
    }

    /**
     * The correction dF = 1/2 Phi (1 - |nu|) delta|f| of one wave, bounded by the TVD constraints
     * (eq. 66) or by the monotonicity-preserving ones (eq. 37-39 written on the flux differences,
     * eq. 67-70).
     *
     *  unlimited: the correction of the accuracy function Phi^o
     *  nu, nu_up: |nu| at this interface (0 < nu < 1) and at the upwind one
     *  g:         delta|f| at this interface
     *  g_ul:      |lambda| of this interface times the strength of the upwind one
     *  d_md:      the curvature d^M4 at this interface, d_lc the one at the upwind interface
     *
     * The correction has to lie in the intersection of two intervals, each the hull of the values
     * it is given: [0, delta|f|, delta f^md] and [0, delta f^ul, delta f^lc]. For the TVD
     * constraints these are [0, delta|f|] and [0, delta f^ul].
     *
     * delta f^ul and delta f^lc carry a factor 1 / |nu| (eq. 68, 70), which is unbounded for a
     * nearly steady wave. Every value is therefore compared after a multiplication by |nu| > 0,
     * which changes no order, and the only division left is the one that brings a bound back when
     * it is the answer. That bound is then smaller than |nu| times the unlimited correction, so
     * the quotient is bounded by the unlimited correction.
     */
    inline double
    limited_correction(OsmpLimiter limiter, double unlimited, double nu, double nu_up, double g, double g_ul, double d_md, double d_lc)
    {
        const double x    = nu * unlimited;
        const double b_1  = nu * g;              // the TVD bound F in [f_j, f_j+1]
        const double b_ul = (1. - nu_up) * g_ul; // the TVD bound F in [f_j, f^ul] (eq. 68)

        double low  = std::max(std::min(0., b_1), std::min(0., b_ul));
        double high = std::min(std::max(0., b_1), std::max(0., b_ul));
        if (limiter == OsmpLimiter::mp)
        {
            const double b_md = 0.5 * nu * (g - d_md);              // eq. 69
            const double b_lc = 0.5 * (b_ul + (1. - nu_up) * d_lc); // eq. 70
            low               = std::max(std::min({0., b_1, b_md}), std::min({0., b_ul, b_lc}));
            high              = std::min(std::max({0., b_1, b_md}), std::max({0., b_ul, b_lc}));
        }

        if (x < low)
        {
            return low / nu;
        }
        if (x > high)
        {
            return high / nu;
        }
        return unlimited;
    }
}

// stencil_size: 6 up to order 6, 8 for order 7, which reads a third interface upwind.
template <class Field, std::size_t stencil_size = 6, class Eos>
auto make_osmp_scheme(Eos eos, const OsmpOptions& options)
{
    static constexpr std::size_t dim = Field::dim;
    static_assert(stencil_size == 6 || stencil_size == 8);
    static_assert(dim == 2, "OSMP is written for the 2D Euler equations");
    // This interface separates the cells c and c + 1; there are stencil_size - 1 interfaces.
    static constexpr std::size_t c  = stencil_size / 2 - 1;
    static constexpr std::size_t ni = stencil_size - 1;

    if (options.order < 1 || options.order > 7)
    {
        throw std::invalid_argument("OSMP: the order is between 1 and 7");
    }
    if (options.order == 7 && stencil_size != 8)
    {
        throw std::invalid_argument("OSMP7 needs a stencil of 8 cells");
    }
    assert(options.dt && "the one-step scheme needs the time step of the sweep");
    assert(options.direction >= 0 && "OSMP is directional: run it in a Strang splitting");

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

                // Each interface in its own Roe basis: the wave strengths, |lambda| with the entropy corrections
                // and the sign of lambda.
                const osmp::RoeBasis<d> basis(prim[c], prim[c + 1], eos);
                std::array<std::array<double, 4>, ni> strength;
                std::array<std::array<double, 4>, ni> speed;
                std::array<std::array<double, 4>, ni> sign;
                for (std::size_t i = 0; i < ni; ++i)
                {
                    const osmp::RoeBasis<d> own = i == c ? basis : osmp::RoeBasis<d>(prim[i], prim[i + 1], eos);
                    strength[i]                 = own.strengths(field[i + 1] - field[i]);
                    const double a              = own.entropy_correction(prim[i], prim[i + 1], field[i], field[i + 1], strength[i], eos);
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        speed[i][k] = own.speed(k, a, options.harten_fix);
                        sign[i][k]  = own.lambda[k] >= 0. ? 1. : -1.;
                    }
                }

                // First-order Roe flux (eq. 65): the centred flux here, the upwind part per wave below
                ConsArray<dim> F = 0.5 * (compute_flux<d>(prim[c], eos) + compute_flux<d>(prim[c + 1], eos));

                for (std::size_t k = 0; k < 4; ++k)
                {
                    const ConsArray<dim> rk = basis.eigenvector(k);
                    const double g          = speed[c][k] * strength[c][k]; // delta|f| (eq. 65)
                    F -= 0.5 * g * rk;

                    // No correction for a steady wave, which has none, and none beyond the CFL
                    // condition, where the scheme is not defined.
                    const double nu = speed[c][k] * dt_dx;
                    if (options.order <= 1 || !(nu > 0. && nu < 1.))
                    {
                        continue;
                    }

                    // The wave reads its stencil along its own direction: s interfaces further is upwind.
                    const int s   = basis.lambda[k] >= 0. ? -1 : 1;
                    const auto at = [&](int m)
                    {
                        return static_cast<std::size_t>(static_cast<int>(c) + s * m);
                    };

                    osmp::Strengths a;
                    a.here    = strength[c][k];
                    a.up[0]   = strength[at(1)][k];
                    a.up[1]   = strength[at(2)][k];
                    a.down[0] = strength[at(-1)][k];
                    a.down[1] = strength[at(-2)][k];
                    if constexpr (stencil_size == 8)
                    {
                        a.up[2] = strength[at(3)][k];
                    }

                    // eq. 64 and 67: 1/2 Phi (1 - |nu|) delta|f|
                    double dF = 0.5 * (1. - nu) * speed[c][k] * osmp::accuracy_times_strength(options.order, nu, a);

                    if (options.limiter != OsmpLimiter::none)
                    {
                        // The curvature of the cell m, between the interfaces m - 1 and m, written
                        // with the SIGNED eigenvalues (eq. 71): for a left-going wave delta|f|
                        // changes sign under the mirror x -> -x, and so does this curvature.
                        const auto curvature = [&](std::size_t m)
                        {
                            return sign[m][k] * speed[m][k] * strength[m][k] - sign[m - 1][k] * speed[m - 1][k] * strength[m - 1][k];
                        };
                        // d^M4 at the interface i, from its two cells (eq. 40).
                        const auto curvature_m4 = [&](std::size_t i)
                        {
                            return osmp::curvature_m4(curvature(i), curvature(i + 1));
                        };

                        const std::size_t up = at(1);
                        const double nu_up   = std::min(speed[up][k] * dt_dx, 1.);

                        dF = osmp::limited_correction(options.limiter,
                                                      dF,
                                                      nu,
                                                      nu_up,
                                                      g,
                                                      speed[c][k] * strength[up][k],
                                                      curvature_m4(c),
                                                      curvature_m4(up));
                    }

                    F += dF * rk;
                }

                flux = F;
            };
        });

    auto scheme = make_flux_based_scheme(osmp_flux);
    scheme.set_name("osmp" + std::to_string(options.order));

    return scheme;
}
