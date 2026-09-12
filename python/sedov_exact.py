# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""
Exact self-similar solution of the Sedov blast wave, in one, two and three
dimensions.

This is what the Sedov case is checked against. A point-like release of energy E
in an ambient gas at rest drives a shock whose radius obeys

    r_s(t) = ( E t^2 / (alpha rho_0) )^(1/(nu+2)),

nu = 1, 2, 3 being planar, cylindrical and spherical geometry. The constant
alpha depends on nu and gamma alone, and is computed here rather than quoted:
the similarity equations are integrated inward from the shock, where the strong
shock jump conditions give the starting values, and alpha follows from the
energy integral over the profile.

    L.I. Sedov, "Similarity and Dimensional Methods in Mechanics", Academic
    Press, New York, 1959, chapter IV.
    J.R. Kamm, F.X. Timmes, "On Efficient Generation of Numerically Robust
    Sedov Solutions", report LA-UR-07-2849, Los Alamos National Laboratory,
    2007.

The planar constant is computed with the energy counted on both sides of the
plane, which is how euler/init/sedov_blast.hpp deposits it. Kamm's planar
standard case counts one side, so his planar energy corresponds to half of
this one; the cylindrical and spherical cases have no such convention.

Run the module to print the constants and check the spherical one against the
value published for gamma = 1.4:

    python sedov_exact.py
"""

import numpy as np

__all__ = ["alpha", "shock_radius", "profile"]

# The similarity integration is deterministic and takes about half a second, so
# the constants are memoised rather than tabulated: no number in this file is
# copied from a paper.
_ALPHA_CACHE = {}


def _derivatives(lam, y, nu, gamma):
    """Right-hand side of the similarity equations, in the variables

        rho = rho_0 G(lam),  u = rdot_s V(lam),  p = rho_0 rdot_s^2 P(lam),

    lam = r / r_s. Continuity, momentum and the entropy condition, with the
    time dependence eliminated through r_s ~ t^(2/(nu+2)), leave three coupled
    ordinary equations in lam.
    """
    G, V, P = y
    D = V - lam
    Vp = (nu - gamma * (nu - 1) * V / lam - nu * V * D * G / (2.0 * P)) / (gamma - D * D * G / P)
    Gp = -(G * Vp + (nu - 1) * G * V / lam) / D
    Pp = G * (nu * V / 2.0 - D * Vp)
    return np.array([Gp, Vp, Pp])


def _integrate(nu, gamma, steps, lam_min):
    """Integrate from the shock at lam = 1 down to lam_min, by RK4.

    Returns lam increasing, with G, V, P alongside.
    """
    y = np.array(
        [
            (gamma + 1.0) / (gamma - 1.0),  # strong shock jump conditions
            2.0 / (gamma + 1.0),
            2.0 / (gamma + 1.0),
        ]
    )

    h = -(1.0 - lam_min) / steps
    lam = 1.0
    lams, ys = [lam], [y.copy()]

    for _ in range(steps):
        k1 = _derivatives(lam, y, nu, gamma)
        k2 = _derivatives(lam + 0.5 * h, y + 0.5 * h * k1, nu, gamma)
        k3 = _derivatives(lam + 0.5 * h, y + 0.5 * h * k2, nu, gamma)
        k4 = _derivatives(lam + h, y + h * k3, nu, gamma)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        lam += h
        if not np.all(np.isfinite(y)) or y[2] <= 0.0:
            break
        lams.append(lam)
        ys.append(y.copy())

    return np.array(lams)[::-1], np.array(ys)[::-1]


def alpha(dim, gamma=1.4, steps=20000, lam_min=1e-6):
    """Dimensionless energy of the similarity solution, for this geometry.

    The energy of the solution, in the similarity variables, is

        E = sigma_nu rho_0 r_s^nu rdot_s^2 Int, Int = int_0^1 (G V^2 / 2 +
            P / (gamma - 1)) lam^(nu-1) dlam,

    and alpha = E t^2 / (rho_0 r_s^(nu+2)) follows from rdot_s = 2 r_s /
    ((nu+2) t). sigma_nu is 2, 2 pi, 4 pi: the planar solution is counted on
    both sides of the plane.
    """
    key = (dim, gamma, steps, lam_min)
    if key not in _ALPHA_CACHE:
        lam, y = _integrate(dim, gamma, steps, lam_min)
        G, V, P = y[:, 0], y[:, 1], y[:, 2]
        integral = np.trapezoid((0.5 * G * V * V + P / (gamma - 1.0)) * lam ** (dim - 1), lam)
        sigma = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}[dim]
        _ALPHA_CACHE[key] = 4.0 * sigma * integral / (dim + 2.0) ** 2
    return _ALPHA_CACHE[key]


def shock_radius(energy, t, dim, gamma=1.4, rho_ambient=1.0):
    """Where the shock stands at time t."""
    return (energy * t * t / (alpha(dim, gamma) * rho_ambient)) ** (1.0 / (dim + 2.0))


def profile(energy, t, dim, gamma=1.4, rho_ambient=1.0, p_ambient=0.0):
    """The exact solution at time t, as (r, rho, u, p) sampled on the profile.

    The points are those the integration produced, which cluster where the
    solution varies; the last one is the shock. Outside it the gas is still
    ambient, which the caller usually knows already.
    """
    lam, y = _integrate(dim, gamma, 20000, 1e-6)
    r_s = shock_radius(energy, t, dim, gamma, rho_ambient)
    speed = 2.0 * r_s / ((dim + 2.0) * t)
    return (
        lam * r_s,
        rho_ambient * y[:, 0],
        speed * y[:, 1],
        rho_ambient * speed * speed * y[:, 2] + p_ambient,
    )


if __name__ == "__main__":
    import sys

    # Published for gamma = 1.4: the spherical blast reaches r = 1 at t = 1 for
    # E = 0.851072, so alpha is that same number. It is the one constant of the
    # three that several codes quote, hence the one worth checking against.
    SPHERICAL = 0.851072

    print(f"gamma = 1.4\n{'geometry':>12} {'alpha':>10}   E for r_s = 1 at t = 1")
    for dim, name in ((1, "planar"), (2, "cylindrical"), (3, "spherical")):
        a = alpha(dim)
        print(f"{name:>12} {a:>10.6f}   {a:.6f}")

    deviation = abs(alpha(3) - SPHERICAL)
    print(f"\nspherical against the published {SPHERICAL}: {deviation:.1e}")
    sys.exit(0 if deviation < 5e-7 else 1)
