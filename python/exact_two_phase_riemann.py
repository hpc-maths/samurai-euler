# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""
Exact solution of a one-dimensional Riemann problem between two stiffened gases.

This is what the water-air shock tube of the article is held against. The
monofluid `exact_riemann.py` cannot serve: the two sides are two different
fluids, each with its own gamma and its own reference pressure, and the contact
between them is a material interface rather than a density jump inside one gas.

A stiffened gas  p = (gamma - 1) rho e - gamma pi  is an ideal gas of the same
gamma in the variable q = p + pi, so the wave relations of Toro chapter 4 carry
over with p replaced by p + pi on each side. What does NOT carry over is a
single shift of the whole problem: the star pressure is common to both sides
while pi is not, so each side keeps its own.

    star_state(left, right, gas_l, gas_r) -> (p_star, u_star)
    solution(x, t, left, right, gas_l, gas_r, x0) -> rho, u, p, alpha

`left` and `right` are (rho, u, p) triples, `gas_*` are Gas(gamma, pi). alpha is
the volume fraction of the left fluid: 1 on its side of the contact, 0 on the
other, which is what a two-phase solver should reproduce as a step that the
diffuse interface smears over a few cells.

Run the module to print the water-air star state:

    python exact_two_phase_riemann.py
"""

from collections import namedtuple

import numpy as np

__all__ = ["Gas", "WATER", "AIR", "star_state", "sample", "solution"]

Gas = namedtuple("Gas", "gamma pi")

# The two fluids of section 6.1.1 of the article.
WATER = Gas(4.4, 6e8)
AIR = Gas(1.4, 0.0)


def sound_speed(rho, p, gas):
    """c = sqrt(gamma (p + pi) / rho)."""
    return np.sqrt(gas.gamma * (p + gas.pi) / rho)


def _f(p, state, gas):
    """Pressure function of one side and its derivative, Toro (4.6)-(4.7) in q."""
    rho_k, _, p_k = state
    gamma, pi = gas.gamma, gas.pi
    q, q_k = p + pi, p_k + pi
    a_k = sound_speed(rho_k, p_k, gas)

    if p > p_k:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * q_k
        root = np.sqrt(A / (B + q))
        return (q - q_k) * root, root * (1.0 - 0.5 * (q - q_k) / (B + q))

    # rarefaction
    ratio = q / q_k
    power = (gamma - 1.0) / (2.0 * gamma)
    return (
        2.0 * a_k / (gamma - 1.0) * (ratio**power - 1.0),
        1.0 / (rho_k * a_k) * ratio ** (-(gamma + 1.0) / (2.0 * gamma)),
    )


def star_state(left, right, gas_l=WATER, gas_r=AIR, tol=1e-12, maxiter=200):
    """Pressure and velocity of the star region between the two states."""
    _, u_l, p_l = left
    _, u_r, p_r = right

    # Newton from the average of the two pressures, bracketed below by the
    # vacuum of the stiffer side: the iterate has to keep p + pi positive on
    # both, and the water branch is steep enough that a bad first guess walks
    # straight out of the physical range.
    floor = -min(gas_l.pi, gas_r.pi) + 1e-6
    p = max(0.5 * (p_l + p_r), floor)

    for _ in range(maxiter):
        f_l, df_l = _f(p, left, gas_l)
        f_r, df_r = _f(p, right, gas_r)
        step = (f_l + f_r + u_r - u_l) / (df_l + df_r)
        p_new = max(p - step, floor)
        if abs(p_new - p) <= tol * max(1.0, abs(p_new)):
            p = p_new
            break
        p = p_new
    else:
        raise RuntimeError("the star pressure did not converge")

    f_l, _ = _f(p, left, gas_l)
    f_r, _ = _f(p, right, gas_r)
    return p, 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)


def _sample_side(s, state, gas, p_star, u_star, mirror):
    """Sample the side of the contact the ray s falls on.

    `mirror` is +1 on the left and -1 on the right: negating x and u turns the
    right state into a left one, so the left-running wave formulas serve both.
    """
    rho_k, u_k, p_k = state
    gamma, pi = gas.gamma, gas.pi
    a_k = sound_speed(rho_k, p_k, gas)

    s, u_k, u_star = mirror * s, mirror * u_k, mirror * u_star
    ratio = (p_star + pi) / (p_k + pi)

    if p_star > p_k:  # shock
        beta = (gamma - 1.0) / (gamma + 1.0)
        speed = u_k - a_k * np.sqrt((gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma))
        if s <= speed:
            return rho_k, mirror * u_k, p_k
        return rho_k * (ratio + beta) / (beta * ratio + 1.0), mirror * u_star, p_star

    # rarefaction
    a_star = a_k * ratio ** ((gamma - 1.0) / (2.0 * gamma))
    head, tail = u_k - a_k, u_star - a_star
    if s <= head:
        return rho_k, mirror * u_k, p_k
    if s >= tail:
        return rho_k * ratio ** (1.0 / gamma), mirror * u_star, p_star

    # inside the fan
    u = 2.0 / (gamma + 1.0) * (a_k + 0.5 * (gamma - 1.0) * u_k + s)
    a = 2.0 / (gamma + 1.0) * (a_k + 0.5 * (gamma - 1.0) * (u_k - s))
    rho = rho_k * (a / a_k) ** (2.0 / (gamma - 1.0))
    q = (p_k + pi) * (a / a_k) ** (2.0 * gamma / (gamma - 1.0))
    return rho, mirror * u, q - pi


def sample(s, left, right, gas_l, gas_r, p_star, u_star):
    """State (rho, u, p, alpha) on the ray x/t = s."""
    if s <= u_star:
        return (*_sample_side(s, left, gas_l, p_star, u_star, +1), 1.0)
    return (*_sample_side(s, right, gas_r, p_star, u_star, -1), 0.0)


def solution(x, t, left, right, gas_l=WATER, gas_r=AIR, x0=0.7):
    """The exact solution at time t on the points x: rho, u, p and alpha."""
    p_star, u_star = star_state(left, right, gas_l, gas_r)
    s = (np.asarray(x, dtype=float) - x0) / t
    out = np.array([sample(si, left, right, gas_l, gas_r, p_star, u_star) for si in s])
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
# The shock tube of section 6.1.1: water at 1e9 against air at 1e5, both at
# rest. The star state is what the figure of the article shows -- a plateau of
# the mixture density a little above 800 and a pressure that has fallen by three
# decades -- and it is what the solver is measured against.
WATER_AIR = ((1e3, 0.0, 1e9), (1.0, 0.0, 1e5))


def _self_test():
    left, right = WATER_AIR
    p_star, u_star = star_state(left, right)

    # Independent of the Newton iteration: the two pressure functions must
    # balance at the root that was found.
    residual = _f(p_star, left, WATER)[0] + _f(p_star, right, AIR)[0] + right[1] - left[1]
    assert abs(residual) < 1e-6 * u_star, f"the star state does not solve its own equation: {residual}"

    # Either side of the contact, which stands at x0 + u* t at time t.
    t = 9e-4
    contact = 0.7 + u_star * t
    rho, u, p, alpha = solution(np.array([contact - 1e-6, contact + 1e-6]), t, left, right)

    print(f"water-air shock tube:  p* = {p_star:.6e} Pa   u* = {u_star:.4f} m/s")
    print(f"  water side of the contact:  rho = {rho[0]:.3f}  p = {p[0]:.6e}  alpha = {alpha[0]}")
    print(f"  air side of the contact:    rho = {rho[1]:.4f}  p = {p[1]:.6e}  alpha = {alpha[1]}")

    for name, value, expected in (("p*", p_star, 4.796906e5), ("u*", u_star, 491.97)):
        assert abs(value - expected) < 1e-4 * abs(expected), f"{name} = {value}, expected {expected}"
    print("self test passed")


if __name__ == "__main__":
    _self_test()
