# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""
Error analysis for the analytic Euler test cases (isentropic vortex, free-stream).

Reads the HDF5 output(s) of euler_2d and compares the numerical solution to the
exact solution on the leaf cells, weighting each cell by its area (so the norms
are consistent across an adapted, multi-level mesh):

    L1   = sum_i |e_i| * area_i
    L2   = sqrt( sum_i e_i^2 * area_i )
    Linf = max_i |e_i|

Given several runs at increasing resolution, it also prints the observed order
of convergence  p = log(E_k / E_{k+1}) / log(2)  between consecutive levels.

Examples
--------
# single file, density error of the vortex at t = Tf
python error_analysis.py vortex --Tf 10 results/isentropic_vortex_hllc

# convergence study (one file per resolution, increasing max-level)
python error_analysis.py vortex --Tf 10 \
    results/vortex_L6 results/vortex_L7 results/vortex_L8

# free-stream preservation check
python error_analysis.py free_stream results/free_stream_hllc
"""

import argparse

import h5py
import numpy as np

GAMMA = 1.4


# ---------------------------------------------------------------------------
# HDF5 reading (same layout as reconstruct.py / save.hpp)
# ---------------------------------------------------------------------------
def read_frame(filename):
    """Return cell centers (x, y), cell areas and the primitive fields."""
    mesh = h5py.File(filename + ".h5", "r")["mesh"]
    points = mesh["points"]
    connectivity = mesh["connectivity"]

    xyz = points[:][connectivity[:]][:, :, :]
    centers = 0.25 * (xyz[:, 0, :] + xyz[:, 1, :] + xyz[:, 2, :] + xyz[:, 3, :])

    # cell area from the bounding box of its 4 corners (axis-aligned quads)
    dx = xyz[:, :, 0].max(axis=1) - xyz[:, :, 0].min(axis=1)
    dy = xyz[:, :, 1].max(axis=1) - xyz[:, :, 1].min(axis=1)
    area = dx * dy

    sol = {
        "rho": mesh["fields"]["rho"][:],
        "pressure": mesh["fields"]["pressure"][:],
        "ux": mesh["fields"]["velocity_0"][:],
        "uy": mesh["fields"]["velocity_1"][:],
    }
    return centers[:, 0], centers[:, 1], area, sol


# ---------------------------------------------------------------------------
# Exact solutions (must match euler/init/*.hpp)
# ---------------------------------------------------------------------------
def exact_vortex(x, y, t, L=5.0, R=1.0, sigma=1.0, x0=0.0, y0=0.0):
    """Isentropic Euler vortex, "Shu" row of Table 1 of Spiegel, Huynh & DeBonis
    (AIAA 2015-2444). Same notation / non-dimensionalization as
    euler/init/isentropic_vortex.hpp. Returns a dict of primitive variables.

    Domain [-L, L]^2 ; periodic, nearest-image evaluation.
    """
    alpha = np.pi / 4.0
    M_inf = np.sqrt(2.0 / GAMMA)
    rho_inf = 1.0
    beta = M_inf * 5.0 * np.sqrt(2.0) / (4.0 * np.pi) * np.exp(0.5)  # eq. (20)

    v_x_inf = M_inf * np.cos(alpha)
    v_y_inf = M_inf * np.sin(alpha)

    domain_length = 2.0 * L
    x_c = x0 + v_x_inf * t  # eq. (24)
    y_c = y0 + v_y_inf * t

    x_bar = x - x_c
    y_bar = y - y_c
    x_bar -= domain_length * np.round(x_bar / domain_length)
    y_bar -= domain_length * np.round(y_bar / domain_length)

    f = -1.0 / (2.0 * sigma**2) * ((x_bar / R) ** 2 + (y_bar / R) ** 2)  # eq. (21)
    Omega = beta * np.exp(f)  # eq. (20)

    delta_v_x = -(y_bar / R) * Omega  # eq. (22)
    delta_v_y = +(x_bar / R) * Omega
    delta_T = -(GAMMA - 1.0) / 2.0 * Omega**2

    rho = rho_inf * (1.0 + delta_T) ** (1.0 / (GAMMA - 1.0))  # eq. (23)
    p = 1.0 / GAMMA * (1.0 + delta_T) ** (GAMMA / (GAMMA - 1.0))

    return {
        "rho": rho,
        "pressure": p,
        "ux": v_x_inf + delta_v_x,
        "uy": v_y_inf + delta_v_y,
    }


def exact_free_stream(x, y, t):
    """Uniform state (matches euler/init/free_stream.hpp)."""
    o = np.ones_like(x)
    return {"rho": o, "pressure": o, "ux": o, "uy": o}


EXACT = {"vortex": exact_vortex, "free_stream": exact_free_stream}


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
def errors(num, ref, area):
    e = np.abs(num - ref)
    total = area.sum()
    l1 = np.sum(e * area) / total
    l2 = np.sqrt(np.sum(e * e * area) / total)
    linf = e.max()
    return l1, l2, linf


def analyse(case, filename, t, var):
    x, y, area, sol = read_frame(filename)
    ref = EXACT[case](x, y, t)
    return errors(sol[var], ref[var], area)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Euler analytic error analysis")
    parser.add_argument("case", choices=EXACT.keys(), help="Test case")
    parser.add_argument("files", nargs="+", help="HDF5 file(s) without .h5")
    parser.add_argument("--Tf", type=float, default=0.0, help="Time of the snapshot")
    parser.add_argument(
        "--var",
        default="rho",
        choices=["rho", "pressure", "ux", "uy"],
        help="Variable used for the error",
    )
    args = parser.parse_args()

    print(f"# case={args.case}  var={args.var}  t={args.Tf}")
    print(f"{'file':<40} {'N':>9} {'L1':>12} {'L2':>12} {'Linf':>12} {'order(L1)':>10}")

    prev = None
    for f in args.files:
        l1, l2, linf = analyse(args.case, f, args.Tf, args.var)
        ncells = read_frame(f)[2].size
        order = ""
        if prev is not None:
            order = f"{np.log(prev / l1) / np.log(2.0):10.2f}"
        print(f"{f:<40} {ncells:>9} {l1:12.4e} {l2:12.4e} {linf:12.4e} {order:>10}")
        prev = l1


if __name__ == "__main__":
    main()
