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

It can also drive the runs itself: give a starting level and a number of points
and it launches euler_2d once per resolution, collects the errors, fits the
order by least squares and draws the convergence curve.

Examples
--------
# convergence study: levels 4, 5, 6, 7 run automatically, then order + plot
python error_analysis.py vortex --Tf 1 --start-level 4 --npoints 4

# single file, density error of the vortex at t = Tf
python error_analysis.py vortex --Tf 10 results/isentropic_vortex_hllc

# convergence study on files already computed by hand
python error_analysis.py vortex --Tf 10 \
    results/vortex_L6 results/vortex_L7 results/vortex_L8

# free-stream preservation on an adapted mesh. --refine-boundary keeps the
# boundary at max-level while the interior coarsens, which is what creates the
# level interfaces the test is meant to exercise: without it the multiresolution
# coarsens the uniform state down to min-level and the check is vacuous (watch
# the nlev column, it must be > 1).
python error_analysis.py free_stream --Tf 0.1 --start-level 4 --npoints 3 \
    --adapt 2 --extra --refine-boundary

# free-stream preservation check on an existing file
python error_analysis.py free_stream results/free_stream_hllc
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

GAMMA = 1.4

# test case name passed to euler_2d --test-case, per exact solution
TEST_CASE_NAME = {"vortex": "isentropic_vortex", "free_stream": "free_stream"}


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

# domain of each test case, must match box_fn in euler/init/*.hpp
BOX = {"vortex": [[-5.0, -5.0], [5.0, 5.0]], "free_stream": [[0.0, 0.0], [1.0, 1.0]]}


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
NORMS = ("L1", "L2", "Linf")
SERIES_COLORS = ("#2a78d6", "#008300", "#e87ba4")
SERIES_MARKERS = ("o", "s", "^")


def errors(num, ref, area):
    e = np.abs(num - ref)
    total = area.sum()
    l1 = np.sum(e * area) / total
    l2 = np.sqrt(np.sum(e * e * area) / total)
    linf = e.max()
    return l1, l2, linf


def analyse_reconstructed(case, filename, t, var, pred_s=1):
    """Same norms, but on the solution reconstructed at the finest level.

    The multiresolution prediction operator is used to fill the cells that the
    adapted mesh does not carry, so every run is compared on the uniform grid of
    its own max-level. This is a cross-check of `analyse`: the area-weighted
    norms on the leaves are already the norms of the piecewise-constant
    numerical solution, so both routines agree (verified to 5 digits on the
    vortex), and reconstruction is not needed to measure an error.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reconstruct import cell_centers, make_exact_bc, read_frame_euler_2d, recons_2d

    box = BOX[case]
    x, y, sol, level = read_frame_euler_2d(filename, box)
    bc = make_exact_bc(lambda X, Y: EXACT[case](X, Y, t))
    num = recons_2d(box, x, y, pred_s, sol, level, bc, var_name=var)

    max_level = int(level.max())
    X, Y = cell_centers(box, max_level, 0)
    ref = EXACT[case](X, Y, t)[var]

    h = np.sqrt(np.prod(np.diff(np.array(box), axis=0))) * 2.0**-max_level
    area = np.full(num.shape, h * h)
    return (h, num.size, 1) + errors(num, ref, area)


def analyse(case, filename, t, var):
    """Return (h, ncells, nlev, L1, L2, Linf) for one output file.

    h is the size of the finest cell of the mesh, i.e. the resolution the
    convergence rate is measured against. nlev is the number of distinct cell
    sizes: it is 1 on a uniform mesh, and tells whether a run that claims to be
    adapted really carries level interfaces.
    """
    x, y, area, sol = read_frame(filename)
    ref = EXACT[case](x, y, t)
    h = np.sqrt(area.min())
    nlev = np.unique(np.round(np.log2(area / area.min()))).size
    return (h, area.size, nlev) + errors(sol[var], ref[var], area)


# ---------------------------------------------------------------------------
# Running the simulations
# ---------------------------------------------------------------------------
def run_simulation(exe, case, level, args, workdir):
    """Run euler_2d at resolution `level` (max-level), return the output file.

    With --adapt 0 the mesh is uniform, which is what measuring the order of a
    scheme requires. With --adapt n the mesh spans n levels below `level`, which
    is what the free-stream preservation check needs.
    """
    test_case = TEST_CASE_NAME[case]
    os.makedirs(workdir, exist_ok=True)

    cmd = [
        exe,
        "--test-case", test_case,
        "--scheme", args.scheme,
        "--Tf", str(args.Tf),
        "--cfl", str(args.cfl),
        "--min-level", str(max(0, level - args.adapt)),
        "--max-level", str(level),
    ]

    if args.mr_eps is not None:
        # The multiresolution threshold has to follow the resolution: with a
        # fixed epsilon the error saturates at the epsilon level and the mesh
        # stops refining, so the measured order collapses. Scaling it as
        # h^eps_order keeps the adaptation error below the discretization one.
        eps = args.mr_eps * 2.0 ** (-args.eps_order * (level - args.start_level))
        cmd += ["--mr-eps", f"{eps:.6e}"]

    cmd += args.extra

    print(f"# level {level}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"euler_2d failed at level {level}")

    # save.hpp writes into <cwd>/results with no suffix when nfiles == 1
    return os.path.join(workdir, "results", f"{test_case}_{args.scheme}")


# ---------------------------------------------------------------------------
# Convergence rate
# ---------------------------------------------------------------------------
def is_exact(err):
    """True when the errors are at (or below) round-off: no order to measure."""
    return np.max(err) <= 1e-14


def fit_order(h, err):
    """Least-squares slope of log(err) vs log(h), i.e. the observed order."""
    return np.polyfit(np.log(h), np.log(err), 1)[0]


def print_table(case, var, t, h, ncells, nlev, err):
    print(f"\n# case={case}  var={var}  t={t}")
    header = f"{'h':>12} {'N':>10} {'nlev':>5}"
    for name in NORMS:
        header += f" {name:>12} {'order':>7}"
    print(header)

    for k in range(len(h)):
        line = f"{h[k]:12.4e} {ncells[k]:10d} {nlev[k]:5d}"
        for name in NORMS:
            order = ""
            if k > 0 and err[name][k] > 0.0 and err[name][k - 1] > 0.0 and h[k] != h[k - 1]:
                order = f"{np.log(err[name][k - 1] / err[name][k]) / np.log(h[k - 1] / h[k]):7.2f}"
            line += f" {err[name][k]:12.4e} {order:>7}"
        print(line)

    print()
    stalled = [k for k in range(1, len(h)) if h[k] == h[k - 1]]
    if stalled:
        print(
            "# WARNING: rows "
            + ", ".join(f"{k}/{k + 1}" for k in stalled)
            + " share the same finest cell: the multiresolution never reached the"
            " requested max-level (epsilon too large), the study is resolution-"
            "limited by --mr-eps, not by the mesh"
        )

    for name in NORMS:
        if is_exact(err[name]):
            print(f"# {name}: exact to round-off, no order to measure")
        else:
            print(f"# least-squares order ({name}) = {fit_order(h, err[name]):.3f}")


def plot_convergence(case, var, h, err, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_facecolor("#fcfcfb")

    for (name, color, marker) in zip(NORMS, SERIES_COLORS, SERIES_MARKERS):
        if is_exact(err[name]):
            continue
        p = fit_order(h, err[name])
        ax.loglog(
            h, err[name],
            color=color, marker=marker, markersize=7, linewidth=2,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label=f"{name}  (order {p:.2f})",
        )

    # reference slope anchored on the coarsest L1 point
    ref_order = 0 if is_exact(err["L1"]) else round(fit_order(h, err["L1"]))
    if ref_order >= 1:
        ref = err["L1"][0] * (np.asarray(h) / h[0]) ** ref_order
        ax.loglog(h, ref, color="#8a8a80", linewidth=1.2, linestyle="--", zorder=0)
        ax.annotate(
            f"slope {ref_order}",
            xy=(h[-1], ref[-1]), xytext=(8, 2), textcoords="offset points",
            color="#5c5c55", fontsize=9,
        )

    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel(f"error on {var}")
    ax.set_title(f"Convergence - {case}", loc="left", fontsize=12)
    ax.grid(True, which="both", color="#e6e6e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c9c0")
    ax.tick_params(colors="#5c5c55")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"# figure saved in {output}")
    plt.show()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Euler analytic error analysis and convergence study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("case", choices=EXACT.keys(), help="Test case")
    parser.add_argument(
        "files", nargs="*", help="HDF5 file(s) without .h5 (skip to run the cases)"
    )
    parser.add_argument("--Tf", type=float, default=0.0, help="Time of the snapshot")
    parser.add_argument(
        "--var",
        default="rho",
        choices=["rho", "pressure", "ux", "uy"],
        help="Variable used for the error",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Measure on the solution reconstructed at the finest level instead"
        " of on the leaves (cross-check, both agree)",
    )

    run = parser.add_argument_group("convergence study (runs euler_2d)")
    run.add_argument("--start-level", type=int, help="Coarsest level of the study")
    run.add_argument("--npoints", type=int, default=4, help="Number of resolutions")
    run.add_argument(
        "--exe", default="build/euler_2d", help="Path to the euler_2d executable"
    )
    run.add_argument("--scheme", default="hllc", help="Finite volume scheme")
    run.add_argument(
        "--adapt",
        type=int,
        default=0,
        help="Number of levels below max-level (0 = uniform mesh)",
    )
    run.add_argument(
        "--mr-eps",
        type=float,
        help="Multiresolution epsilon at the coarsest resolution of the study",
    )
    run.add_argument(
        "--eps-order",
        type=float,
        default=1.0,
        help="--mr-eps is scaled as h^eps_order along the study",
    )
    run.add_argument("--cfl", type=float, default=0.4, help="CFL number")
    run.add_argument(
        "--workdir",
        default="convergence",
        help="Directory holding one subdirectory per resolution",
    )
    run.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Remaining arguments are forwarded to euler_2d",
    )

    plot = parser.add_argument_group("plot")
    plot.add_argument("--no-plot", action="store_true", help="Do not draw the curve")
    plot.add_argument("--output", help="Save the figure to this file")

    args = parser.parse_args()

    if args.start_level is not None:
        exe = os.path.abspath(args.exe)
        if not os.path.isfile(exe):
            parser.error(f"executable not found: {exe} (build it, or use --exe)")
        levels = range(args.start_level, args.start_level + args.npoints)
        files = [
            run_simulation(
                exe, args.case, level, args,
                os.path.join(os.path.abspath(args.workdir), f"level_{level}"),
            )
            for level in levels
        ]
    elif args.files:
        files = args.files
    else:
        parser.error("give either some files or --start-level")

    h, ncells, nlev = [], [], []
    err = {name: [] for name in NORMS}
    for f in files:
        measure = analyse_reconstructed if args.reconstruct else analyse
        hk, nk, lk, l1, l2, linf = measure(args.case, f, args.Tf, args.var)
        h.append(hk)
        ncells.append(nk)
        nlev.append(lk)
        for name, value in zip(NORMS, (l1, l2, linf)):
            err[name].append(value)

    print_table(args.case, args.var, args.Tf, h, ncells, nlev, err)

    if not args.no_plot and len(h) > 1 and not all(is_exact(e) for e in err.values()):
        plot_convergence(args.case, args.var, h, err, args.output)


if __name__ == "__main__":
    main()
