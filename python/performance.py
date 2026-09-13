# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""
Build the performance table of a test case: sparsity index, throughput and time
to solution, one row per resolution.

    python performance.py --levels 6 7 8 9

Runs the solver once per max-level, reads back the metrics each run writes
(`--metrics-file`, see euler/metrics.hpp) and prints them in the column layout
of the article this repository reproduces, so that the two tables can be read
side by side.

Reading the table
-----------------
    l_min/l_max  the levels the multiresolution was allowed to span
    resolution   the uniform mesh at l_max, i.e. the equivalent resolution
    Mcu/s        millions of cell updates per second, one update being one cell
                 advanced by one time step, averaged over the whole run
    time         time to solution in seconds: the time loop, output excluded
    AMR          the share of that time spent adapting the mesh
    cells        leaves at the initial and at the final time
    sparsity     those two counts over the uniform mesh, in percent

What the rows are, and what they are not
----------------------------------------
The article varies the number of cells per octree leaf at a FIXED equivalent
resolution; samurai carries one cell per leaf, so that axis does not exist here
and the table sweeps the resolution instead. The row to compare with a row of
the article is the one with the same equivalent resolution, and nothing else is
comparable: the sparsity index of a two-dimensional Riemann problem falls
roughly as 2^-l_max, the discontinuities being curves in a plane, so a table
read across rows says as much about the resolutions chosen as about the two
codes.

The threshold of the multiresolution is the knob that trades cells for
accuracy, the way the block size is in the article, and --mr-eps sweeps it: pass
several values and each one is measured at each level. Sparsity alone is not a
figure of merit, though — a large enough threshold makes any mesh sparse and the
solution wrong — so a row is only worth reading next to the error it carries,
which is what python/error_analysis.py measures on the cases that have an exact
solution.

--uniform adds the row the article puts last: the same run on a uniform mesh at
l_max, which is the reference both for the throughput (no adaptation to pay
for, no level interface to cross) and for the time to solution (the speed-up
adaptation buys). It costs as much as the whole sweep above it, hence opt-in.

Examples
--------
# the case the article uses for its performance table, over four resolutions
python performance.py --levels 6 7 8 9

# with the uniform reference run at the finest level
python performance.py --levels 6 7 8 9 --uniform

# what the multiresolution threshold buys, at one resolution
python performance.py --levels 8 --mr-eps 1e-4 1e-3 1e-2

# the same in three dimensions, where only configuration 3 exists
python performance.py --dim 3 --levels 4 5 6 --min-level 2

# first order, to see what the reconstruction costs and what it saves
python performance.py --levels 6 7 8 --order 1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# The reference case of the article: Lax & Liu configuration 3, to t_f = 0.8.
DEFAULT_CASE = "lax_liu"
DEFAULT_TF = 0.8


def run(exe, workdir, level, min_level, eps, args):
    """Run one simulation and return the metrics it wrote, as a dict."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    metrics_file = os.path.join(workdir, "metrics.json")

    cmd = [
        exe,
        "--test-case", args.test_case,
        "--scheme", args.scheme,
        "--order", str(args.order),
        "--Tf", str(args.Tf),
        "--cfl", str(args.cfl),
        "--min-level", str(min_level),
        "--max-level", str(level),
        "--nfiles", "1",
        "--metrics-file", metrics_file,
    ]

    if args.test_case == "lax_liu":
        cmd += ["--riemann-config", str(args.riemann_config)]
    if eps is not None:
        cmd += ["--mr-eps", str(eps)]
    cmd += args.extra

    print(f"# {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise RuntimeError(f"{os.path.basename(exe)} failed at level {level}")

    with open(metrics_file) as handle:
        metrics = json.load(handle)
    # The levels and the dimension come back from the run itself; the threshold
    # is ours, the solver having no reason to know it was swept.
    metrics["mr_eps"] = eps
    return metrics


def resolution(metrics):
    """The equivalent uniform resolution, as the article writes it."""
    return f"{2 ** metrics['max_level']}^{metrics['dim']}"


def cells(n):
    """Cell counts the way the article prints them: millions past a million."""
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n}"


def print_table(rows, header):
    print()
    print(header)
    columns = (
        f"{'l_min':>6} {'l_max':>6} {'resolution':>12} {'mr-eps':>9} {'Mcu/s':>8}"
        f" {'time (s)':>9} {'AMR':>6} {'cells ti/tf':>21} {'sparsity ti/tf':>17}"
    )
    print(columns)
    print("-" * len(columns))
    for m in rows:
        counts = f"{cells(m['initial_cells'])} / {cells(m['final_cells'])}"
        sparsity = f"{m['initial_sparsity']:.1f}% / {m['final_sparsity']:.1f}%"
        eps = f"{m['mr_eps']:.0e}" if m["mr_eps"] is not None else "default"
        print(
            f"{m['min_level']:>6} {m['max_level']:>6} {resolution(m):>12} {eps:>9}"
            f" {m['mcu_per_second']:>8.1f} {m['run_time']:>9.2f} {m['adapt_fraction']:>5.1f}%"
            f" {counts:>21} {sparsity:>17}"
        )


def speedup(adapted, uniform):
    """What the adaptation bought at equal resolution, printed under the table."""
    print()
    print(
        f"# uniform reference at l_max = {uniform['max_level']}: "
        f"{uniform['run_time'] / adapted['run_time']:.2f}x the time to solution of the adapted run, "
        f"{uniform['mcu_per_second'] / adapted['mcu_per_second']:.2f}x its throughput, "
        f"{uniform['cell_updates'] / adapted['cell_updates']:.2f}x its cell updates"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dim", type=int, default=2, choices=(1, 2, 3), help="which solver to drive")
    parser.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8], help="max-levels to measure")
    parser.add_argument("--min-level", type=int, default=3, help="min-level, the same for every row")
    parser.add_argument("--test-case", default=DEFAULT_CASE)
    parser.add_argument("--riemann-config", type=int, default=3, help="Lax & Liu configuration, for the lax_liu case")
    parser.add_argument("--Tf", type=float, default=DEFAULT_TF)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--scheme", default="hllc")
    parser.add_argument("--order", type=int, default=2, choices=(1, 2), help="the article runs second order")
    parser.add_argument("--mr-eps", type=float, nargs="+", default=[None],
                        help="multiresolution thresholds to measure; every one of them is run at every level")
    parser.add_argument("--uniform", action="store_true", help="add the uniform reference run at the finest level")
    parser.add_argument("--exe", default=None, help="path to the solver (default: build/euler_<dim>d)")
    parser.add_argument("--workdir", default=None, help="where to run (default: a temporary directory)")
    parser.add_argument("--json", default=None, help="also write every row to this file")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed on to the solver; must come last")
    args = parser.parse_args(argv)

    exe = args.exe or os.path.join("build", f"euler_{args.dim}d")
    exe = os.path.abspath(exe)
    if not os.path.exists(exe):
        parser.error(f"{exe} not found; build the project or pass --exe")

    workdir = args.workdir or tempfile.mkdtemp(prefix="performance_")
    rows = [
        run(exe, os.path.join(workdir, f"level{level}_eps{eps}"), level, args.min_level, eps, args)
        for level in args.levels
        for eps in args.mr_eps
    ]

    case = args.test_case + (f" #{args.riemann_config}" if args.test_case == "lax_liu" else "")
    header = f"# {case}, order {args.order}, {args.scheme}, Tf = {args.Tf}, euler_{args.dim}d"
    print_table(rows, header)

    if args.uniform:
        # The threshold plays no part here: with one level there is nothing to
        # coarsen. One reference run per sweep, at the finest resolution of it.
        finest = max(args.levels)
        reference = run(exe, os.path.join(workdir, "uniform"), finest, finest, None, args)
        print_table([reference], "# uniform mesh, same resolution")
        speedup([row for row in rows if row["max_level"] == finest][0], reference)
        rows.append(reference)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\n# rows written to {args.json}")

    return rows


if __name__ == "__main__":
    main()
