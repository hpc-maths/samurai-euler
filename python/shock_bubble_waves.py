# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""
The five wave speeds of the air-helium shock-bubble interaction.

Section 6.1.3 of the article this repository reproduces is the one case in it
validated against an experiment rather than against another code, and what is
compared is a table of wavefront velocities. This script measures them the way
the article does: snapshots every ten microseconds, the position of each front
read along the axis of the tube, and a straight line fitted through each.

    python shock_bubble_waves.py --level 8                 # run, then measure
    python shock_bubble_waves.py --from results/sb         # measure what is there

What is tracked, on the centreline of the tube, the flow running right to left:

    shock        the incident shock in the air, before it reaches the bubble
    refracted    the front inside the bubble, which runs ahead of the incident
                 one because helium carries sound at three times the speed
    transmitted  the same front once it has left the bubble and is in air again
    downstream   the left-hand edge of the bubble, the one the flow pushes
    jet          the right-hand edge, which caves in and drives a spike of air
                 through the bubble

The first three are one measurement, not three: the leading pressure front on the
centreline IS the incident shock while it is right of the bubble, the refracted
wave while it is inside it, and the transmitted wave once it is out. Its speed
changes at each crossing, and the three segments are separated at the two edges
of the bubble rather than by eye.

The two interfaces are read from the volume fraction, and the fits use the
article's windows: 100 to 250 microseconds after impact for the downstream edge,
which barely moves before that, and everything after 100 microseconds for the
jet.

Time is counted from the impact, as the article counts it: the fitted shock
trajectory crossing the upstream edge of the bubble.
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np

# Geometry of euler/two_phase/init/shock_bubble.hpp, in metres.
LENGTH = 0.445
HEIGHT = 0.089
BUBBLE_X = 0.225
BUBBLE_R = 0.025
AMBIENT_P = 101325.0

UPSTREAM_EDGE = BUBBLE_X + BUBBLE_R  # the shock reaches this one first
DOWNSTREAM_EDGE = BUBBLE_X - BUBBLE_R

# Table 3 of the article, and the experiment it is held against. Velocities in
# m/s; the experimental error is the 10% Haas and Sturtevant quote.
PUBLISHED = {
    "shock": {"experiment": (410, 41), "article": 423.2},
    "refracted": {"experiment": (900, 90), "article": 953.0},
    "transmitted": {"experiment": (393, 39), "article": 381.2},
    "downstream": {"experiment": (145, 15), "article": 141.5},
    "jet": {"experiment": (230, 23), "article": 222.9},
}


def read_cut(filename, y=HEIGHT / 2.0):
    """The cells the horizontal line y crosses, sorted by x.

    The mesh is adapted, so the cut is not a row of equal cells: what comes back
    is the centre and the fields of every cell the line passes through, which is
    all a front detector needs.
    """
    with h5py.File(filename, "r") as handle:
        mesh = handle["mesh"]
        corners = mesh["points"][:][mesh["connectivity"][:]]
        lower = corners[:, :, :2].min(axis=1)
        upper = corners[:, :, :2].max(axis=1)

        crossed = (lower[:, 1] <= y) & (upper[:, 1] > y)
        x = 0.5 * (lower[crossed, 0] + upper[crossed, 0])
        order = np.argsort(x)

        group = mesh["fields"]
        fields = {name: group[name][:][crossed][order] for name in ("alpha", "rho", "pressure")}
        return x[order], fields


def leading_front(x, pressure, threshold=1.02):
    """Where the leading pressure front stands: the smallest x above ambient.

    Everything ahead of the front is still at rest at one atmosphere, so the
    front is the edge of the region that is not. The threshold is well above the
    numerical foot of the wave and well below its strength, which is 57%.
    """
    disturbed = np.nonzero(pressure > threshold * AMBIENT_P)[0]
    if disturbed.size == 0:
        return None
    return x[disturbed[0]]


def interfaces(x, alpha):
    """The two edges of the bubble on the cut, from the volume fraction of air.

    alpha is 1 in air and 0 in the bubble, so the edges are where it crosses one
    half. Returns (downstream, upstream), the left one and the right one, or None
    once the bubble no longer crosses the line.
    """
    inside = np.nonzero(alpha < 0.5)[0]
    if inside.size == 0:
        return None, None
    return x[inside[0]], x[inside[-1]]


def measure(directory, stem, nfiles):
    """Read every snapshot and return the fronts it holds, in metres and seconds."""
    records = []
    for i in range(nfiles):
        path = os.path.join(directory, f"{stem}_ite_{i}.h5")
        if not os.path.exists(path):
            continue
        x, fields = read_cut(path)
        downstream, upstream = interfaces(x, fields["alpha"])
        records.append(
            {
                "file": i,
                "front": leading_front(x, fields["pressure"]),
                "downstream": downstream,
                "upstream": upstream,
            }
        )
    return records


def fit(times, positions, window=None):
    """Speed of a front, as the slope of a straight line through its positions.

    The sign is dropped: every front here runs towards decreasing x, and the
    table of the article gives speeds and not velocities. The second return value
    is the standard error of the slope, which is what the article prints as its
    error interval.
    """
    times, positions = np.asarray(times), np.asarray(positions)
    if window is not None:
        inside = (times >= window[0]) & (times <= window[1])
        times, positions = times[inside], positions[inside]

    if times.size < 3:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(times, positions, 1)
    residual = positions - (slope * times + intercept)
    variance = np.sum(residual**2) / max(times.size - 2, 1)
    spread = np.sum((times - times.mean()) ** 2)
    return abs(slope), np.sqrt(variance / spread)


def wave_speeds(records, dt_save):
    """The five speeds, each from the segment of the run where its front exists.

    Snapshot i stands at (i + 1) dt_save: the solver writes its first numbered
    file one save interval into the run, the initial state having gone to
    <filename>_init. Getting that wrong shifts the impact time by ten
    microseconds and moves both interface windows with it.
    """
    times = np.array([(r["file"] + 1) * dt_save for r in records])

    front = np.array([r["front"] if r["front"] is not None else np.nan for r in records])
    known = ~np.isnan(front)

    # The leading front is the incident shock while it is right of the bubble,
    # the refracted wave while it is inside, the transmitted wave once out.
    segments = {
        "shock": front > UPSTREAM_EDGE,
        "refracted": (front <= UPSTREAM_EDGE) & (front >= DOWNSTREAM_EDGE),
        "transmitted": front < DOWNSTREAM_EDGE,
    }

    speeds = {}
    for name, inside in segments.items():
        take = known & inside
        speeds[name] = fit(times[take], front[take])

    # Impact: when the fitted shock trajectory reaches the upstream edge. It is
    # the origin the article counts its fitting windows from.
    take = known & segments["shock"]
    slope, intercept = np.polyfit(times[take], front[take], 1)
    impact = (UPSTREAM_EDGE - intercept) / slope

    since_impact = times - impact
    for name, key, window in (
        ("downstream", "downstream", (100e-6, 250e-6)),
        ("jet", "upstream", (100e-6, np.inf)),
    ):
        position = np.array([r[key] if r[key] is not None else np.nan for r in records])
        take = ~np.isnan(position)
        speeds[name] = fit(since_impact[take], position[take], window)

    return speeds, impact


def report(speeds, impact):
    """The five speeds against the article and against the experiment."""
    print(f"\nshock reaches the bubble at t = {impact * 1e6:.1f} us  (the article says about 58)")
    print(f"\n{'':13} {'measured':>18} {'article':>9} {'experiment':>14}   {'within 10%':>10}")
    print("-" * 72)

    ok = True
    for name, published in PUBLISHED.items():
        speed, error = speeds[name]
        value, margin = published["experiment"]
        agrees = abs(speed - value) <= margin
        ok = ok and agrees
        print(
            f"{name:13} {speed:10.1f} +- {error:4.1f} {published['article']:9.1f} "
            f"{value:9.0f}+-{margin:<3.0f}   {'yes' if agrees else 'NO':>10}"
        )
    return ok


def run(exe, workdir, level, min_level, tf, nfiles, extra):
    os.makedirs(workdir, exist_ok=True)
    cmd = [
        exe,
        "--test-case", "shock_bubble",
        "--min-level", str(min_level),
        "--max-level", str(level),
        "--Tf", str(tf),
        "--order", "2",
        "--thinc",
        "--nfiles", str(nfiles),
        "--filename", "shock_bubble",
    ] + extra

    print(f"# {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise RuntimeError("two_phase_2d failed")
    print(proc.stdout[proc.stdout.rfind("performance") :])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=8, help="max-level; 10 is the resolution of the article")
    parser.add_argument("--min-level", type=int, default=5)
    parser.add_argument("--Tf", type=float, default=360e-6, help="final time; the fits need 250 us after impact")
    parser.add_argument("--nfiles", type=int, default=36, help="snapshots; 36 over 360 us is one every 10 us")
    parser.add_argument("--exe", default=os.path.join("build", "two_phase_2d"))
    parser.add_argument("--workdir", default="shock_bubble")
    parser.add_argument("--from", dest="existing", default=None, help="measure an existing results directory instead")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed on to the solver; must come last")
    args = parser.parse_args(argv)

    if args.existing:
        directory = args.existing
    else:
        run(os.path.abspath(args.exe), args.workdir, args.level, args.min_level, args.Tf, args.nfiles, args.extra)
        directory = os.path.join(args.workdir, "results")

    records = measure(directory, "shock_bubble", args.nfiles)
    if len(records) < 10:
        raise RuntimeError(f"only {len(records)} snapshots in {directory}")

    speeds, impact = wave_speeds(records, args.Tf / args.nfiles)
    report(speeds, impact)
    return speeds


if __name__ == "__main__":
    main()
