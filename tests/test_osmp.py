# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""The one-step monotonicity-preserving schemes, `--order 3 --osmp-order o`.

OSMP is `euler_2d` only and takes neither a Riemann solver nor a slope limiter,
so it has its own file instead of one more entry in the lists of the others.

What is held here is what broke, or would break silently:

- the scheme survives the double Mach reflection at every order. Its limiter
  used to be written on ratios of wave strengths, which are inf or NaN at the
  foot of a wave, and OSMP7 did not reach the final time;
- a left-going wave is the mirror image of a right-going one, which is what the
  sign of the curvature (eq. 71 of Daru and Tenaud) is there for;
- each order converges at its own rate on a smooth solution, limiter on;
- the Mach stem of the double Mach reflection is straight.
"""

import numpy as np
import pytest

from util import read, run_case

OSMP_ORDERS = [3, 5, 7]

# What the pulse measures at levels 7 to 8, limiter on: 2.95, 4.95, 6.92.
MIN_ORDER = {3: 2.85, 5: 4.85, 7: 6.8}


def osmp(workdir, case, level, osmp_order, **options):
    """One uniform run of OSMP, and the file it wrote."""
    out, stem = run_case("euler_2d", workdir, case, min_level=level, max_level=level,
                         order=3, osmp_order=osmp_order, **options)
    return out / stem


def on_grid(centers, values):
    """A field of a uniform run as an array indexed [j, i]."""
    dx = np.diff(np.unique(centers[:, 0])).min()
    i = np.rint((centers[:, 0] - centers[:, 0].min()) / dx).astype(int)
    j = np.rint((centers[:, 1] - centers[:, 1].min()) / dx).astype(int)
    grid = np.full((j.max() + 1, i.max() + 1), np.nan)
    grid[j, i] = values
    return grid, dx


def mach_stem_spread(centers, rho):
    """How far the Mach stem of the double Mach reflection is from a straight line.

    Below the triple point the stem is a nearly normal shock: the rightmost
    shocked cell sits at almost the same abscissa on every row. A kinked stem
    runs ahead of itself along the wall, by 0.15 and more, against 0.02 to 0.05
    for a straight one, the stem being slightly curved.
    """
    grid, dx = on_grid(centers, rho)
    rows = grid[: int(0.3 * grid.shape[0])]
    front = np.array([np.nonzero(row > 2.0)[0].max() for row in rows]) * dx
    return front.max() - front.min()


@pytest.mark.parametrize("osmp_order", OSMP_ORDERS)
def test_free_stream_stays_uniform(osmp_order, tmp_path):
    """A uniform flow is an exact solution: it must not move at all."""
    _, _, fields = read(osmp(tmp_path, "free_stream", 6, osmp_order, Tf=0.05))

    assert np.abs(fields["rho"] - 1.0).max() < 1e-14
    assert np.abs(fields["pressure"] - 1.0).max() < 1e-14
    assert np.abs(fields["velocity"] - 1.0).max() < 1e-14


@pytest.mark.parametrize("osmp_order", OSMP_ORDERS)
def test_double_mach_reflection_reaches_the_final_time(osmp_order, tmp_path):
    """Finite and admissible at t = 0.2, with nothing but the scheme of the article.

    256 x 64 cells is the coarsest mesh on which OSMP7 used to diverge.
    """
    centers, _, fields = read(osmp(tmp_path, "double_mach_reflection", 6, osmp_order, Tf=0.2))

    assert np.isfinite(fields["rho"]).all()
    assert np.isfinite(fields["pressure"]).all()
    assert fields["rho"].min() > 0.0
    assert fields["pressure"].min() > 0.0
    # The reflection has happened: the density behind the Mach stem is far above the 8 of the incident shock.
    assert fields["rho"].max() > 18.0
    assert mach_stem_spread(centers, fields["rho"]) < 0.08


@pytest.mark.parametrize("osmp_order", OSMP_ORDERS)
def test_sedov_is_its_own_mirror_image(osmp_order, tmp_path):
    """The blast is centred, so the solution is symmetric in x and in y.

    A sweep treats the waves going left and the waves going right with the same
    formulas read in opposite directions. Any term that does not change sign as
    it should under the mirror shows here. The rotation by 90 degrees of the
    other Sedov test is not a symmetry of a Strang splitting, which runs x
    before y.
    """
    centers, _, fields = read(osmp(tmp_path, "sedov_blast", 6, osmp_order, Tf=0.05))

    for name in ("rho", "pressure"):
        grid, _ = on_grid(centers, fields[name])
        scale = np.abs(grid).max()
        assert np.abs(grid - grid[:, ::-1]).max() < 1e-11 * scale
        assert np.abs(grid - grid[::-1, :]).max() < 1e-11 * scale


@pytest.mark.slow
@pytest.mark.parametrize("osmp_order", OSMP_ORDERS)
def test_each_order_converges_at_its_own_rate(osmp_order, tmp_path):
    """The advected pulse is a linear wave, where OSMPo is of order o in time and space.

    The limiter is on: the monotonicity-preserving constraints leave a smooth
    extremum alone, which is the point of them over the TVD ones.
    """
    tf, amplitude, sigma, x0, u0 = 0.2, 0.5, 0.04, 0.3, 1.0  # euler/init/advected_pulse.hpp

    l1 = []
    for level in (7, 8):
        centers, volume, fields = read(osmp(tmp_path / f"level{level}", "advected_pulse", level, osmp_order, Tf=tf))
        shifted = centers[:, 0] - x0 - u0 * tf
        exact = 1.0 + amplitude * np.exp(-0.5 * shifted * shifted / (sigma * sigma))
        l1.append(np.sum(np.abs(fields["rho"] - exact) * volume) / np.sum(volume))
    order = np.log2(l1[0] / l1[1])

    assert order > MIN_ORDER[osmp_order], f"L1 errors {l1}, order {order}"


@pytest.mark.slow
def test_tvd_constraints_clip_the_extremum_and_mp_ones_do_not(tmp_path):
    """OSTVD7 loses the order at the top of the pulse, OSMP7 keeps it (table 1 of the article)."""
    tf, amplitude, sigma, x0, u0 = 0.2, 0.5, 0.04, 0.3, 1.0

    error = {}
    for limiter in ("mp", "tvd", "none"):
        centers, volume, fields = read(osmp(tmp_path / limiter, "advected_pulse", 8, 7, Tf=tf, osmp_limiter=limiter))
        shifted = centers[:, 0] - x0 - u0 * tf
        exact = 1.0 + amplitude * np.exp(-0.5 * shifted * shifted / (sigma * sigma))
        error[limiter] = np.sum(np.abs(fields["rho"] - exact) * volume) / np.sum(volume)

    assert error["mp"] < 1.01 * error["none"], error
    assert error["tvd"] > 10.0 * error["none"], error


@pytest.mark.slow
@pytest.mark.parametrize(
    "osmp_order,options",
    [
        (7, {}),
        (5, {}),
        (3, {"osmp_fix_all_waves": True, "osmp_entropy_fix": 0.2}),
    ],
    ids=["osmp7", "osmp5", "osmp3-all-waves"],
)
def test_mach_stem_is_straight(osmp_order, options, tmp_path):
    """512 x 128 cells, where the carbuncle of the Roe flux shows.

    OSMP5 and OSMP7 keep the stem straight as they are. OSMP3 does not: its stem
    runs 0.15 ahead along the wall, and what cures it is the entropy fix on the
    linear waves, which is what is asked of it here.
    """
    centers, _, fields = read(osmp(tmp_path, "double_mach_reflection", 7, osmp_order, Tf=0.2, **options))

    assert mach_stem_spread(centers, fields["rho"]) < 0.08
