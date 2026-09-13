# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""Quantitative checks against the exact solution.

These run longer than the rest and are marked slow, so they stay out of the
default run. They assert on scalars rather than on fields, which is what makes
them usable on an adapted mesh: the mesh may legitimately differ from one
machine to the next, the measured order and the error norms may not.

The exact isentropic vortex comes from python/error_analysis.py rather than
being restated here. It has to agree with euler/init/isentropic_vortex.hpp, and
two copies of the same formula would eventually stop agreeing.
"""

import sys

import numpy as np
import pytest

from util import ROOT, level_count, read, run_case, sedov_blast_energy

sys.path.insert(0, str(ROOT / "python"))
from error_analysis import errors, exact_vortex  # noqa: E402

pytestmark = pytest.mark.slow

# The vortex is smooth, so the error decays at the design order of the scheme.
# HLLC without reconstruction and explicit Euler in time are first order.
#
# The measured order approaches the design value from below, so the coarsest
# pair of resolutions is the loosest: at levels 5 to 7 it reads 0.95 then 0.97.
# Hence two thresholds rather than one. Move both up by one when the
# MUSCL-Hancock update lands: this is the assertion that keeps second order once
# it has been gained.
MIN_ORDER = 0.90         # every consecutive pair
MIN_FINEST_ORDER = 0.95  # the finest pair, where the asymptotic rate shows

TF = 0.2
LEVELS = [5, 6, 7]


def vortex_error(workdir, level, min_level=None, **options):
    """L1 error on the density of one vortex run, with the mesh it used.

    `level` is the finest level. `min_level` defaults to it, which makes the mesh
    uniform; pass a coarser one to let the multiresolution adapt.
    """
    out, stem = run_case(
        "euler_2d",
        workdir,
        "isentropic_vortex",
        min_level=level if min_level is None else min_level,
        max_level=level,
        Tf=TF,
        **options,
    )
    centers, volume, fields = read(out / stem)
    reference = exact_vortex(centers[:, 0], centers[:, 1], TF)
    l1, _, _ = errors(fields["rho"], reference["rho"], volume)
    return l1, volume


def test_vortex_converges_at_the_design_order(tmp_path):
    """Measure the order on uniform meshes and hold it to the design value."""
    l1 = [vortex_error(tmp_path / f"level{level}", level)[0] for level in LEVELS]
    orders = [np.log2(a / b) for a, b in zip(l1, l1[1:])]

    detail = f"L1 errors {l1}, orders {orders}"
    assert all(order > MIN_ORDER for order in orders), f"{detail}, all expected above {MIN_ORDER}"
    assert orders[-1] > MIN_FINEST_ORDER, f"{detail}, finest expected above {MIN_FINEST_ORDER}"


def test_adaptation_costs_no_accuracy(tmp_path):
    """An adapted mesh must reach the same error as the uniform one, with fewer cells.

    Two assertions on purpose. Precision alone would pass with a threshold so
    loose that the mesh never coarsens, which is the failure mode that matters:
    adaptation that no longer adapts. Compression alone would pass with a
    threshold so aggressive that the solution is wrong.
    """
    level = 7
    uniform_l1, uniform_volume = vortex_error(tmp_path / "uniform", level)
    adapted_l1, adapted_volume = vortex_error(
        tmp_path / "adapted", level, min_level=level - 3, mr_eps=1e-5
    )

    assert level_count(adapted_volume) > 1, "the mesh stayed uniform, this exercises nothing"

    compression = adapted_volume.size / uniform_volume.size
    assert compression < 0.8, f"only {100 * (1 - compression):.0f}% of the cells saved"
    assert adapted_l1 < 1.5 * uniform_l1, (
        f"adapted L1 {adapted_l1:.3e} against uniform {uniform_l1:.3e}"
    )


# ---------------------------------------------------------------------------
# The published test cases, against the solution their paper gives
# ---------------------------------------------------------------------------
# These check that the cases reproduce the papers they were taken from.
# test_regression.py checks something else, that they still compute what they
# computed yesterday. The thresholds are what a first-order scheme
# reaches at these resolutions, with room to spare; they should be tightened
# when the MUSCL-Hancock update lands.
from exact_riemann import solution as exact_riemann_solution  # noqa: E402
from exact_riemann import star_state  # noqa: E402
from sedov_exact import shock_radius  # noqa: E402


def test_double_rarefaction_matches_toro_test_2(tmp_path):
    """The 123 problem against the exact solution of Toro's Table 4.1, test 2."""
    level, tf = 12, 0.15
    left, right = (1.0, -2.0, 0.4), (1.0, 2.0, 0.4)

    out, stem = run_case("euler_1d", tmp_path, "double_rarefaction",
                         min_level=level, max_level=level, Tf=tf)
    centers, volume, fields = read(out / stem)
    x = centers[:, 0]
    rho, u, p = exact_riemann_solution(x, tf, left, right, x0=0.5)

    def l1(computed, exact):
        return float(np.sum(np.abs(computed - exact) * volume) / np.sum(volume))

    assert l1(fields["rho"], rho) < 1e-2
    assert l1(fields["velocity"][:, 0], u) < 3e-2
    assert l1(fields["pressure"], p) < 1e-2

    # the near-vacuum is what the case is for: a first-order scheme sits above
    # the exact star pressure, and must not sit far above it
    p_star, _ = star_state(left, right)
    assert p_star < fields["pressure"].min() < 3.0 * p_star


def test_sod_matches_its_exact_solution(tmp_path):
    """Sod's tube, rotated 45 degrees, against the exact solution.

    Two things at once: the waves must be in the right place, and the solution
    must stay one-dimensional along the diagonal. The transverse velocity is the
    isotropy measure the rotation was introduced for.
    """
    level, tf = 8, 0.2
    left, right = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)

    out, stem = run_case("euler_2d", tmp_path, "sod",
                         min_level=level, max_level=level, Tf=tf)
    centers, volume, fields = read(out / stem)
    x, y = centers[:, 0], centers[:, 1]

    # coordinate across the interface, and the velocity split along and across it
    band = np.abs(x - y) < 0.3  # away from the corners, where outflow is not exact
    xi = (x + y - 1.0) / np.sqrt(2.0)
    along = (fields["velocity"][:, 0] + fields["velocity"][:, 1]) / np.sqrt(2.0)
    across = (fields["velocity"][:, 0] - fields["velocity"][:, 1]) / np.sqrt(2.0)

    rho, u, p = exact_riemann_solution(xi[band], tf, left, right, x0=0.0)
    weight = volume[band]

    def l1(computed, exact):
        return float(np.sum(np.abs(computed - exact) * weight) / np.sum(weight))

    assert l1(fields["rho"][band], rho) < 2e-2
    assert l1(along[band], u) < 2e-2
    assert l1(fields["pressure"][band], p) < 2e-2
    assert np.abs(across[band]).max() < 1e-2


@pytest.mark.parametrize("binary,dim,level,tf", [("euler_2d", 2, 9, 0.6), ("euler_1d", 1, 12, 0.6)])
def test_sedov_shock_sits_where_the_similarity_solution_puts_it(binary, dim, level, tf, tmp_path):
    """The blast energy is only meaningful through the shock radius it produces.

    Measured as the outermost radius at which the density is still above halfway
    to its peak, which is where a smeared shock front has its middle. A wrong
    blast energy shows up here and nowhere else: every other check of this case
    is a symmetry or a positivity, and both survive any energy at all.
    """
    out, stem = run_case(binary, tmp_path, "sedov_blast",
                         min_level=level, max_level=level, Tf=tf)
    centers, _, fields = read(out / stem)

    r = np.linalg.norm(centers[:, :dim], axis=1)
    rho = fields["rho"]
    front = r[rho > 0.5 * (rho.max() + 1.0)].max()

    expected = shock_radius(sedov_blast_energy(dim), tf, dim)
    assert abs(front / expected - 1.0) < 0.1, f"shock at {front:.4f}, similarity solution at {expected:.4f}"


@pytest.mark.parametrize("case", ["riemann2d_config3", "riemann2d_config4", "riemann2d_config12"])
def test_riemann_2d_keeps_the_symmetry_of_its_configuration(case, tmp_path):
    """Lax & Liu configurations 3, 4 and 12 are symmetric about the diagonal.

    Their initial data is invariant under (x,y,u,v) -> (y,x,v,u), so the solution
    is too, and a uniform mesh carries that symmetry exactly. It is the cheapest
    check that the states were copied correctly: a single mistyped digit in one
    quadrant breaks it, while leaving a picture that still looks plausible.
    """
    level, tf = 7, 0.2
    out, stem = run_case("euler_2d", tmp_path, case,
                         min_level=level, max_level=level, Tf=tf)
    centers, _, fields = read(out / stem)

    n = int(round(np.sqrt(centers.shape[0])))
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    rho = fields["rho"][order].reshape(n, n)
    vx = fields["velocity"][order, 0].reshape(n, n)
    vy = fields["velocity"][order, 1].reshape(n, n)

    assert np.abs(rho - rho.T).max() < 1e-12
    assert np.abs(vx - vy.T).max() < 1e-12


# ---------------------------------------------------------------------------
# Second order
# ---------------------------------------------------------------------------
# These keep the second order once it has been gained. They fail as soon as a
# reconstruction, a boundary condition or a prediction operator drops back to
# first order, which is a change no other test in the suite would notice.
#
# The measurement is made without a slope limiter. Every limiter clips at a
# smooth extremum, which is exactly where these two solutions live, and the
# clipping costs a fraction of an order that says nothing about the scheme.
# The runs that follow the limiter are checked too, more loosely.
SECOND_ORDER = {"order": 2, "slope_limiter": "none"}
MIN_SECOND_ORDER = 1.9

PULSE_LEVELS = [8, 9, 10]


def test_vortex_reaches_second_order(tmp_path):
    """Two dimensions, uniform mesh. This is the exit criterion of the lot."""
    l1 = [vortex_error(tmp_path / f"level{level}", level, **SECOND_ORDER)[0] for level in LEVELS]
    orders = [np.log2(a / b) for a, b in zip(l1, l1[1:])]

    assert orders[-1] > MIN_SECOND_ORDER, f"L1 errors {l1}, orders {orders}"


def test_second_order_survives_adaptation(tmp_path):
    """Same order on an adapted mesh, with the threshold scaled as h^2.

    An epsilon held fixed while the mesh refines eventually dominates the
    discretization error and flattens the curve, so it follows the resolution
    down. What is being tested is that adaptation costs no order, not that a
    particular epsilon is a good one.
    """
    l1 = [
        vortex_error(
            tmp_path / f"adapted{level}",
            level,
            min_level=level - 2,
            mr_eps=1e-3 * 4.0 ** -(level - LEVELS[0]),
            **SECOND_ORDER,
        )[0]
        for level in LEVELS
    ]
    orders = [np.log2(a / b) for a, b in zip(l1, l1[1:])]

    assert orders[-1] > MIN_SECOND_ORDER, f"L1 errors {l1}, orders {orders}"


def pulse_error(workdir, level, **options):
    """L1 error on the advected pulse, against the profile translated by u t.

    The three constants are those of euler/init/advected_pulse.hpp. They are
    restated rather than read from it because a Gaussian written twice is
    cheaper to keep in step than a parser, but they do have to be kept in step.
    """
    amplitude, sigma, x0, u0 = 0.5, 0.04, 0.3, 1.0

    out, stem = run_case("euler_1d", workdir, "advected_pulse",
                         min_level=level, max_level=level, Tf=TF, **options)
    centers, volume, fields = read(out / stem)

    shifted = centers[:, 0] - x0 - u0 * TF
    exact = 1.0 + amplitude * np.exp(-0.5 * shifted * shifted / (sigma * sigma))

    return float(np.sum(np.abs(fields["rho"] - exact) * volume) / np.sum(volume))


@pytest.mark.parametrize("integrator", ["euler", "ssprk2"])
def test_second_order_in_one_dimension(integrator, tmp_path):
    """Both integrators are second order in one dimension.

    With `euler` the flux carries the Hancock predictor, and this is the test
    that says the predictor is right: it is the one place where it is expected
    to reach second order, transverse terms being what it cannot see and one
    dimension having none.
    """
    l1 = [pulse_error(tmp_path / f"level{level}", level, time_integrator=integrator, **SECOND_ORDER)
          for level in PULSE_LEVELS]
    orders = [np.log2(a / b) for a, b in zip(l1, l1[1:])]

    assert orders[-1] > MIN_SECOND_ORDER, f"L1 errors {l1}, orders {orders}"


def test_limited_reconstruction_stays_close_to_second_order(tmp_path):
    """The default limiter costs accuracy at the extremum, and not much more.

    A limiter that has stopped limiting would pass the tests above and lose the
    property they exist for; one that clips everything would keep the property
    and lose the order. This holds the default between the two.
    """
    l1 = [pulse_error(tmp_path / f"level{level}", level, order=2, slope_limiter="moncen")
          for level in PULSE_LEVELS]
    orders = [np.log2(a / b) for a, b in zip(l1, l1[1:])]

    assert orders[-1] > 1.8, f"L1 errors {l1}, orders {orders}"
