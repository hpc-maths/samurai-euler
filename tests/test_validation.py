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

from util import ROOT, level_count, read, run_case

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
