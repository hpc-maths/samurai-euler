# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""The performance numbers a run reports, checked against what it computed.

A metric nobody checks drifts from what it claims to measure, and a wrong one is
worse than none: the whole point of the sparsity index and of the throughput is
to be compared with the numbers of another code. So they are held here against
quantities measured independently — the cells of the file the run wrote, and the
cell count of the uniform mesh, which on a unit domain is arithmetic.

The timings themselves are not asserted on. What can be asserted is that they
add up: the mesh adaptation is part of the time to solution and the output is
not.
"""

import pytest

from util import read, run_case_with_metrics

# Resolutions chosen so that each run takes a second or two.
DIMENSIONS = [("euler_1d", 1, 8), ("euler_2d", 2, 6), ("euler_3d", 3, 4)]


@pytest.mark.parametrize("binary,dim,level", DIMENSIONS)
def test_a_uniform_mesh_is_the_reference_of_the_sparsity_index(binary, dim, level, tmp_path):
    """On a uniform mesh the sparsity index is 100% by definition.

    It is the denominator of every other run, so it is worth pinning: the domain
    of sod_x is the unit cube, and the mesh at max-level holds 2^(dim * level)
    cells whatever the adaptation does or does not do.
    """
    metrics, _ = run_case_with_metrics(binary, tmp_path, "sod_x", min_level=level, max_level=level, Tf=0.05)

    assert metrics["uniform_cells"] == 2 ** (dim * level)
    assert metrics["initial_cells"] == metrics["uniform_cells"]
    assert metrics["final_cells"] == metrics["uniform_cells"]
    assert metrics["initial_sparsity"] == pytest.approx(100.0)
    assert metrics["final_sparsity"] == pytest.approx(100.0)

    # Nothing refines or coarsens, so every step updates the same cells.
    assert metrics["steps"] > 0
    assert metrics["cell_updates"] == metrics["steps"] * metrics["uniform_cells"]


def test_the_sparsity_index_counts_the_cells_of_the_output(tmp_path):
    """The mesh the run reports at the final time is the mesh it wrote out.

    The count comes from the solver, the comparison from the HDF5 file: a
    sparsity index computed on ghosts, or on the mesh of one level, would pass
    every arithmetic check and fail this one.
    """
    metrics, output = run_case_with_metrics(
        "euler_2d", tmp_path, "lax_liu", riemann_config=3, min_level=3, max_level=6, Tf=0.05
    )
    _, volume, _ = read(output)

    assert metrics["final_cells"] == volume.size
    assert volume.size < metrics["uniform_cells"], "the mesh never coarsened, this exercises nothing"
    assert metrics["final_sparsity"] == pytest.approx(100.0 * volume.size / metrics["uniform_cells"])
    assert metrics["mcu_per_second"] == pytest.approx(1e-6 * metrics["cell_updates"] / metrics["run_time"])


def test_the_time_to_solution_holds_the_adaptation_and_not_the_output(tmp_path):
    """Adapting is part of solving; writing files is not.

    The second half is what makes two runs comparable: --nfiles is a choice of
    whoever runs the solver, and a table of times to solution that moved with it
    would compare nothing.
    """
    metrics, _ = run_case_with_metrics(
        "euler_2d", tmp_path, "lax_liu", riemann_config=3, min_level=3, max_level=6, Tf=0.05, nfiles=5
    )

    assert 0.0 < metrics["adapt_time"] < metrics["run_time"]
    assert metrics["adapt_fraction"] == pytest.approx(100.0 * metrics["adapt_time"] / metrics["run_time"])
    assert metrics["output_time"] > 0.0, "five files were asked for and none was timed"
