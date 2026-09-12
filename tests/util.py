# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""Helpers shared by the test suite: run a solver, read what it wrote.

The solvers are driven as subprocesses rather than linked against, so the tests
exercise exactly what a user runs, command line included. Every run gets its own
working directory: output file names depend only on the test case and the
scheme, so two binaries asked for the same case would otherwise overwrite each
other.
"""

import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path(os.environ.get("EULER_BUILD_DIR", ROOT / "build"))

# Number of corners of a cell, by dimension, as written in the connectivity.
_DIM_OF_CORNERS = {2: 1, 4: 2, 8: 3}


def executable(name):
    """Path to a built solver. EULER_BUILD_DIR overrides the default <root>/build."""
    for candidate in (BUILD / name, BUILD / "Release" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{name} not found under {BUILD}. Build the project first, or point "
        f"EULER_BUILD_DIR at the build directory."
    )


def run(binary, workdir, **options):
    """Run `binary` in `workdir` and return the directory holding its output.

    Options are passed through as command line flags: ``max_level=7`` becomes
    ``--max-level 7``, and ``refine_boundary=True`` becomes ``--refine-boundary``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [str(executable(binary))]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            cmd.append(flag)
        elif value is not False and value is not None:
            cmd += [flag, str(value)]

    completed = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)}\nexit {completed.returncode}\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return workdir / options.get("path", "results")


def run_case(binary, workdir, case, scheme="hllc", **options):
    """Run one test case and return (output directory, output file stem).

    The scheme is always passed explicitly: its default differs between the
    binaries, and the output file is named after it.
    """
    out = run(binary, workdir, test_case=case, scheme=scheme, **options)
    return out, f"{case}_{scheme}"


def read(h5file):
    """Return cell centers, cell volumes and the primitive fields of one output.

    Works in one, two and three dimensions. Cells are axis-aligned boxes, so the
    volume is the product of the extents of their corners. In 1D samurai writes
    the velocity as a single field named `velocity` rather than `velocity_0`;
    either way it comes back with shape (ncells, dim).
    """
    h5file = Path(h5file)
    if h5file.suffix != ".h5":
        h5file = h5file.with_suffix(".h5")

    with h5py.File(str(h5file), "r") as handle:
        mesh = handle["mesh"]
        corners = mesh["points"][:][mesh["connectivity"][:]]
        dim = _DIM_OF_CORNERS[corners.shape[1]]

        lower = corners[:, :, :dim].min(axis=1)
        upper = corners[:, :, :dim].max(axis=1)
        centers = 0.5 * (lower + upper)
        volume = np.prod(upper - lower, axis=1)

        group = mesh["fields"]
        names = [f"velocity_{d}" for d in range(dim)] if dim > 1 else ["velocity"]
        fields = {
            "rho": group["rho"][:],
            "pressure": group["pressure"][:],
            "velocity": np.stack([group[n][:] for n in names], axis=1),
        }

    return centers, volume, fields


def conservative(volume, fields, gamma=1.4):
    """Total mass, momentum and energy of a solution, for conservation checks."""
    rho, p, v = fields["rho"], fields["pressure"], fields["velocity"]
    kinetic = 0.5 * rho * np.sum(v * v, axis=1)
    energy = p / (gamma - 1.0) + kinetic
    return {
        "mass": np.sum(rho * volume),
        "momentum": np.sum(v * (rho * volume)[:, None], axis=0),
        "energy": np.sum(energy * volume),
    }


def finest_cell_size(volume, dim):
    """Size of the smallest cell, i.e. the resolution the mesh actually reaches."""
    return volume.min() ** (1.0 / dim)


def sedov_blast_energy(dim):
    """The blast energy euler/init/sedov_blast.hpp compiles in, read from it.

    Restating the three numbers here would make a second source of truth that
    eventually stops agreeing with the first.
    """
    text = (ROOT / "euler" / "init" / "sedov_blast.hpp").read_text()
    body = text.split("constexpr double blast_energy()", 1)[1]
    values = re.findall(r"return\s+([0-9.eE+-]+);", body)[:3]
    assert len(values) == 3, f"expected one blast energy per dimension, read {values}"
    return float(values[dim - 1])


def level_count(volume):
    """Number of distinct cell sizes: 1 on a uniform mesh, more once adapted.

    Worth asserting on in any test that claims to exercise level interfaces: the
    multiresolution coarsens a smooth solution back to a uniform mesh, and the
    test would then pass without testing anything.
    """
    return np.unique(np.round(np.log2(volume / volume.min()))).size


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------
REFERENCE = Path(__file__).resolve().parent / "reference"


def compare_or_generate(output, name, generate, atol=1e-10, rtol=1e-12):
    """Compare one output against its stored reference, or write that reference.

    Only the fields and the cell centers are stored, not the whole HDF5 file:
    on a uniform mesh the mesh follows from the domain and the level, so the
    centers are enough to notice that it moved.
    """
    centers, volume, fields = read(output)
    current = {"centers": centers, "volume": volume, **fields}
    path = REFERENCE / f"{name}.npz"

    if generate:
        REFERENCE.mkdir(exist_ok=True)
        np.savez_compressed(path, **current)
        return

    assert path.exists(), f"no reference for {name}; run pytest --generate-ref"
    with np.load(path) as stored:
        for key, value in current.items():
            expected = stored[key]
            assert value.shape == expected.shape, f"{name}: {key} shape {value.shape} vs {expected.shape}"
            np.testing.assert_allclose(value, expected, atol=atol, rtol=rtol, err_msg=f"{name}: {key}")
