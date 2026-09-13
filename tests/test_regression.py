# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""Field comparisons against stored references.

Every case is run on a UNIFORM mesh. On an adapted mesh this comparison is not
portable: a rounding difference of the order of 1e-16 on a detail sitting near
the multiresolution threshold flips a refinement decision, the mesh changes, and
the difference becomes macroscopic. The test would then fail on another compiler
without anything being wrong. Adapted runs are covered in test_validation.py,
through quantities that survive a change of mesh.

Regenerating the references is deliberate: pass --generate-ref and say in the
commit message why they moved.
"""

import itertools

import numpy as np
import pytest

from util import REFERENCE, compare_or_generate, run_case

SCHEMES = ["rusanov", "hll", "hllc"]

# (binary, case, level, final time, extra options). Levels are low on purpose:
# the reference files are versioned, and a case is not made sharper by holding
# more cells. The extra options are what a case needs beyond its name, and the
# `label` among them is what names the reference file: the three Lax & Liu
# configurations are one case run three ways.
CASES = [
    ("euler_1d", "advected_pulse", 8, 0.05, {}),
    ("euler_1d", "double_rarefaction", 8, 0.05, {}),
    ("euler_1d", "free_stream", 8, 0.05, {}),
    ("euler_1d", "sedov_blast", 8, 0.05, {}),
    ("euler_1d", "closed_box", 8, 0.05, {}),
    ("euler_1d", "sod_x", 8, 0.05, {}),
    ("euler_1d", "blast_periodic", 8, 0.05, {}),
    ("euler_2d", "sod", 5, 0.02, {}),
    ("euler_2d", "sod_x", 5, 0.02, {}),
    ("euler_2d", "lax_liu", 5, 0.02, dict(label="lax_liu3", riemann_config=3)),
    ("euler_2d", "lax_liu", 5, 0.02, dict(label="lax_liu4", riemann_config=4)),
    ("euler_2d", "lax_liu", 5, 0.02, dict(label="lax_liu12", riemann_config=12)),
    ("euler_2d", "sedov_blast", 5, 0.02, {}),
    ("euler_2d", "kelvin_helmholtz", 5, 0.02, {}),
    ("euler_2d", "free_stream", 5, 0.02, {}),
    ("euler_2d", "isentropic_vortex", 5, 0.02, {}),
    ("euler_2d", "double_mach_reflection", 5, 0.02, {}),
    ("euler_2d", "closed_box", 5, 0.02, {}),
    ("euler_2d", "blast_periodic", 5, 0.02, {}),
    ("euler_2d", "triple_point_single_gamma", 5, 0.02, {}),
    # level 5 in 3D, not 4: at level 4 the smallest cell centre sits at radius
    # 0.108 and the blast has radius 0.1, so not one cell falls inside it and the
    # case reduces to a uniform gas at rest.
    ("euler_3d", "sedov_blast", 5, 0.01, {}),
    ("euler_3d", "free_stream", 4, 0.01, {}),
    ("euler_3d", "closed_box", 4, 0.01, {}),
    # t_f = 0.03, not 0.01: one step is not enough, see the test below.
    ("euler_3d", "sod_x", 4, 0.03, {}),
    ("euler_3d", "lax_liu", 4, 0.01, {}),
    ("euler_3d", "blast_periodic", 4, 0.01, {}),
]

# The reference file is named after the run, so the test is too: without this
# the three lax_liu entries would be told apart only by the index pytest gives
# a dictionary.
IDS = [f"{binary}-{extra.get('label', case)}" for binary, case, _, _, extra in CASES]


@pytest.mark.parametrize("binary,case,level,tf,extra", CASES, ids=IDS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_matches_reference(binary, case, level, tf, extra, scheme, tmp_path, generate_ref):
    out, stem = run_case(
        binary, tmp_path, case, scheme=scheme, min_level=level, max_level=level, Tf=tf, **extra
    )
    compare_or_generate(out / stem, f"{binary}_{stem}", generate_ref)


# The free stream is the one case whose reference is meant to be the same for
# every solver: it is a uniform state that must not move, whoever computes the
# flux.
INDIFFERENT = {"free_stream"}
DISCRIMINATING = [case for case in CASES if case[1] not in INDIFFERENT]
DISCRIMINATING_IDS = [i for i, case in zip(IDS, CASES) if case[1] not in INDIFFERENT]


@pytest.mark.parametrize("binary,case,level,tf,extra", DISCRIMINATING, ids=DISCRIMINATING_IDS)
def test_the_reference_tells_the_solvers_apart(binary, case, level, tf, extra):
    """Three identical reference files say the run tested no Riemann solver.

    A run can be degenerate in two ways that both look healthy. It can be too
    short: from a gas at rest the HLL wave speeds come out at -c and +c, the
    solver reduces to Rusanov algebraically, and a single time step from such an
    initial state gives the two the same answer to the bit. Or it can be too
    coarse to see its own initial condition, which is what happened to the 3D
    Sedov blast at level 4.

    Comparing the stored references costs nothing and catches both.
    """
    name = f"{binary}_{extra.get('label', case)}"
    fields = ("rho", "pressure", "velocity")
    stored = {}
    for scheme in SCHEMES:
        path = REFERENCE / f"{name}_{scheme}.npz"
        assert path.exists(), f"no reference for {path.name}; run pytest --generate-ref"
        with np.load(path) as reference:
            stored[scheme] = {name_: reference[name_] for name_ in fields}

    # Two solvers are told apart as soon as one field differs. Asking for a
    # difference on the density alone would be stricter than the statement: the
    # density flux is rho*u, so a first step taken from rest updates the density
    # identically under every solver while the pressure already parts.
    for a, b in itertools.combinations(SCHEMES, 2):
        assert any(not np.array_equal(stored[a][name_], stored[b][name_]) for name_ in fields), (
            f"{name}: {a} and {b} produced the same solution, so this entry tests nothing about either"
        )
