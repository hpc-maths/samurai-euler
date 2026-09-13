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

import pytest

from util import compare_or_generate, run_case

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
    ("euler_3d", "sedov_blast", 4, 0.01, {}),
    ("euler_3d", "free_stream", 4, 0.01, {}),
    ("euler_3d", "closed_box", 4, 0.01, {}),
    ("euler_3d", "sod_x", 4, 0.01, {}),
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
