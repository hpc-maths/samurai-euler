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

# (binary, case, level, final time). Levels are low on purpose: the reference
# files are versioned, and a case is not made sharper by holding more cells.
CASES = [
    ("euler_1d", "double_rarefaction", 8, 0.05),
    ("euler_1d", "free_stream", 8, 0.05),
    ("euler_1d", "sedov_blast", 8, 0.05),
    ("euler_1d", "closed_box", 8, 0.05),
    ("euler_2d", "sod", 5, 0.02),
    ("euler_2d", "riemann2d_config3", 5, 0.02),
    ("euler_2d", "riemann2d_config4", 5, 0.02),
    ("euler_2d", "riemann2d_config12", 5, 0.02),
    ("euler_2d", "sedov_blast", 5, 0.02),
    ("euler_2d", "kelvin_helmholtz", 5, 0.02),
    ("euler_2d", "free_stream", 5, 0.02),
    ("euler_2d", "isentropic_vortex", 5, 0.02),
    ("euler_2d", "double_mach_reflection", 5, 0.02),
    ("euler_2d", "closed_box", 5, 0.02),
    ("euler_3d", "sedov_blast", 4, 0.01),
    ("euler_3d", "free_stream", 4, 0.01),
    ("euler_3d", "closed_box", 4, 0.01),
]


@pytest.mark.parametrize("binary,case,level,tf", CASES)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_matches_reference(binary, case, level, tf, scheme, tmp_path, generate_ref):
    out, stem = run_case(
        binary, tmp_path, case, scheme=scheme, min_level=level, max_level=level, Tf=tf
    )
    compare_or_generate(out / stem, f"{binary}_{stem}", generate_ref)
