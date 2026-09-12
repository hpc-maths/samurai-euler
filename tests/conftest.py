# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""Test suite configuration.

Reference files for the field comparisons live in tests/reference and are
versioned. Regenerating them is deliberate: pass --generate-ref, and say in the
commit message why the reference moved.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--generate-ref",
        action="store_true",
        help="write the reference files instead of comparing against them",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: validation runs, excluded by default")


@pytest.fixture
def generate_ref(request):
    return request.config.getoption("--generate-ref")
