# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""Properties the solver must hold whatever the scheme computes.

These tests own no reference file. They check statements that stay true when the
numerics legitimately change, so a better scheme cannot make them fail, and they
cannot be satisfied by regenerating anything.
"""

import numpy as np
import pytest

from util import conservative, level_count, read, run_case

# Resolutions chosen so that each test runs in a few seconds.
DIMENSIONS = [("euler_1d", 8), ("euler_2d", 6), ("euler_3d", 4)]

# Both orders are held to the same invariants. The second one is where they
# bite: a MUSCL reconstruction reads a second layer of ghost cells and takes
# slopes across level jumps, which is two more ways to break a uniform state.
ORDERS = [1, 2]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("binary,level", DIMENSIONS)
def test_free_stream_stays_uniform(binary, level, order, tmp_path):
    """A uniform flow is an exact solution: it must not move at all."""
    out, stem = run_case(binary, tmp_path, "free_stream", min_level=level, max_level=level, Tf=0.05, order=order)
    _, _, fields = read(out / stem)

    assert np.abs(fields["rho"] - 1.0).max() < 1e-14
    assert np.abs(fields["pressure"] - 1.0).max() < 1e-14
    assert np.abs(fields["velocity"] - 1.0).max() < 1e-14


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("binary,level", DIMENSIONS)
def test_free_stream_across_level_interfaces(binary, level, order, tmp_path):
    """Same, on a mesh that really carries level jumps.

    A uniform state has no detail anywhere, so the multiresolution coarsens it
    back to a single level and the check above would prove nothing about the
    adapted machinery. --refine-boundary pins the boundary at max_level, which
    restores real interfaces; the assertion on level_count is what makes sure
    they are there.
    """
    out, stem = run_case(
        binary,
        tmp_path,
        "free_stream",
        min_level=level - 2,
        max_level=level,
        Tf=0.05,
        refine_boundary=True,
        order=order,
    )
    _, volume, fields = read(out / stem)

    assert level_count(volume) > 1, "the mesh is uniform, this exercises nothing"
    assert np.abs(fields["rho"] - 1.0).max() < 1e-14
    assert np.abs(fields["pressure"] - 1.0).max() < 1e-14
    assert np.abs(fields["velocity"] - 1.0).max() < 1e-14


def test_strang_is_a_single_sweep_in_one_dimension(tmp_path):
    """With one direction to sweep, Strang splitting is the Hancock step itself.

    Not an approximation of it: half a step along nothing, then the whole step
    along x, then nothing again. The two runs have to agree bit for bit, which
    makes this the cheapest check that the sweep machinery does not disturb
    what it wraps.
    """
    runs = {}
    for integrator in ("euler", "strang"):
        out, stem = run_case("euler_1d", tmp_path / integrator, "advected_pulse",
                             min_level=9, max_level=9, Tf=0.05,
                             order=2, time_integrator=integrator)
        _, _, runs[integrator] = read(out / stem)

    for name in ("rho", "pressure", "velocity"):
        np.testing.assert_array_equal(runs["euler"][name], runs["strang"][name])


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "binary,case,level",
    [
        ("euler_1d", "double_rarefaction", 8),
        ("euler_1d", "sedov_blast", 8),
        ("euler_2d", "sedov_blast", 6),
        ("euler_2d", "double_mach_reflection", 6),
        # level 5, not 4: at level 4 no cell centre falls inside the blast in
        # three dimensions, and the case is then a uniform gas at rest, whose
        # positivity says nothing about the scheme.
        ("euler_3d", "sedov_blast", 5),
    ],
)
def test_density_and_pressure_stay_positive(binary, case, level, order, tmp_path):
    """The Euler system is only well posed for positive density and pressure.

    double_rarefaction leaves a near vacuum where the internal energy is a small
    difference of large numbers, and the Sedov blast starts from an ambient
    pressure of 1e-5: both fail loudly rather than quietly when a scheme stops
    being admissible.
    """
    out, stem = run_case(binary, tmp_path, case, min_level=level, max_level=level, Tf=0.05, order=order)
    _, _, fields = read(out / stem)

    assert fields["rho"].min() > 0.0
    assert fields["pressure"].min() > 0.0


def test_sedov_keeps_its_symmetry(tmp_path):
    """The Sedov blast is radial, so the solution must be invariant by rotation.

    Comparing the solution with itself rotated by 90 degrees catches a
    directional bias in the flux, the prediction or the adaptation, which a norm
    against a radial profile would average away.
    """
    out, stem = run_case("euler_2d", tmp_path, "sedov_blast", min_level=6, max_level=6, Tf=0.05)
    centers, _, fields = read(out / stem)

    # The domain is [-1, 1]^2 centred on the blast, so rotating by 90 degrees
    # maps (x, y) to (-y, x) and permutes the cells among themselves.
    rotated = np.column_stack((-centers[:, 1], centers[:, 0]))
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    order_rotated = np.lexsort((rotated[:, 0], rotated[:, 1]))

    for name in ("rho", "pressure"):
        direct = fields[name][order]
        turned = fields[name][order_rotated]
        assert np.abs(direct - turned).max() < 1e-12


def test_closed_box_conserves_mass_and_energy(tmp_path):
    """Solid walls let nothing through, so mass and energy are exactly constant.

    What this catches is a non-conservative *addition* to the update: a limiter
    that rewrites a cell without compensating elsewhere, a source term, a badly
    written predictor step. It does not catch a wrong flux, because a flux-based
    scheme computes one value per face and applies it with opposite signs to the
    two cells: scaling every flux leaves the interior contributions cancelling in
    pairs and conservation intact. Checked by mutation, a 1e-9 per-step rescaling
    of the density fires this test at 2e-8.

    It is also the only test that exercises the Reflective boundary.
    """
    out, stem = run_case("euler_2d", tmp_path, "closed_box", min_level=6, max_level=6, Tf=0.05)
    _, volume_0, fields_0 = read(out / f"{stem}_init")
    _, volume_1, fields_1 = read(out / stem)

    before = conservative(volume_0, fields_0)
    after = conservative(volume_1, fields_1)

    assert abs(after["mass"] - before["mass"]) / before["mass"] < 1e-12
    assert abs(after["energy"] - before["energy"]) / before["energy"] < 1e-12
    # and the solution did move, otherwise conservation is trivial
    assert np.abs(fields_1["rho"] - fields_0["rho"]).max() > 1e-3


@pytest.mark.parametrize("interface", [0.5, 0.8])
def test_riemann_interface_sits_where_it_is_asked_to(interface, tmp_path):
    """`--riemann-interface` moves the corner the four quadrants meet at.

    0.8 is the default, the position the article uses; 0.5 is the convention of
    the papers that classify the configurations. The upper right quadrant of
    configuration 3 is the only one at density 1.5, so counting its cells says
    exactly where the corner landed, with no tolerance needed on a uniform mesh.
    """
    level = 6
    out, stem = run_case("euler_2d", tmp_path, "lax_liu", riemann_config=3,
                         riemann_interface=interface, min_level=level, max_level=level, Tf=0.002)
    _, _, fields = read(out / f"{stem}_init")

    n = 2**level
    across = sum(1 for i in range(n) if (i + 0.5) / n >= interface)
    assert (fields["rho"] == 1.5).sum() == across * across


def test_periodic_box_conserves_mass_energy_and_momentum(tmp_path):
    """A periodic box has no boundary at all, so nothing can leave it.

    The closed box above conserves through the reflective wall; here the ghost
    cells are filled by samurai's periodic update instead, which is different
    code, and momentum is conserved as well, which a wall does not conserve.

    It also pins the gas. The total energy is reconstructed here from the
    pressure at gamma = 5/3, the value blast_periodic declares; the solver
    conserves the energy it computed with its own gamma, and the two agree only
    if they are the same gamma. Run the case at 1.4 and this test fails by
    several percent, which makes it the check that --gamma and the equation of
    state carried by a test case are really data.
    """
    gamma = 5.0 / 3.0

    out, stem = run_case("euler_2d", tmp_path, "blast_periodic", min_level=6, max_level=6, Tf=0.05)
    _, volume_0, fields_0 = read(out / f"{stem}_init")
    _, volume_1, fields_1 = read(out / stem)

    before = conservative(volume_0, fields_0, gamma)
    after = conservative(volume_1, fields_1, gamma)

    assert abs(after["mass"] - before["mass"]) / before["mass"] < 1e-12
    assert abs(after["energy"] - before["energy"]) / before["energy"] < 1e-12
    # the gas starts at rest and the box is symmetric, so momentum stays at zero
    assert np.abs(after["momentum"]).max() < 1e-12 * before["mass"]
    # and the solution did move, otherwise conservation is trivial
    assert np.abs(fields_1["rho"] - fields_0["rho"]).max() > 1e-3


@pytest.mark.parametrize("binary,case,level", [("euler_1d", "double_rarefaction", 8), ("euler_2d", "sod", 6)])
def test_restart_reproduces_the_run(binary, case, level, tmp_path):
    """Reloading the dumped state and running from it must give the same answer.

    The comparison is against a restart taken at the initial time rather than
    mid-run: a run that stops at T computes its last step as T minus the current
    time, so its sequence of time steps differs from that of a longer run, and
    the two trajectories would legitimately part. Restarting from the initial
    dump keeps the sequence identical, which makes the check exact and turns it
    into a test of the dump and load round trip.
    """
    direct, stem = run_case(binary, tmp_path / "direct", case, min_level=level, max_level=level, Tf=0.04)
    restarted, _ = run_case(
        binary,
        tmp_path / "restarted",
        case,
        min_level=level,
        max_level=level,
        Tf=0.04,
        restart_file=str(direct / f"{stem}_restart_init"),
    )

    _, volume_a, fields_a = read(direct / stem)
    _, volume_b, fields_b = read(restarted / stem)

    assert volume_a.shape == volume_b.shape
    for name in ("rho", "pressure", "velocity"):
        assert np.array_equal(fields_a[name], fields_b[name]), f"{name} differs after restart"
