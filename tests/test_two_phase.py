# Copyright 2025 the samurai team
# SPDX-License-Identifier:  BSD-3-Clause
"""The five-equation two-phase model.

Three things are worth asserting on, and they are different in kind.

The first is the property the model exists for: an interface carried by a
uniform flow must not disturb the pressure or the velocity. A conservative
scheme on two materials fails it -- averaging two gases in one cell gives a
mixture whose pressure is neither one's -- and no amount of resolution repairs
that, so this is an invariant and not a convergence measurement.

The second is that the four conservation laws are conserved and the fifth stays
in the unit interval, which is what says the non-conservative term is applied to
the right cells with the right sign.

The third is the shock tube of section 6.1.1 against its exact solution, which
python/exact_two_phase_riemann.py computes. It runs longer and is marked slow.
"""

import sys

import numpy as np
import pytest

from util import REFERENCE, ROOT, compare_or_generate, read, run_case

sys.path.insert(0, str(ROOT / "python"))
from exact_two_phase_riemann import AIR, WATER, WATER_AIR, solution, star_state  # noqa: E402

ORDERS = [1, 2]


def total_energy(fields):
    """The conserved energy of the mixture, from the fields a run writes.

    rho e = G(alpha) p + P(alpha), the two sums of euler/two_phase/eos.hpp, with
    the gases of the exact solution rather than a second copy of their
    coefficients.
    """
    alpha = fields["alpha"]
    g = alpha / (WATER.gamma - 1.0) + (1.0 - alpha) / (AIR.gamma - 1.0)
    pi = alpha * WATER.gamma * WATER.pi / (WATER.gamma - 1.0) + (1.0 - alpha) * AIR.gamma * AIR.pi / (AIR.gamma - 1.0)
    kinetic = 0.5 * fields["rho"] * np.sum(fields["velocity"] ** 2, axis=1)
    return g * fields["pressure"] + pi + kinetic

# Where the diaphragm of the water-air tube is, and how long it runs. Must agree
# with euler/two_phase/init/water_air_shock_tube.hpp.
DIAPHRAGM = 0.7
TF = 9e-4


# ---------------------------------------------------------------------------
# T1 -- invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("binary,level", [("two_phase_1d", 8), ("two_phase_2d", 6)])
def test_an_advected_interface_leaves_pressure_and_velocity_alone(binary, level, order, tmp_path):
    """The interface condition, on the hardest pair of fluids in the repository.

    Water and air at one atmosphere, moving together at 100 m/s: the volume
    fraction travels and nothing else happens. The tolerance is round-off, not a
    physical smallness -- the model is built so that this is exact.
    """
    out, stem = run_case(
        binary, tmp_path, "advected_interface", min_level=level, max_level=level, Tf=1e-3, order=order
    )
    _, _, fields = read(out / stem)

    assert np.abs(fields["pressure"] / 1e5 - 1.0).max() < 1e-9
    assert np.abs(fields["velocity"][:, 0] / 100.0 - 1.0).max() < 1e-11
    assert fields["alpha"].min() >= 0.0 and fields["alpha"].max() <= 1.0

    # The interface must still be there: a run that dissolved it would pass the
    # two assertions above for the wrong reason.
    assert fields["alpha"].max() > 0.99, "the water is gone"
    assert fields["alpha"].min() < 0.01, "the air is gone"


@pytest.mark.parametrize("order", ORDERS)
def test_the_two_masses_and_the_energy_are_conserved(order, tmp_path):
    """A periodic box conserves what the four conservation laws say it does.

    The volume fraction is not one of them and must not be conserved; what it
    must do is stay in [0, 1], which is the other half of this test. The partial
    masses are computed from alpha and the mixture density, so a scheme that
    advected alpha inconsistently with the masses would break the first
    assertion, not only the last.
    """
    out, stem = run_case(
        "two_phase_1d", tmp_path, "advected_interface", min_level=8, max_level=8, Tf=1e-3, order=order
    )

    totals = []
    for name in (f"{stem}_init", stem):
        _, volume, fields = read(out / name)
        totals.append(
            (
                np.sum(fields["partial_rho_0"] * volume),
                np.sum(fields["partial_rho_1"] * volume),
                np.sum(fields["rho"] * fields["velocity"][:, 0] * volume),
                np.sum(total_energy(fields) * volume),
            )
        )

    names = ("water mass", "air mass", "momentum", "energy")
    for before, after, what in zip(totals[0], totals[1], names):
        assert abs(after - before) <= 1e-12 * abs(before), f"{what}: {before} -> {after}"


@pytest.mark.parametrize("order", ORDERS)
def test_the_volume_fraction_stays_in_the_unit_interval(order, tmp_path):
    """Through a strong shock tube, not only through a uniform flow.

    A volume fraction outside [0, 1] is a mixture that contains more or less than
    itself: the equation of state still returns a number, the run still finishes,
    and everything downstream is wrong. The floor in the solver clamps it, so
    what this really measures is how often it has to.
    """
    out, stem = run_case(
        "two_phase_1d", tmp_path, "water_air_shock_tube", min_level=9, max_level=9, Tf=TF, order=order
    )
    _, _, fields = read(out / stem)

    assert fields["alpha"].min() >= 0.0
    assert fields["alpha"].max() <= 1.0
    assert fields["rho"].min() > 0.0
    # The pressure may undershoot at the contact, but never below the vacuum of
    # the mixture: p + pi_inf(alpha) must stay positive or the sound speed is NaN.
    assert (fields["pressure"] + 6e8 * fields["alpha"]).min() > 0.0


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_one_fluid_reproduces_the_monofluid_solver(scheme, order, tmp_path):
    """With alpha = 1 everywhere the five-equation model IS the Euler system.

    Both solvers run Sod's tube, one with dim + 2 components and one with
    dim + 4, and they have to agree to round-off: the mixture equation of state
    at alpha = 1 must give back phase 0 alone, the volume fraction must stay
    exactly 1 through a shock and a rarefaction, and the partial density of the
    absent phase must stay exactly 0.

    This is the sharpest check available on the two-phase kernel, because the
    thing it is compared against is itself held to an exact solution. The two
    are not bit-for-bit -- the mixture law divides by 1/(gamma - 1) where the
    ideal gas multiplies by (gamma - 1), which is one unit in the last place --
    hence twelve digits rather than sixteen.
    """
    settings = dict(scheme=scheme, min_level=9, max_level=9, Tf=0.2, order=order)

    mono_out, mono_stem = run_case("euler_1d", tmp_path / "mono", "sod_x", **settings)
    two_out, two_stem = run_case("two_phase_1d", tmp_path / "two", "sod_x_pure", **settings)

    _, _, mono = read(mono_out / mono_stem)
    _, _, two = read(two_out / two_stem)

    assert np.all(two["alpha"] == 1.0), "the volume fraction moved away from a pure fluid"
    assert np.all(two["partial_rho_1"] == 0.0), "the absent phase gained mass"

    for name in ("rho", "pressure", "velocity"):
        np.testing.assert_allclose(two[name], mono[name], rtol=1e-12, atol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# T2 -- field comparison, uniform mesh
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_water_air_shock_tube_reference(scheme, generate_ref, tmp_path):
    """What the tube computed yesterday, on a uniform mesh and at low resolution."""
    out, stem = run_case(
        "two_phase_1d", tmp_path, "water_air_shock_tube", scheme=scheme, min_level=7, max_level=7, Tf=TF, order=2
    )
    compare_or_generate(out / stem, f"two_phase_1d_water_air_{scheme}", generate_ref)


def test_the_reference_tells_the_solvers_apart():
    """Three identical reference files would say the run tested no Riemann solver.

    The same check test_regression.py makes on the monofluid cases, for the same
    reason: a run too short or too coarse to leave its initial state gives the
    three solvers the same answer, and the comparison above would then pass
    without comparing anything.
    """
    stored = {}
    for scheme in ("rusanov", "hll", "hllc"):
        path = REFERENCE / f"two_phase_1d_water_air_{scheme}.npz"
        assert path.exists(), f"no reference for {path.name}; run pytest --generate-ref"
        with np.load(path) as reference:
            stored[scheme] = {name: reference[name] for name in ("rho", "pressure", "alpha")}

    for a, b in (("rusanov", "hll"), ("rusanov", "hllc"), ("hll", "hllc")):
        difference = max(np.abs(stored[a][name] - stored[b][name]).max() / np.abs(stored[a][name]).max() for name in stored[a])
        assert difference > 1e-3, f"the {a} and {b} references agree to {difference:.1e}"


# ---------------------------------------------------------------------------
# T3 -- against the exact solution
# ---------------------------------------------------------------------------
# The thresholds are the L1 errors the solver actually reaches at level 10, with
# a factor of two of room. They are not a convergence measurement: a shock tube
# has no order to measure, the error being dominated by the three discontinuities
# and falling as h^(2/3) at best.
TOLERANCE = {
    #        L1 rho   L1 u    L1 p    L1 alpha   shock, in fine cells
    1: dict(rho=1.5e-2, u=3e-2, p=1e-2, alpha=1.5e-2, shock=12),
    2: dict(rho=3e-3, u=5e-3, p=1e-3, alpha=4e-3, shock=4),
}


@pytest.mark.slow
@pytest.mark.parametrize("order", ORDERS)
def test_water_air_shock_tube_matches_its_exact_solution(order, tmp_path):
    """Section 6.1.1 against the exact stiffened-gas Riemann solution.

    Three waves, one of which is the material interface this model is for: a
    rarefaction running back into the water, the contact, and a shock running out
    into the air. The star state is p* = 4.7969e5 Pa and u* = 491.97 m/s, and the
    contact is what the article's figure shows at x = 1.14.
    """
    level = 10
    out, stem = run_case(
        "two_phase_1d", tmp_path, "water_air_shock_tube", min_level=level, max_level=level, Tf=TF, order=order
    )
    centers, volume, fields = read(out / stem)

    x = centers[:, 0]
    rho, u, p, alpha = solution(x, TF, *WATER_AIR, x0=DIAPHRAGM)

    def l1(numerical, exact, scale):
        return float(np.sum(np.abs(numerical - exact) * volume) / np.sum(volume) / scale)

    _, u_star = star_state(*WATER_AIR)
    tol = TOLERANCE[order]
    errors = {
        "rho": l1(fields["rho"], rho, 1e3),
        "u": l1(fields["velocity"][:, 0], u, u_star),
        "p": l1(fields["pressure"], p, 1e9),
        "alpha": l1(fields["alpha"], alpha, 1.0),
    }
    for name, error in errors.items():
        assert error < tol[name], f"L1 error on {name} is {error:.3e}, expected below {tol[name]:.3e}"

    # The contact is where the volume fraction crosses one half, and it must be
    # where the exact solution puts it, to the cell.
    dx = volume.min()
    contact = x[np.argmin(np.abs(fields["alpha"] - 0.5))]
    assert abs(contact - (DIAPHRAGM + u_star * TF)) <= 2 * dx, f"the contact is at {contact}"
