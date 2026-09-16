"""add_repulsive_forces must repel distinct molecules only, never a molecule itself.

The NonbondedForce exception list covers 1-2/1-3/1-4 pairs only, so anything
further apart inside a probe stays inside a residueA == residueB interaction
group. Benzene's para C-H sits at 3.88 A, inside the default 4.0 A sigma, which
made each molecule push its own ring apart. These tests pin the fix.
"""

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest

from cosolvkit.cosolvent_system import (
    CosolventSystem,
    DEFAULT_REPULSIVE_EPSILON,
    DEFAULT_REPULSIVE_SIGMA,
)

BOX = 4.0  # nm, large enough that no pair is seen through the periodic images
FORCE_GROUP = 17


class _Stub:
    """The bits of CosolventSystem that add_repulsive_forces actually touches."""

    def __init__(self, system, topology, logger):
        self.system = system
        self.modeller = app.Modeller(topology, [mm.Vec3(0, 0, 0)] * topology.getNumAtoms())
        self.logger = logger


def _build(molecules, bonds=False):
    """A bare system of `molecules`: a list of (resname, [positions in nm])."""
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * u.nanometer)

    positions = []
    for resname, coords in molecules:
        residue = topology.addResidue(resname, chain)
        atoms = [topology.addAtom(f"C{k}", app.element.carbon, residue)
                 for k in range(len(coords))]
        if bonds:
            for a, b in zip(atoms, atoms[1:]):
                topology.addBond(a, b)
        for xyz in coords:
            system.addParticle(12.0 * u.amu)
            nb.addParticle(0.0, 0.3 * u.nanometer, 0.1 * u.kilojoule_per_mole)
            positions.append(mm.Vec3(*xyz) * u.nanometer)

    if bonds:
        nb.createExceptionsFromBonds(
            [(b[0].index, b[1].index) for b in topology.bonds()], 0.833, 0.5)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(BOX, 0, 0) * u.nanometer,
        mm.Vec3(0, BOX, 0) * u.nanometer,
        mm.Vec3(0, 0, BOX) * u.nanometer,
    )
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    return system, topology, positions


def _energy(system, positions, n_forces_before):
    """Potential energy of the forces added after `n_forces_before`, in kcal/mol."""
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(FORCE_GROUP if i >= n_forces_before else 0)
    context = mm.Context(system, mm.VerletIntegrator(1.0 * u.femtosecond),
                         mm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    state = context.getState(getEnergy=True, groups={FORCE_GROUP})
    return state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)


def _pair_energy(positions, pairs, epsilon=DEFAULT_REPULSIVE_EPSILON,
                 sigma=DEFAULT_REPULSIVE_SIGMA):
    """Reference sum of 4*eps*(sigma/r)^12 over `pairs`, switched as OpenMM does."""
    sigma_nm, cutoff, switch = sigma / 10.0, 1.0, 0.9
    total = 0.0
    for i, j in pairs:
        r = np.linalg.norm(np.array(positions[i].value_in_unit(u.nanometer))
                           - np.array(positions[j].value_in_unit(u.nanometer)))
        if r >= cutoff:
            continue
        term = 4 * epsilon * (sigma_nm / r) ** 12
        if r > switch:
            x = (r - switch) / (cutoff - switch)
            term *= 1 - 6 * x**5 + 15 * x**4 - 10 * x**3
        total += term
    return total


def _add(system, topology, spec, logger=None):
    stub = _Stub(system, topology, logger or _Logger())
    n_before = system.getNumForces()
    CosolventSystem.add_repulsive_forces(stub, spec)
    return n_before


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a):
        self.warnings.append(msg % a if a else msg)

    def info(self, msg, *a):
        self.infos.append(msg % a if a else msg)


# --- the regression -------------------------------------------------------

def test_no_energy_between_atoms_of_the_same_molecule():
    """One molecule alone, atoms well inside sigma: the force must read zero."""
    # 0.30 and 0.39 nm apart, both inside the 0.40 nm default sigma, and none of
    # them excluded because the topology carries no bonds.
    system, topology, positions = _build(
        [("BEN", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.30), (1.0, 1.0, 1.69)])])
    n_before = _add(system, topology, {"BB": {"residueA": "BEN", "residueB": "BEN"}})

    assert system.getNumForces() == n_before + 1
    assert _energy(system, positions, n_before) == pytest.approx(0.0, abs=1e-9)


def test_intermolecular_repulsion_is_still_applied():
    """Two molecules: only the 3x3 cross pairs contribute."""
    mol_a = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.30), (1.0, 1.0, 1.69)]
    mol_b = [(1.45, 1.0, 1.0), (1.45, 1.0, 1.30), (1.45, 1.0, 1.69)]
    system, topology, positions = _build([("BEN", mol_a), ("BEN", mol_b)])
    n_before = _add(system, topology, {"BB": {"residueA": "BEN", "residueB": "BEN"}})

    cross = [(i, j) for i in range(3) for j in range(3, 6)]
    expected = _pair_energy(positions, cross)
    assert expected > 0.01, "test geometry must produce a measurable repulsion"
    # rel=1e-5: the CPU platform holds positions in single precision and r^-12
    # amplifies that by a factor of 12.
    assert _energy(system, positions, n_before) == pytest.approx(expected, rel=1e-5)


def test_self_term_would_have_dominated():
    """Guard the magnitude: the masked-out intramolecular pairs are not small."""
    mol_a = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.30), (1.0, 1.0, 1.69)]
    mol_b = [(1.45, 1.0, 1.0), (1.45, 1.0, 1.30), (1.45, 1.0, 1.69)]
    system, topology, positions = _build([("BEN", mol_a), ("BEN", mol_b)])
    n_before = _add(system, topology, {"BB": {"residueA": "BEN", "residueB": "BEN"}})

    intra = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert _pair_energy(positions, intra) > _energy(system, positions, n_before)


# --- everything else must be untouched ------------------------------------

def test_cross_residue_force_is_unaffected_by_the_mask():
    """residueA != residueB shares no molecule, so the mask changes nothing."""
    system, topology, positions = _build(
        [("BEN", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.30)]),
         ("PRP", [(1.42, 1.0, 1.0), (1.42, 1.0, 1.30)])])
    n_before = _add(system, topology, {"BP": {"residueA": "BEN", "residueB": "PRP"}})

    cross = [(i, j) for i in range(2) for j in range(2, 4)]
    assert _energy(system, positions, n_before) == pytest.approx(
        _pair_energy(positions, cross), rel=1e-5)


def test_exclusions_match_the_nonbonded_force():
    """The CPU platform rejects a system whose forces disagree on exclusions."""
    system, topology, positions = _build(
        [("BEN", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.15), (1.0, 1.0, 1.30)]),
         ("BEN", [(1.45, 1.0, 1.0), (1.45, 1.0, 1.15), (1.45, 1.0, 1.30)])],
        bonds=True)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    assert nb.getNumExceptions() > 0

    n_before = _add(system, topology, {"BB": {"residueA": "BEN", "residueB": "BEN"}})
    repulsive = system.getForce(n_before)

    def pairs(n, get):
        return {tuple(sorted(get(k)[:2])) for k in range(n)}

    assert pairs(repulsive.getNumExclusions(), repulsive.getExclusionParticles) == \
        pairs(nb.getNumExceptions(), nb.getExceptionParameters)
    _energy(system, positions, n_before)  # must not raise on the CPU platform


def test_missing_parameters_fall_back_to_the_documented_defaults():
    """params without epsilon/sigma used to raise TypeError on None ** 2."""
    mols = [("BEN", [(1.0, 1.0, 1.0)]), ("BEN", [(1.4, 1.0, 1.0)])]
    bare, topology_a, positions = _build(mols)
    n_bare = _add(bare, topology_a, {"BB": {"residueA": "BEN", "residueB": "BEN"}})

    explicit, topology_b, _ = _build(mols)
    n_explicit = _add(explicit, topology_b, {"BB": {
        "residueA": "BEN", "residueB": "BEN",
        "epsilon": DEFAULT_REPULSIVE_EPSILON, "sigma": DEFAULT_REPULSIVE_SIGMA}})

    assert _energy(bare, positions, n_bare) == pytest.approx(
        _energy(explicit, positions, n_explicit), rel=1e-9)


def test_negative_epsilon_is_taken_as_repulsive():
    """abs() keeps the sign convention the original np.sqrt(eps**2) implied."""
    mols = [("BEN", [(1.0, 1.0, 1.0)]), ("BEN", [(1.4, 1.0, 1.0)])]
    neg, topology_a, positions = _build(mols)
    n_neg = _add(neg, topology_a, {"BB": {
        "residueA": "BEN", "residueB": "BEN", "epsilon": -0.02, "sigma": 4.0}})

    pos, topology_b, _ = _build(mols)
    n_pos = _add(pos, topology_b, {"BB": {
        "residueA": "BEN", "residueB": "BEN", "epsilon": 0.02, "sigma": 4.0}})

    assert _energy(neg, positions, n_neg) == pytest.approx(
        _energy(pos, positions, n_pos), rel=1e-9)
    assert _energy(neg, positions, n_neg) > 0


def test_absent_residue_is_skipped_with_a_warning():
    system, topology, _ = _build([("BEN", [(1.0, 1.0, 1.0)])])
    logger = _Logger()
    n_before = _add(system, topology,
                    {"BX": {"residueA": "BEN", "residueB": "NOPE"}}, logger)

    assert system.getNumForces() == n_before
    assert any("not found" in w for w in logger.warnings)
