"""scale_interactions must scale a group pair's LJ attraction by lambda and
nothing else, leaving the repulsive core at full strength.

It adds (1 - lambda) * attraction on top of the untouched NonbondedForce, so
lambda = 1 has to be exactly zero, not approximately.
"""

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest

from cosolvkit.cosolvent_system import CosolventSystem

BOX = 4.0  # nm
CUTOFF = 1.0
SWITCH = 0.9
SIGMA = 0.32          # nm, per particle
EPSILON = 0.5         # kJ/mol, per particle
LJ_GROUP = 19


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a):
        self.warnings.append(msg % a if a else msg)

    def info(self, msg, *a):
        self.infos.append(msg % a if a else msg)


class _Stub(CosolventSystem):
    """Only the attributes the group-force methods touch; no system building."""

    def __init__(self, system, topology, logger):  # deliberately not calling super()
        self.system = system
        self.modeller = app.Modeller(topology, [mm.Vec3(0, 0, 0)] * topology.getNumAtoms())
        self.logger = logger


def _build(molecules, dispersion_correction=True):
    """Bare system of `molecules`, a list of (resname, [positions in nm])."""
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(CUTOFF * u.nanometer)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(SWITCH * u.nanometer)
    nb.setUseDispersionCorrection(dispersion_correction)

    positions = []
    for resname, coords in molecules:
        residue = topology.addResidue(resname, chain)
        for k in range(len(coords)):
            topology.addAtom(f"C{k}", app.element.carbon, residue)
        for xyz in coords:
            system.addParticle(12.0 * u.amu)
            nb.addParticle(0.0, SIGMA * u.nanometer, EPSILON * u.kilojoule_per_mole)
            positions.append(mm.Vec3(*xyz) * u.nanometer)

    nb.setForceGroup(LJ_GROUP)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0) * u.nanometer,
                                        mm.Vec3(0, BOX, 0) * u.nanometer,
                                        mm.Vec3(0, 0, BOX) * u.nanometer)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    return system, topology, positions


def _context(system, positions):
    ctx = mm.Context(system, mm.VerletIntegrator(1.0 * u.femtosecond),
                     mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions)
    ctx.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    return ctx


def _energy(ctx, groups=None):
    state = (ctx.getState(getEnergy=True) if groups is None
             else ctx.getState(getEnergy=True, groups=groups))
    return state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


def _scale(system, topology, spec, logger=None, force_group=31):
    stub = _Stub(system, topology, logger or _Logger())
    n_before = system.getNumForces()
    stub.scale_interactions(spec)
    for i in range(n_before, system.getNumForces()):
        system.getForce(i).setForceGroup(force_group)
    return n_before


def _reference_terms(positions, pairs):
    """(repulsion, attraction) summed over `pairs`, switched as OpenMM does.

    Both are returned positive: LJ = repulsion - attraction.
    """
    rep = att = 0.0
    for i, j in pairs:
        r = np.linalg.norm(np.array(positions[i].value_in_unit(u.nanometer))
                           - np.array(positions[j].value_in_unit(u.nanometer)))
        if r >= CUTOFF:
            continue
        sr = (SIGMA / r) ** 6
        switch = 1.0
        if r > SWITCH:
            x = (r - SWITCH) / (CUTOFF - SWITCH)
            switch = 1 - 6 * x**5 + 15 * x**4 - 10 * x**3
        rep += 4 * EPSILON * sr * sr * switch
        att += 4 * EPSILON * sr * switch
    return rep, att


def _two_benzene_like():
    """Two 3-atom BEN molecules 0.42 nm apart, plus one 2-atom PRP further out."""
    mol_a = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.35), (1.0, 1.0, 1.70)]
    mol_b = [(1.42, 1.0, 1.0), (1.42, 1.0, 1.35), (1.42, 1.0, 1.70)]
    prp = [(1.0, 1.5, 1.0), (1.0, 1.5, 1.35)]
    return [("BEN", mol_a), ("BEN", mol_b), ("PRP", prp)]


BEN_CROSS = [(i, j) for i in range(3) for j in range(3, 6)]
BEN_INTRA = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]


# --- the contract ---------------------------------------------------------

def test_lambda_one_is_an_exact_no_op():
    """Not approximately zero: the (lambda-1) prefactor must vanish identically."""
    system, topology, positions = _build(_two_benzene_like())
    baseline = _energy(_context(system, positions))

    system, topology, positions = _build(_two_benzene_like())
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 1.0}})
    ctx = _context(system, positions)

    assert _energy(ctx, groups={31}) == 0.0
    assert _energy(ctx) == pytest.approx(baseline, rel=1e-9)


def test_lambda_zero_removes_exactly_the_group_attraction():
    """The correction must equal the group's own attraction, checked against numpy."""
    system, topology, positions = _build(_two_benzene_like(),
                                         dispersion_correction=False)
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.0}})
    ctx = _context(system, positions)

    _, attraction = _reference_terms(positions, BEN_CROSS)
    assert attraction > 0.1, "test geometry must give a measurable attraction"
    assert _energy(ctx, groups={31}) == pytest.approx(attraction, rel=1e-5)


def test_scaling_is_linear_in_lambda():
    system, topology, positions = _build(_two_benzene_like(),
                                         dispersion_correction=False)
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 1.0}})
    ctx = _context(system, positions)
    repulsion, attraction = _reference_terms(positions, BEN_CROSS)

    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        ctx.setParameter("lambda_BEN_BEN", lam)
        scaled = repulsion - attraction + _energy(ctx, groups={31})
        # abs floor: the CPU platform holds positions in single precision.
        assert scaled == pytest.approx(repulsion - lam * attraction,
                                       rel=1e-5, abs=1e-5)


def test_other_groups_are_untouched():
    """Scaling BEN-BEN must not change BEN-PRP or PRP's internal energy."""
    system, topology, positions = _build(_two_benzene_like(),
                                         dispersion_correction=False)
    full = _energy(_context(system, positions), groups={LJ_GROUP})

    system, topology, positions = _build(_two_benzene_like(),
                                         dispersion_correction=False)
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.0}})
    ctx = _context(system, positions)

    # only the BEN-BEN cross attraction should have disappeared
    _, attraction = _reference_terms(positions, BEN_CROSS)
    assert _energy(ctx, groups={LJ_GROUP, 31}) == pytest.approx(
        full + attraction, rel=1e-5)


def test_a_molecule_is_not_scaled_against_itself():
    """Intramolecular pairs outside the exception list must be masked out."""
    system, topology, positions = _build(
        [("BEN", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.35), (1.0, 1.0, 1.70)])],
        dispersion_correction=False)
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.0}})
    ctx = _context(system, positions)

    assert _reference_terms(positions, [(0, 1), (0, 2), (1, 2)])[1] > 0.1
    assert _energy(ctx, groups={31}) == pytest.approx(0.0, abs=1e-9)


def test_repulsive_core_survives_lambda_zero():
    """Two molecules pushed inside contact stay strongly repulsive at lambda = 0.

    Full LJ scaling would remove the core too and let the pair collapse; here
    only the attraction goes, so the pair energy is the untouched repulsion.
    """
    close = [("BEN", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.35), (1.0, 1.0, 1.70)]),
             ("BEN", [(1.22, 1.0, 1.0), (1.22, 1.0, 1.35), (1.22, 1.0, 1.70)])]
    system, topology, positions = _build(close, dispersion_correction=False)
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.0}})
    ctx = _context(system, positions)

    repulsion, _ = _reference_terms(positions, BEN_CROSS)
    # the bare test topology has no bonds, so NonbondedForce also counts the
    # intramolecular pairs; remove them to isolate the pair energy
    intra_rep, intra_att = _reference_terms(positions, BEN_INTRA)
    pair_energy = _energy(ctx, groups={LJ_GROUP, 31}) - (intra_rep - intra_att)
    assert repulsion > 10.0, "test geometry must sit inside the repulsive core"
    assert pair_energy == pytest.approx(repulsion, rel=1e-5)


def test_exclusions_match_the_nonbonded_force():
    """The CPU platform rejects a system whose forces disagree on exclusions."""
    system, topology, positions = _build(_two_benzene_like())
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    n_before = _scale(system, topology,
                      {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.5}})
    scaling = system.getForce(n_before)

    assert scaling.getNumExclusions() == nb.getNumExceptions()
    _context(system, positions)  # must not raise


def test_cutoff_and_switching_are_copied_from_the_nonbonded_force():
    system, topology, _ = _build(_two_benzene_like())
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    n_before = _scale(system, topology,
                      {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.5}})
    scaling = system.getForce(n_before)

    assert scaling.getCutoffDistance() == nb.getCutoffDistance()
    assert scaling.getUseSwitchingFunction() == nb.getUseSwitchingFunction()
    assert scaling.getSwitchingDistance() == nb.getSwitchingDistance()


@pytest.mark.parametrize("enabled", [True, False])
def test_dispersion_correction_follows_the_nonbonded_force(enabled):
    """It is interaction-group aware, so leaving it off would bias the tail."""
    system, topology, _ = _build(_two_benzene_like(), dispersion_correction=enabled)
    n_before = _scale(system, topology,
                      {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.5}})
    assert system.getForce(n_before).getUseLongRangeCorrection() is enabled


def test_lambda_survives_system_serialization():
    """The value must round-trip through system.xml into equilibration/production."""
    system, topology, positions = _build(_two_benzene_like())
    _scale(system, topology,
           {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 0.35}})
    reloaded = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))

    ctx = _context(reloaded, positions)
    assert ctx.getParameter("lambda_BEN_BEN") == pytest.approx(0.35)


# --- validation -----------------------------------------------------------

def test_missing_lambda_is_an_error():
    system, topology, _ = _build(_two_benzene_like())
    with pytest.raises(ValueError, match="no 'lambda'"):
        _scale(system, topology, {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN"}})


def test_negative_lambda_is_an_error():
    system, topology, _ = _build(_two_benzene_like())
    with pytest.raises(ValueError, match="negative factor"):
        _scale(system, topology,
               {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": -0.5}})


def test_lambda_above_one_warns_but_is_allowed():
    system, topology, _ = _build(_two_benzene_like())
    logger = _Logger()
    n_before = _scale(system, topology,
                      {"BEN_BEN": {"residueA": "BEN", "residueB": "BEN", "lambda": 1.5}},
                      logger)
    assert system.getNumForces() == n_before + 1
    assert any("strengthens" in w for w in logger.warnings)


def test_absent_residue_is_skipped_with_a_warning():
    system, topology, _ = _build(_two_benzene_like())
    logger = _Logger()
    n_before = _scale(system, topology,
                      {"BEN_X": {"residueA": "BEN", "residueB": "NOPE", "lambda": 0.5}},
                      logger)
    assert system.getNumForces() == n_before
    assert any("not found" in w for w in logger.warnings)


# --- config / CLI wiring --------------------------------------------------

def test_config_accepts_interaction_scaling(tmp_path):
    from cosolvkit.config import Config

    assert "interaction_scaling" in Config.get_defaults_dict()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "cosolvents: []\n"
        "md_engine:\n  openmm:\n    - amber14-all.xml\n"
        "box_size: 30.0\n"
        "solvent_smiles: 'H2O'\n"
        "output_dir: 'out'\n"
        "interaction_scaling:\n"
        "  BEN_BEN:\n    residueA: 'BEN'\n    residueB: 'BEN'\n    lambda: 0.8\n")
    parsed = Config.from_config(str(cfg))
    assert parsed.interaction_scaling["BEN_BEN"]["lambda"] == 0.8


def test_config_defaults_to_no_scaling():
    from cosolvkit.config import Config

    assert Config.get_defaults_dict()["interaction_scaling"] == {}
