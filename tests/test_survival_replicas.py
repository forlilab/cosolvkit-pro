"""Survival probability must be averaged ACROSS replicas, never concatenated.

SP is a dynamical quantity: joining two independent replicas into one trajectory makes
the join look like a departure event (the probe present in the last frame of one replica
is absent from the first frame of the next, and the resids do not even refer to the same
molecule). With ``max_tau`` lag frames each join corrupts that many lag windows and biases
residence times downward. So replicas are run separately and their curves averaged.

These tests exercise the aggregation and plumbing with a stub SP so they need neither
``waterdynamics`` nor a trajectory.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from cosolvkit.analysis.multi_report import _build_cosolvent_universes_map
from cosolvkit.analysis.sites.properties import PocketPropertyCalculator


class _StubSP:
    """Stands in for waterdynamics.SurvivalProbability.

    Returns a decay whose rate depends on the universe it was handed, so a per-replica
    average is distinguishable from using one replica.
    """

    registry = {}

    def __init__(self, universe, select, verbose=False):
        self.universe = universe
        self.select = select

    def run(self, tau_max=5, residues=False, intermittency=2):
        rate = self.registry[id(self.universe)]
        self.tau_timeseries = list(range(tau_max + 1))
        self.sp_timeseries = [float(np.exp(-rate * t)) for t in self.tau_timeseries]


@pytest.fixture
def stub_sp(monkeypatch):
    """Install a fake ``waterdynamics`` module exposing _StubSP."""
    mod = types.ModuleType("waterdynamics")
    mod.SurvivalProbability = _StubSP
    monkeypatch.setitem(sys.modules, "waterdynamics", mod)
    _StubSP.registry = {}
    return _StubSP


def _calculator(tmp_path, universe):
    calc = PocketPropertyCalculator.__new__(PocketPropertyCalculator)
    calc.universe = universe
    calc.out_path = str(tmp_path)
    import logging
    calc.logger = logging.getLogger("test")
    return calc


class _FakeUniverse:
    pass


def test_single_universe_keeps_the_old_output_shape(tmp_path, stub_sp):
    u = _FakeUniverse()
    stub_sp.registry[id(u)] = 0.5
    calc = _calculator(tmp_path, u)
    calc.run_survival_probability(["BEN"], [[0.0, 0.0, 0.0]], max_tau=4)

    out = tmp_path / "survival_probability_BEN.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert set(df.columns) == {"Group", "Zone", "Time", "SP", "Cosolvent"}
    # No per-replica file when there is only one replica.
    assert not (tmp_path / "survival_probability_BEN_per_replica.csv").exists()


def test_multiple_replicas_average_the_curves(tmp_path, stub_sp):
    slow, fast = _FakeUniverse(), _FakeUniverse()
    stub_sp.registry[id(slow)] = 0.1     # long residence
    stub_sp.registry[id(fast)] = 1.0     # short residence
    calc = _calculator(tmp_path, slow)
    calc.run_survival_probability(["BEN"], [[0.0, 0.0, 0.0]], max_tau=4,
                                 universes=[slow, fast])

    df = pd.read_csv(tmp_path / "survival_probability_BEN.csv")
    assert {"SP", "SP_sd", "n_replicas"} <= set(df.columns)
    assert (df["n_replicas"] == 2).all()

    expected = [(np.exp(-0.1 * t) + np.exp(-1.0 * t)) / 2 for t in df["Time"]]
    np.testing.assert_allclose(df["SP"].values, expected, atol=1e-9)
    # The averaged curve must sit between the two replicas, not equal either.
    assert df.loc[df.Time == 4, "SP"].iloc[0] > np.exp(-1.0 * 4)
    assert df.loc[df.Time == 4, "SP"].iloc[0] < np.exp(-0.1 * 4)


def test_per_replica_curves_are_kept_for_every_zone(tmp_path, stub_sp):
    a, b = _FakeUniverse(), _FakeUniverse()
    stub_sp.registry[id(a)] = 0.2
    stub_sp.registry[id(b)] = 0.8
    calc = _calculator(tmp_path, a)
    calc.run_survival_probability(["BEN"], [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
                                 max_tau=3, universes=[a, b])

    raw = pd.read_csv(tmp_path / "survival_probability_BEN_per_replica.csv")
    assert sorted(raw["Replica"].unique().tolist()) == [0, 1]
    assert sorted(raw["Group"].unique().tolist()) == [0, 1]
    # Every replica contributes a full curve in every zone, so the spread is recoverable.
    assert len(raw) == 2 * 2 * 4          # zones x replicas x timepoints


def test_mean_of_replica_mrts_equals_mrt_of_mean_curve(tmp_path, stub_sp):
    """sp_mrt is a trapezoid integral, i.e. LINEAR in the curve.

    So averaging curves and averaging per-replica MRTs agree exactly — which is why the
    per-replica spread is free. (The exponential FITS are nonlinear and must be fitted to
    the averaged curve instead.)
    """
    a, b = _FakeUniverse(), _FakeUniverse()
    stub_sp.registry[id(a)] = 0.3
    stub_sp.registry[id(b)] = 0.9
    calc = _calculator(tmp_path, a)
    calc.run_survival_probability(["BEN"], [[0.0, 0.0, 0.0]], max_tau=6,
                                 universes=[a, b])

    raw = pd.read_csv(tmp_path / "survival_probability_BEN_per_replica.csv")
    mean_curve = pd.read_csv(tmp_path / "survival_probability_BEN.csv")

    per_rep = [np.trapz(g["SP"].values, g["Time"].values)
               for _, g in raw.groupby("Replica")]
    mrt_of_mean = np.trapz(mean_curve["SP"].values, mean_curve["Time"].values)
    assert np.mean(per_rep) == pytest.approx(mrt_of_mean, abs=1e-9)


# ---------------------------------------------------------------------------
# multi_report plumbing
# ---------------------------------------------------------------------------

class _Sim:
    def __init__(self, cosolvents):
        self.cosolvents = cosolvents


class _Rep:
    def __init__(self, universe):
        self.universe = universe
        self.out_path = "/tmp"


def test_universes_map_collects_every_replica():
    u1, u2, u3 = _FakeUniverse(), _FakeUniverse(), _FakeUniverse()
    sims = [_Sim(["BEN"]), _Sim(["BEN"]), _Sim(["ACT"])]
    reps = [_Rep(u1), _Rep(u2), _Rep(u3)]
    m = _build_cosolvent_universes_map(sims, reps)
    assert m["BEN"] == [u1, u2]      # both replicas, not just the first
    assert m["ACT"] == [u3]


def test_universes_map_handles_multi_cosolvent_simulations():
    u1 = _FakeUniverse()
    m = _build_cosolvent_universes_map([_Sim(["BEN", "ACT"])], [_Rep(u1)])
    assert m == {"BEN": [u1], "ACT": [u1]}


# ---------------------------------------------------------------------------
# Adaptive (probe-scaled) zone radius
# ---------------------------------------------------------------------------

def _diatomic_universe(sep=1.4, resname="MOH"):
    """Two carbons `sep` apart in one residue."""
    import MDAnalysis as mda
    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0],
                           residue_segindex=[0], trajectory=True)
    u.add_TopologyAttr("name", ["C1", "C2"])
    u.add_TopologyAttr("type", ["C", "C"])
    u.add_TopologyAttr("resname", [resname])
    u.add_TopologyAttr("resid", [1])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
    return u


def test_adaptive_radius_is_rg_plus_buffer():
    u = _diatomic_universe(sep=1.4)
    # Rg of two points 1.4 apart about their midpoint = 0.7, plus the buffer.
    r = PocketPropertyCalculator.probe_zone_radius(u, "MOH", tolerance=1.7)
    assert r == pytest.approx(0.7 + 1.7, abs=1e-6)


def test_adaptive_radius_uses_unweighted_rg():
    """Rg must not depend on atomic masses: the zone is a geometric criterion."""
    import MDAnalysis as mda
    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0],
                           residue_segindex=[0], trajectory=True)
    u.add_TopologyAttr("name", ["C1", "O1"])
    u.add_TopologyAttr("type", ["C", "O"])       # unequal masses
    u.add_TopologyAttr("resname", ["MOH"])
    u.add_TopologyAttr("resid", [1])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    # Unweighted Rg is exactly half the separation regardless of the mass difference.
    assert PocketPropertyCalculator.probe_zone_radius(
        u, "MOH", tolerance=0.0) == pytest.approx(0.7, abs=1e-6)


def test_adaptive_radius_none_for_absent_cosolvent():
    assert PocketPropertyCalculator.probe_zone_radius(
        _diatomic_universe(), "NOPE") is None


def test_run_sp_resolves_adaptive_radius(tmp_path, stub_sp):
    """radius='adaptive' must size the zone from the probe, not raise."""
    u = _diatomic_universe(sep=1.4)
    stub_sp.registry[id(u)] = 0.5
    calc = _calculator(tmp_path, u)
    calc.run_survival_probability(["MOH"], [[0.0, 0.0, 0.0]], radius="adaptive",
                                 max_tau=3, radius_tolerance=1.7, universes=[u])
    df = pd.read_csv(tmp_path / "survival_probability_MOH.csv")
    assert len(df) == 4


def test_run_sp_rejects_a_bad_radius_string(tmp_path, stub_sp):
    u = _diatomic_universe()
    stub_sp.registry[id(u)] = 0.5
    calc = _calculator(tmp_path, u)
    with pytest.raises(ValueError, match="must be a number or 'adaptive'"):
        calc.run_survival_probability(["MOH"], [[0.0, 0.0, 0.0]], radius="big",
                                     max_tau=3, universes=[u])


def test_run_sp_adaptive_raises_when_probe_missing(tmp_path, stub_sp):
    u = _diatomic_universe(resname="MOH")
    stub_sp.registry[id(u)] = 0.5
    calc = _calculator(tmp_path, u)
    with pytest.raises(ValueError, match="not found in the trajectory topology"):
        calc.run_survival_probability(["ZZZ"], [[0.0, 0.0, 0.0]], radius="adaptive",
                                     max_tau=3, universes=[u])
