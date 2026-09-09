"""LIE wrapper: means over frames, the probe-exclude mask, and the record round trip."""

import os
import types

import numpy as np
import pandas as pd
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import Hotspot, LieResult, ProbeOccupancy
from cosolvkit.analysis.sites.lie import (
    DEFAULT_ION_EXCLUDE, compute_lie, probe_exclude_mask,
)


def _occ(resid=810, resname="BEN", frames=(0, 1, 2, 3)):
    return ProbeOccupancy(source_label="benzene_rep1", topology="/tmp/s.prmtop",
                          trajectory="/tmp/t.dcd", probe_resname=resname,
                          probe_resid=resid, probe_resindex=resid - 1,
                          frames=list(frames), n_frames_scanned=100, stride=1)


class _StubAnalyzer:
    """Records the kwargs compute_LIE was called with and returns a fixed frame table."""

    last_kwargs = None

    def __init__(self, eelec, vdw):
        self._df = pd.DataFrame({"EELEC": eelec, "VDW": vdw})
        self._df["Total"] = self._df.EELEC + self._df.VDW

    def compute_LIE(self, **kwargs):
        _StubAnalyzer.last_kwargs = kwargs
        return self._df


def _factory(eelec, vdw):
    return lambda top, trj, resname, outdir: _StubAnalyzer(eelec, vdw)


# ---------------------------------------------------------------------------
# averaging
# ---------------------------------------------------------------------------

def test_means_and_sem_are_computed_over_frames(tmp_path):
    res = compute_lie(_occ(), "/tmp/frames.dcd", str(tmp_path),
                      analyzer_factory=_factory([-1.0, -3.0], [-4.0, -6.0]))
    assert res.eelec == pytest.approx(-2.0)
    assert res.vdw == pytest.approx(-5.0)
    assert res.total == pytest.approx(-7.0)
    assert res.n_frames == 2
    # totals are -5 and -9: sd = 2.828, sem = 2.0
    assert res.std_dev == pytest.approx(np.std([-5.0, -9.0], ddof=1))
    assert res.std_err == pytest.approx(res.std_dev / np.sqrt(2))


def test_single_frame_gives_zero_spread_not_nan(tmp_path):
    res = compute_lie(_occ(), "/tmp/frames.dcd", str(tmp_path),
                      analyzer_factory=_factory([-1.0], [-2.0]))
    assert res.n_frames == 1
    assert res.std_dev == 0.0 and res.std_err == 0.0
    assert not np.isnan(res.total)


def test_empty_result_returns_none_rather_than_crashing(tmp_path):
    res = compute_lie(_occ(), "/tmp/frames.dcd", str(tmp_path),
                      analyzer_factory=_factory([], []))
    assert res is None


def test_ligand_mask_and_cutoff_are_passed_through(tmp_path):
    compute_lie(_occ(resid=289, resname="FMD"), "/tmp/f.dcd", str(tmp_path),
                cutoff=8.0, analyzer_factory=_factory([-1.0], [-1.0]))
    kw = _StubAnalyzer.last_kwargs
    assert kw["ligand_amber_selection"] == ":289"
    assert kw["cutoff"] == 8.0
    assert kw["use_residues"] is None, "the cutoff shell must be auto-selected"


def test_water_is_kept_in_the_environment_by_default(tmp_path):
    """LIE without solvent is not LIE; the default exclude is ions only."""
    compute_lie(_occ(), "/tmp/f.dcd", str(tmp_path),
                analyzer_factory=_factory([-1.0], [-1.0]))
    excl = _StubAnalyzer.last_kwargs["exclude_amber_selection"]
    for water in ("WAT", "HOH", "SOL"):
        assert water not in excl
    assert excl == DEFAULT_ION_EXCLUDE


# ---------------------------------------------------------------------------
# probe exclusion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")
def test_probe_exclude_mask_lists_every_copy_but_the_ligand():
    u = mda.Universe.empty(4, n_residues=4, n_segments=1, atom_resindex=[0, 1, 2, 3],
                           residue_segindex=[0] * 4, trajectory=True)
    u.add_TopologyAttr("name", ["C1"] * 4)
    u.add_TopologyAttr("resname", ["BEN", "BEN", "BEN", "ALA"])
    u.add_TopologyAttr("resid", [10, 11, 12, 13])

    mask = probe_exclude_mask(u, ["BEN"], keep_resid=11)
    assert mask.startswith(DEFAULT_ION_EXCLUDE)
    assert ",10," in mask + "," and mask.endswith("12")
    assert ",11" not in mask.replace(DEFAULT_ION_EXCLUDE, ""), "the ligand must survive"
    assert "13" not in mask, "a non-probe residue must not be excluded"


@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")
def test_probe_exclude_mask_is_the_base_when_there_are_no_other_copies():
    u = mda.Universe.empty(1, n_residues=1, n_segments=1, atom_resindex=[0],
                           residue_segindex=[0], trajectory=True)
    u.add_TopologyAttr("name", ["C1"])
    u.add_TopologyAttr("resname", ["BEN"])
    u.add_TopologyAttr("resid", [10])
    assert probe_exclude_mask(u, ["BEN"], keep_resid=10) == DEFAULT_ION_EXCLUDE


def test_exclude_probes_flag_is_recorded_on_the_result(tmp_path):
    res = compute_lie(_occ(), "/tmp/f.dcd", str(tmp_path),
                      exclude_mask=DEFAULT_ION_EXCLUDE + ",811,812",
                      analyzer_factory=_factory([-1.0], [-1.0]))
    assert res.exclude_probes is True
    res2 = compute_lie(_occ(), "/tmp/f.dcd", str(tmp_path),
                       analyzer_factory=_factory([-1.0], [-1.0]))
    assert res2.exclude_probes is False


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------

def test_lie_result_round_trips():
    r = LieResult(probe_resname="BEN", probe_resid=810, source_label="r0",
                  eelec=-2.21, vdw=-7.58, total=-9.79, std_dev=4.8, std_err=1.08,
                  n_frames=20, cutoff=6.0, n_env_residues=55)
    back = LieResult.from_dict(r.to_dict())
    assert back == r


def test_hotspot_record_carries_lie_and_row_stays_flat():
    h = Hotspot(rank=1, site_id=7, cosolvent="BEN", n_voxels=8,
                centroid=np.zeros(3), agfe_min=-2.0, agfe_mean_top_pct=-1.5,
                voxel_mask=np.zeros((4, 4, 4), dtype=bool))
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    h.lie = [LieResult(probe_resname="BEN", probe_resid=810, source_label="r0",
                       eelec=-2.2, vdw=-7.6, total=-9.8, std_dev=4.8, std_err=1.1,
                       n_frames=20)]
    rec = h.to_record()
    assert rec["lie"][0]["total"] == pytest.approx(-9.8)

    back = Hotspot.from_record(rec, h.voxel_mask, h.grid_origin, h.grid_delta)
    assert len(back.lie) == 1
    assert back.lie[0].probe_resid == 810

    row = h.to_row()
    assert row["lie_total"] == pytest.approx(-9.8)
    assert row["lie_n_results"] == 1
    bad = {k: v for k, v in row.items() if isinstance(v, (list, dict))}
    assert bad == {}


def test_row_lie_columns_are_none_without_results():
    h = Hotspot(rank=1, site_id=1, cosolvent="BEN")
    row = h.to_row()
    assert row["lie_total"] is None
    assert row["lie_n_results"] == 0


# ---------------------------------------------------------------------------
# Attribution + idempotency. Found by running the CLI twice: a pre-fix pass had
# attached an IMI result to a BEN hotspot, and re-running only purged the key from
# the hotspots it re-attached to, so the stale row survived.
# ---------------------------------------------------------------------------

def _hotspot_with_occ(site_id, cosolvent, occ):
    h = Hotspot(rank=1, site_id=site_id, cosolvent=cosolvent)
    h.probe_occupancy = [occ] if occ is not None else []
    return h


def test_lie_is_attached_only_to_the_hotspot_that_owns_the_molecule():
    """A BindingSite's probe_occupancy is the union over members, so attaching to every
    member puts a result on blobs the molecule never entered."""
    from cosolvkit.cli.refine_hotspots import lie_rows

    owner = _hotspot_with_occ(12, "IMI", _occ(resid=807, resname="IMI"))
    stranger = _hotspot_with_occ(15, "BEN", _occ(resid=810, resname="BEN"))
    result = LieResult(probe_resname="IMI", probe_resid=807,
                       source_label="benzene_rep1", eelec=-9.4, vdw=-9.8,
                       total=-19.2, std_dev=9.9, std_err=3.1, n_frames=10)
    owner.lie = [result]

    rows = lie_rows({"IMI": [owner], "BEN": [stranger]})
    assert len(rows) == 1
    assert rows[0]["site_id"] == 12
    assert not stranger.lie, "a BEN hotspot must not carry an IMI result"


def test_reattaching_purges_the_molecule_from_every_hotspot():
    """The purge must be global, or a stale mis-attachment outlives the fix."""
    key = ("benzene_rep1", "IMI", 807)
    stale = LieResult(probe_resname="IMI", probe_resid=807,
                      source_label="benzene_rep1", eelec=-1.0, vdw=-1.0, total=-2.0,
                      std_dev=0.0, std_err=0.0, n_frames=1)
    fresh = LieResult(probe_resname="IMI", probe_resid=807,
                      source_label="benzene_rep1", eelec=-9.4, vdw=-9.8, total=-19.2,
                      std_dev=9.9, std_err=3.1, n_frames=10)

    owner = _hotspot_with_occ(12, "IMI", _occ(resid=807, resname="IMI"))
    polluted = _hotspot_with_occ(15, "BEN", _occ(resid=810, resname="BEN"))
    polluted.lie = [stale]
    results = {"IMI": [owner], "BEN": [polluted]}

    # the global purge the runner performs before re-attaching
    for hs in results.values():
        for h in hs:
            h.lie = [r for r in h.lie
                     if (r.source_label, r.probe_resname, r.probe_resid) != key]
    owner.lie.append(fresh)

    assert polluted.lie == [], "the stale record must be gone"
    assert [r.total for r in owner.lie] == [-19.2]
    assert len(lie_rows_total(results)) == 1


def lie_rows_total(results):
    from cosolvkit.cli.refine_hotspots import lie_rows
    return lie_rows(results)
