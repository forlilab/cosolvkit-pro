"""Hotspot/BindingSite must work as plain site records with no AGFE data.

This is what lets the same models (and the same mask-connectivity grouping) describe
sites that did not come from a density map -- e.g. crystallographic ligand copies used
as benchmark ground truth.
"""

import numpy as np

from cosolvkit.analysis.core.models import BindingSite, Hotspot
from cosolvkit.analysis.core.scoring import score_binding_sites
from cosolvkit.analysis.sites.binding_sites import (
    build_binding_site,
    group_hotspots,
)


def _bare_hs(comp_id, site_id, blob, centroid, shape=(20, 20, 20)):
    """A hotspot carrying only species, mask and centroid -- no AGFE at all."""
    mask = np.zeros(shape, dtype=bool)
    mask[blob] = True
    h = Hotspot(rank=None, site_id=site_id, cosolvent=comp_id,
                n_voxels=int(mask.sum()), centroid=np.asarray(centroid, dtype=float),
                voxel_mask=mask)
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    return h


# ---------------------------------------------------------------------------
# Hotspot
# ---------------------------------------------------------------------------

def test_hotspot_constructs_without_agfe():
    h = Hotspot(rank=None, site_id=1, cosolvent="VOB")
    assert h.agfe_min is None
    assert h.agfe_mean_top_pct is None
    assert h.voxel_mask is None
    assert h.centroid is None
    assert h.favorable_atomtypes == []
    assert h.per_type_agfe == {}


def test_hotspot_to_dict_and_repr_tolerate_missing_agfe():
    h = Hotspot(rank=None, site_id=1, cosolvent="VOB",
                centroid=np.array([1.234567, 2.0, 3.0]))
    d = h.to_dict()
    assert d["agfe_min"] is None
    assert d["agfe_mean_top_pct"] is None
    assert d["centroid_x"] == 1.235
    assert d["cosolvent"] == "VOB"
    assert "n/a" in repr(h)


def test_hotspot_to_dict_drops_non_finite_agfe():
    h = Hotspot(rank=1, site_id=1, cosolvent="BEN", centroid=np.zeros(3),
                agfe_min=float("nan"), agfe_mean_top_pct=float("-inf"))
    d = h.to_dict()
    assert d["agfe_min"] is None
    assert d["agfe_mean_top_pct"] is None


def test_hotspot_mutable_defaults_are_not_shared():
    a = Hotspot(rank=None, site_id=1, cosolvent="A")
    b = Hotspot(rank=None, site_id=2, cosolvent="B")
    a.favorable_atomtypes.append("Car")
    a.per_type_agfe["Car"] = -1.0
    assert b.favorable_atomtypes == []
    assert b.per_type_agfe == {}


# ---------------------------------------------------------------------------
# BindingSite
# ---------------------------------------------------------------------------

def test_binding_site_derives_cosolvents_from_members():
    members = [Hotspot(rank=None, site_id=1, cosolvent="VOB"),
               Hotspot(rank=None, site_id=2, cosolvent="TJ1"),
               Hotspot(rank=None, site_id=3, cosolvent="VOB")]
    bs = BindingSite(site_id=1, member_hotspots=members)
    assert bs.cosolvents == ["TJ1", "VOB"]
    assert bs.n_cosolvents == 2
    assert bs.n_total_cosolvents == 2
    assert bs.probe_coverage == 1.0
    assert bs.n_hotspots == 3


def test_binding_site_explicit_n_total_wins_over_derived():
    members = [Hotspot(rank=None, site_id=1, cosolvent="VOB")]
    bs = BindingSite(site_id=1, member_hotspots=members, n_total_cosolvents=10)
    assert bs.n_total_cosolvents == 10
    assert bs.probe_coverage == 0.1


def test_binding_site_to_dict_tolerates_missing_aggregates():
    bs = BindingSite(site_id=7,
                     member_hotspots=[Hotspot(rank=None, site_id=1, cosolvent="VOB")],
                     centroid=np.array([10.0, 20.0, 30.0]))
    d = bs.to_dict()
    assert d["site_id"] == 7
    assert d["cosolvents"] == "VOB"
    assert d["centroid_x"] == 10.0
    for key in ("agfe_min", "volume", "solidity", "extent",
                "axis_major_length", "axis_minor_length", "residence", "combined"):
        assert d[key] is None, key
    assert "n/a" in repr(bs)


def test_binding_site_empty_members_is_valid():
    bs = BindingSite(site_id=1)
    assert bs.member_hotspots == []
    assert bs.cosolvents == []
    assert bs.n_hotspots == 0
    assert bs.probe_coverage == 0.0


# ---------------------------------------------------------------------------
# Grouping + aggregation with no AGFE
# ---------------------------------------------------------------------------

def test_group_and_build_site_from_agfe_free_hotspots():
    # Two overlapping "ligand copies" of different species -> one ground-truth site.
    a = _bare_hs("VOB", 1, np.s_[5:9, 5:9, 5:9], centroid=[1.0, 0.0, 0.0])
    b = _bare_hs("TJ1", 2, np.s_[7:11, 7:11, 7:11], centroid=[3.0, 0.0, 0.0])
    groups = group_hotspots({"VOB": [a], "TJ1": [b]})
    assert len(groups) == 1

    site = build_binding_site(site_id=1, group=groups[0], n_total_cosolvents=2)
    assert site.agfe_min is None
    assert site.agfe_mean_top_pct is None
    assert site.cosolvents == ["TJ1", "VOB"]
    # No AGFE to weight by -> plain mean of member centroids.
    np.testing.assert_allclose(site.centroid, [2.0, 0.0, 0.0])
    assert site.volume > 0.0
    # Serialisable despite the missing affinity.
    assert site.to_dict()["agfe_min"] is None


def test_partial_agfe_falls_back_to_unweighted_centroid():
    a = _bare_hs("VOB", 1, np.s_[5:9, 5:9, 5:9], centroid=[0.0, 0.0, 0.0])
    b = _bare_hs("TJ1", 2, np.s_[7:11, 7:11, 7:11], centroid=[4.0, 0.0, 0.0])
    a.agfe_min = -5.0          # only one member has affinity
    a.agfe_mean_top_pct = -5.0
    groups = group_hotspots({"VOB": [a], "TJ1": [b]})
    site = build_binding_site(site_id=1, group=groups[0], n_total_cosolvents=2)
    # min() over the finite values only
    assert site.agfe_min == -5.0
    # Mixed availability must NOT weight by |agfe| (that would bias to member a).
    np.testing.assert_allclose(site.centroid, [2.0, 0.0, 0.0])


def test_score_binding_sites_handles_agfe_free_sites():
    a = _bare_hs("VOB", 1, np.s_[2:5, 2:5, 2:5], centroid=[0.0, 0.0, 0.0])
    b = _bare_hs("TJ1", 2, np.s_[14:18, 14:18, 14:18], centroid=[9.0, 9.0, 9.0])
    groups = group_hotspots({"VOB": [a], "TJ1": [b]})
    sites = [build_binding_site(site_id=i + 1, group=g, n_total_cosolvents=2)
             for i, g in enumerate(groups)]
    score_binding_sites(sites)
    assert all(s.combined is not None for s in sites)
    assert sorted(s.rank for s in sites) == [1, 2]
