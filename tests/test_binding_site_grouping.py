import numpy as np
from cosolvkit.analysis.core.models import Hotspot
from cosolvkit.analysis.sites.binding_sites import group_hotspots


def _hs(cosolvent, site_id, blob, agfe_min=-2.0, shape=(20, 20, 20)):
    mask = np.zeros(shape, dtype=bool)
    mask[blob] = True
    h = Hotspot(rank=1, site_id=site_id, cosolvent=cosolvent, n_voxels=int(mask.sum()),
                centroid=np.zeros(3), agfe_min=agfe_min, agfe_mean_top_pct=agfe_min,
                voxel_mask=mask, favorable_atomtypes=["Car"], per_type_agfe={"Car": agfe_min})
    h.grid_origin = np.array([0.0, 0.0, 0.0]); h.grid_delta = np.array([0.5, 0.5, 0.5])
    return h


def test_overlapping_cross_cosolvent_hotspots_merge():
    a = _hs("BEN", 1, np.s_[5:9, 5:9, 5:9])
    b = _hs("IMI", 2, np.s_[7:11, 7:11, 7:11])  # overlaps a
    groups = group_hotspots({"BEN": [a], "IMI": [b]})
    assert len(groups) == 1
    assert {h.site_id for h in groups[0]["members"]} == {1, 2}


def test_separated_hotspots_stay_distinct():
    a = _hs("BEN", 1, np.s_[2:5, 2:5, 2:5])
    b = _hs("BEN", 2, np.s_[14:17, 14:17, 14:17])  # far apart, same cosolvent
    groups = group_hotspots({"BEN": [a, b]})
    assert len(groups) == 2


def test_same_cosolvent_touching_merge():
    # naphthalene case: two same-cosolvent blobs that touch -> one binding site
    a = _hs("BEN", 1, np.s_[5:8, 5:8, 5:8])
    b = _hs("BEN", 2, np.s_[8:11, 5:8, 5:8])  # face-adjacent to a
    groups = group_hotspots({"BEN": [a, b]})
    assert len(groups) == 1
    assert len(groups[0]["members"]) == 2
