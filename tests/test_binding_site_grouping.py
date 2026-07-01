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


def test_merge_tolerance_merges_near_but_untouching_peaks():
    # Two same-grid blobs with a 2-voxel (1.0 Å) gap between their surfaces:
    #   A occupies x=5..7, B occupies x=9..11  -> nearest surfaces at 7 and 9 (gap = 1 voxel = 0.5 Å?).
    # Make the gap ~1.5 Å (3 voxels): A x=5..7, B x=11..13 -> surfaces at 7 and 11 -> gap 3 voxels = 1.5 Å.
    a = _hs("BEN", 1, np.s_[5:8, 5:8, 5:8])
    b = _hs("IMI", 2, np.s_[11:14, 5:8, 5:8])
    # touch-only (tol=0): stay separate
    assert len(group_hotspots({"BEN": [a], "IMI": [b]}, merge_tolerance_ang=0.0)) == 2
    # tol=2.0 Å (closes gaps <= 2.0 Å; 1.5 Å gap merges): one binding site
    merged = group_hotspots({"BEN": [a], "IMI": [b]}, merge_tolerance_ang=2.0)
    assert len(merged) == 1
    assert {h.site_id for h in merged[0]["members"]} == {1, 2}


def test_merge_tolerance_union_mask_is_undilated():
    # union_mask stored for the group must equal the OR of the ORIGINAL member masks,
    # NOT the dilated grouping mask.
    a = _hs("BEN", 1, np.s_[5:8, 5:8, 5:8])
    b = _hs("IMI", 2, np.s_[11:14, 5:8, 5:8])
    g = group_hotspots({"BEN": [a], "IMI": [b]}, merge_tolerance_ang=2.0)[0]
    expected = a.voxel_mask | b.voxel_mask
    assert g["union_mask"].sum() == int(expected.sum())   # 2 * 3^3 = 54, not dilated
