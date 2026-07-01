import numpy as np
from cosolvkit.analysis.core.models import Hotspot
from cosolvkit.analysis.sites.binding_sites import group_hotspots, build_binding_site


def _hs(cosolvent, site_id, blob, agfe_min, atoms, sp_mrt=None, shape=(20, 20, 20)):
    mask = np.zeros(shape, dtype=bool); mask[blob] = True
    h = Hotspot(rank=1, site_id=site_id, cosolvent=cosolvent, n_voxels=int(mask.sum()),
                centroid=np.array([float(blob[0].start), 0.0, 0.0]),
                agfe_min=agfe_min, agfe_mean_top_pct=agfe_min + 0.5, voxel_mask=mask,
                favorable_atomtypes=list(atoms), per_type_agfe={a: agfe_min for a in atoms})
    h.grid_origin = np.array([0.0, 0.0, 0.0]); h.grid_delta = np.array([0.5, 0.5, 0.5])
    if sp_mrt is not None:
        h.add_property("sp_mrt", sp_mrt)
    return h


def test_build_binding_site_aggregates_members():
    a = _hs("BEN", 1, np.s_[5:9, 5:9, 5:9], -3.0, ["Car"], sp_mrt=20.0)
    b = _hs("IMI", 2, np.s_[7:11, 7:11, 7:11], -2.0, ["HBA"], sp_mrt=10.0)
    group = group_hotspots({"BEN": [a], "IMI": [b]})[0]
    bs = build_binding_site(site_id=1, group=group, n_total_cosolvents=2)
    assert sorted(bs.cosolvents) == ["BEN", "IMI"]
    assert bs.n_hotspots == 2 and bs.n_cosolvents == 2 and bs.probe_coverage == 1.0
    assert bs.agfe_min == -3.0                       # best (most-negative) across members
    assert sorted(bs.favorable_atomtypes) == ["Car", "HBA"]
    assert bs.residence == 20.0                      # max sp_mrt across members
    assert bs.pharmacophore["BEN"]["Car"] == -3.0 and bs.pharmacophore["IMI"]["HBA"] == -2.0
    # union volume = |union| * 0.5^3; union of two 4^3 blobs overlapping in [7:9]^3
    assert bs.volume > 0.0
    assert 0.0 <= bs.solidity <= 1.0
