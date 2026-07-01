import numpy as np
from cosolvkit.analysis.core.models import BindingSite, Hotspot


def _hotspot(cosolvent, site_id, agfe_min, atomtypes):
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    return Hotspot(
        rank=1, site_id=site_id, cosolvent=cosolvent, n_voxels=int(mask.sum()),
        centroid=np.array([1.0, 1.0, 1.0]), agfe_min=agfe_min, agfe_mean_top_pct=agfe_min + 0.5,
        voxel_mask=mask, favorable_atomtypes=atomtypes, per_type_agfe={a: agfe_min for a in atomtypes},
    )


def test_binding_site_holds_members_and_features():
    h1 = _hotspot("BEN", 1, -3.0, ["Car"])
    h2 = _hotspot("IMI", 2, -2.0, ["HBA"])
    union = h1.voxel_mask | h2.voxel_mask
    bs = BindingSite(
        site_id=1, member_hotspots=[h1, h2], voxel_mask=union,
        centroid=np.array([1.0, 1.0, 1.0]), agfe_min=-3.0, agfe_mean_top_pct=-2.5,
        volume=12.0, solidity=0.9, extent=0.5, axis_major_length=3.0, axis_minor_length=2.0,
        favorable_atomtypes=["Car", "HBA"], pharmacophore={"BEN": {"Car": -3.0}, "IMI": {"HBA": -2.0}},
        residence=12.5, cosolvents=["BEN", "IMI"], n_total_cosolvents=2,
    )
    assert bs.n_hotspots == 2
    assert bs.n_cosolvents == 2
    assert bs.probe_coverage == 1.0
    assert bs.member_hotspot_ids == [1, 2]
    d = bs.to_dict()
    assert d["site_id"] == 1 and d["n_cosolvents"] == 2 and d["volume"] == 12.0
    assert d["cosolvents"] == "BEN,IMI"
