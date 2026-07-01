import json
import numpy as np
import pandas as pd
from cosolvkit.analysis.core.models import Hotspot
from cosolvkit.analysis.sites.binding_sites import identify_binding_sites, export_binding_sites


def _hs(cosolvent, site_id, blob, agfe_min, atoms, shape=(20, 20, 20)):
    mask = np.zeros(shape, dtype=bool); mask[blob] = True
    h = Hotspot(rank=1, site_id=site_id, cosolvent=cosolvent, n_voxels=int(mask.sum()),
                centroid=np.zeros(3), agfe_min=agfe_min, agfe_mean_top_pct=agfe_min,
                voxel_mask=mask, favorable_atomtypes=list(atoms),
                per_type_agfe={a: agfe_min for a in atoms})
    h.grid_origin = np.array([0.0, 0.0, 0.0]); h.grid_delta = np.array([0.5, 0.5, 0.5])
    return h


def test_export_writes_csv_json_pharmacophore_and_dx(tmp_path):
    a = _hs("BEN", 1, np.s_[5:9, 5:9, 5:9], -3.0, ["Car"])
    b = _hs("BEN", 2, np.s_[15:18, 15:18, 15:18], -1.0, ["Car"])
    sites = identify_binding_sites({"BEN": [a, b]})
    export_binding_sites(sites, str(tmp_path))

    df = pd.read_csv(tmp_path / "binding_sites.csv")
    assert {"site_id", "rank", "combined", "cosolvents", "agfe_min", "volume"} <= set(df.columns)
    assert len(df) == 2
    assert (tmp_path / "binding_sites.json").exists()
    pharm = json.loads((tmp_path / "binding_sites_pharmacophore.json").read_text())
    assert isinstance(pharm, list) and "pharmacophore" in pharm[0]
    assert (tmp_path / "binding_site_labels.dx").exists()
