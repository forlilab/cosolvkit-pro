import numpy as np
import pandas as pd
import pytest
from cosolvkit.analysis.hotspot_dashboard import (
    _load_binding_sites_csv, rerank_binding_sites, DEFAULT_DASHBOARD_WEIGHTS,
)


def _bs_df():
    # Two binding sites. A is best on every feature; B worst.
    return pd.DataFrame([
        dict(site_id=1, rank=2, combined=0.0, cosolvents="BEN,IMI", n_cosolvents=2,
             probe_coverage=1.0, n_hotspots=2, member_hotspot_ids="1,2",
             centroid_x=0.0, centroid_y=0.0, centroid_z=0.0, agfe_min=-3.0,
             agfe_mean_top_pct=-2.5, volume=100.0, solidity=0.9, extent=0.5,
             axis_major_length=3.0, axis_minor_length=2.0, favorable_atomtypes="Car,HBA",
             n_chemotypes=2, residence=20.0),
        dict(site_id=2, rank=1, combined=0.0, cosolvents="BEN", n_cosolvents=1,
             probe_coverage=0.5, n_hotspots=1, member_hotspot_ids="3",
             centroid_x=5.0, centroid_y=5.0, centroid_z=5.0, agfe_min=-1.0,
             agfe_mean_top_pct=-0.8, volume=50.0, solidity=0.6, extent=0.4,
             axis_major_length=2.0, axis_minor_length=1.0, favorable_atomtypes="Car",
             n_chemotypes=1, residence=10.0),
    ])


def test_default_weights_constant():
    assert DEFAULT_DASHBOARD_WEIGHTS == {
        "affinity": 3.0, "probe_coverage": 2.0, "volume": 1.0,
        "kinetics": 1.0, "shape": 1.0, "diversity": 1.0,
    }


def test_rerank_default_weights_ranks_best_site_first():
    out = rerank_binding_sites(_bs_df(), DEFAULT_DASHBOARD_WEIGHTS)
    # site 1 is best on all features -> rank 1, combined 9.0; site 2 -> rank 2, combined 0.0
    r = out.set_index("site_id")
    assert r.loc[1, "rank"] == 1
    assert r.loc[2, "rank"] == 2
    assert r.loc[1, "combined"] == pytest.approx(9.0, abs=1e-9)
    assert r.loc[2, "combined"] == pytest.approx(0.0, abs=1e-9)
    # output is sorted by rank ascending
    assert list(out["rank"]) == [1, 2]


def test_rerank_negative_volume_weight_flips_toward_smaller():
    w = {"affinity": 0, "probe_coverage": 0, "kinetics": 0, "shape": 0, "diversity": 0, "volume": -1}
    out = rerank_binding_sites(_bs_df(), w).set_index("site_id")
    # smaller-volume site (2) preferred
    assert out.loc[2, "rank"] == 1 and out.loc[1, "rank"] == 2


def test_rerank_handles_blank_residence():
    df = _bs_df()
    df.loc[df["site_id"] == 1, "residence"] = np.nan  # missing kinetics
    w = {"affinity": 0, "probe_coverage": 0, "volume": 0, "shape": 0, "diversity": 0, "kinetics": 1}
    out = rerank_binding_sites(df, w).set_index("site_id")
    # site 2 has the only finite residence -> flat minmax -> 1.0; site 1 (NaN) -> 0.0
    assert out.loc[2, "rank"] == 1


def test_load_missing_returns_empty(tmp_path):
    assert _load_binding_sites_csv(str(tmp_path)).empty


def test_load_reads_binding_sites_csv(tmp_path):
    _bs_df().to_csv(tmp_path / "binding_sites.csv", index=False)
    df = _load_binding_sites_csv(str(tmp_path))
    assert len(df) == 2 and "combined" in df.columns


def _write_example(tmp_path):
    _bs_df().to_csv(tmp_path / "binding_sites.csv", index=False)


def test_dashboard_constructs_on_binding_sites(tmp_path):
    _write_example(tmp_path)
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    dash_ok = pytest.importorskip("dash")  # skip if dash missing
    dboard = HotspotDashboard(out_path=str(tmp_path), pdb_path=None, port=8055)
    layout_str = str(dboard._app.layout)
    # six weight sliders present; cosolvent dropdown gone
    for wid in ("weight-affinity", "weight-probe_coverage", "weight-volume",
                "weight-kinetics", "weight-shape", "weight-diversity"):
        assert wid in layout_str
    assert "cosolvent-dd" not in layout_str
    assert "score-slider" not in layout_str
    assert "Binding Sites" in layout_str          # tab renamed
    # the class holds the binding-sites df
    assert not dboard._bs_df.empty and "combined" in dboard._bs_df.columns
