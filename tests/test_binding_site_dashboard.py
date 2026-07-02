import os
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


def test_dashboard_has_expected_callbacks(tmp_path):
    _write_example(tmp_path)
    pytest.importorskip("dash")
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    d = HotspotDashboard(out_path=str(tmp_path), pdb_path=None, port=8056)
    # at least the figure + table outputs are wired
    layout = str(d._app.layout)
    assert "binding-sites-graph" in layout and "bs-table" in layout and "bs-detail" in layout
    assert len(d._app.callback_map) >= 3


def test_ranked_topn_weight_order_and_truncation(tmp_path):
    # Headless check of the weights-order mapping + top-N truncation that
    # the checklist/figure/table callbacks all delegate to.
    _write_example(tmp_path)
    pytest.importorskip("dash")
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    d = HotspotDashboard(out_path=str(tmp_path), pdb_path=None, port=8057)
    # slider values in _WEIGHT_SPECS order: affinity, probe_coverage, volume,
    # kinetics, shape, diversity -- matches DEFAULT_DASHBOARD_WEIGHTS.
    top = d._ranked_topn((3, 2, 1, 1, 1, 1), 1)
    assert len(top) == 1
    assert int(top.iloc[0]["rank"]) == 1
    assert int(top.iloc[0]["site_id"]) == 1  # site 1 is best on every feature in the fixture


def _asymmetric_bs_df():
    """Two sites tied on every feature except volume.

    ``_bs_df`` has site 1 dominate every single feature, so a negative
    weight placed on *any* slider slot flips the ranking toward site 2 --
    that fixture can't tell a correct slider->weight mapping from a
    scrambled (transposed) one. Here only volume differs (site 1 = 100,
    larger; site 2 = 50, smaller), so only a weight placed at the "volume"
    position (index 2 in ``_WEIGHT_SPECS`` order) can move the ranking.
    """
    df = _bs_df()
    tied_cols = ["agfe_min", "probe_coverage", "residence", "solidity", "favorable_atomtypes"]
    site1_vals = df.loc[df["site_id"] == 1, tied_cols].iloc[0]
    for col in tied_cols:
        df.loc[df["site_id"] == 2, col] = site1_vals[col]
    return df


def test_ranked_topn_weight_position_discriminates_transposition(tmp_path):
    # Guards against a scrambled slider->weight mapping (e.g. the value
    # meant for the "volume" slider landing on "affinity" instead): with a
    # fixture where only volume differs between sites, a -1 weight must land
    # on the volume slot (index 2) to change the ranking; the same -1 value
    # placed on a tied feature (affinity, index 0) must leave it unchanged.
    df = _asymmetric_bs_df()
    df.to_csv(tmp_path / "binding_sites.csv", index=False)
    pytest.importorskip("dash")
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    d = HotspotDashboard(out_path=str(tmp_path), pdb_path=None, port=8058)

    # Correct mapping: -1 on volume (position 2) -> smaller-volume site (2) wins.
    top_volume = d._ranked_topn((0, 0, -1, 0, 0, 0), 2)
    assert int(top_volume.iloc[0]["site_id"]) == 2

    # Transposed mapping: same -1 value shifted to affinity (position 0),
    # a feature that is tied between the two sites -> must NOT flip the
    # ranking (site 1, listed first with the larger volume, stays on top).
    top_affinity = d._ranked_topn((-1, 0, 0, 0, 0, 0), 2)
    assert int(top_affinity.iloc[0]["site_id"]) == 1


def test_find_pdb_searches_sibling_sim_dirs(tmp_path):
    # Layout: results/merged (dashboard target) + results/<sim>/averaged_trajectory.pdb
    pytest.importorskip("dash")
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    results = tmp_path / "results"
    merged = results / "merged"
    sim = results / "benzene_rep1"
    merged.mkdir(parents=True)
    sim.mkdir(parents=True)
    _bs_df().to_csv(merged / "binding_sites.csv", index=False)
    # No PDB anywhere yet -> constructs cleanly, _find_pdb returns None.
    d = HotspotDashboard(out_path=str(merged), pdb_path=None, port=8064)
    assert d._find_pdb() is None
    # Now a per-sim averaged structure exists in a SIBLING dir of merged/.
    (sim / "averaged_trajectory.pdb").write_text("REMARK placeholder\n")
    found = d._find_pdb()
    assert found is not None
    assert found.endswith(os.path.join("benzene_rep1", "averaged_trajectory.pdb"))


def test_empty_binding_sites_warns_and_shows_banner(tmp_path):
    # No binding_sites.csv, no PDB -> loud warnings + visible banner, not a blank page.
    pytest.importorskip("dash")
    from cosolvkit.analysis.hotspot_dashboard import HotspotDashboard
    d = HotspotDashboard(out_path=str(tmp_path), pdb_path=None, port=8066)
    assert d._bs_df.empty
    warns = d._data_warnings()
    assert any("binding_sites.csv" in w for w in warns)
    assert any("averaged_trajectory.pdb" in w for w in warns)
    layout_str = str(d._app.layout)
    assert "data-warning-banner" in layout_str
    assert "binding_sites.csv" in layout_str  # banner text rendered
