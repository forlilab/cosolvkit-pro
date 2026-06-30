# tests/test_scoring_golden.py
"""Golden-master characterization of the two scoring paths.

Captured BEFORE the Phase-0 scoring unify. The unified scorer must reproduce
these exactly. detect() values are hand-derived from a deterministic two-blob
fixture; compute_composite_score values are hand-derived from fixed sites.
"""
import numpy as np
import pytest
from gridData import Grid

from cosolvkit.analysis.hotspots_detection import HotspotDetector
from cosolvkit.analysis.pocket_properties import compute_composite_score


def _write_two_blob_agfe(out_dir, cosolvent="BEN"):
    """Two separated constant blobs: A=6^3 voxels @ -3.0, B=4^3 voxels @ -2.0."""
    shape = (20, 20, 20)
    arr = np.zeros(shape, dtype=float)
    arr[3:9, 3:9, 3:9] = -3.0     # 216 voxels
    arr[12:16, 12:16, 12:16] = -2.0  # 64 voxels
    edges = [np.linspace(0, shape[i] * 0.5, shape[i] + 1) for i in range(3)]
    Grid(arr, edges=edges).export(str(out_dir / f"map_agfe_{cosolvent}.dx"))


def test_detect_scoring_golden(tmp_path):
    _write_two_blob_agfe(tmp_path, "BEN")
    det = HotspotDetector(
        out_path=str(tmp_path), cosolvent_names=["BEN"], universe=None,
        agfe_cutoff=-1.0, min_cluster_voxels=10,
        compute_survival_probability=False, score_weights=None,
    )
    sites = det.detect("BEN")
    assert len(sites) == 2

    by_rank = {s.rank: s for s in sites}
    a, b = by_rank[1], by_rank[2]

    # Blob A: all voxels -3.0 -> favorability inverted-minmax = 1.0; volume 216/216 = 1.0
    # composite = 0.5*1.0 + 0.3*0.0 + 0.2*1.0 = 0.7
    assert a.n_voxels == 216
    assert a.favorability_score == pytest.approx(1.0, abs=1e-9)
    assert a.diversity_score == pytest.approx(0.0, abs=1e-9)
    assert a.volume_score == pytest.approx(1.0, abs=1e-9)
    assert a.composite_score == pytest.approx(0.7, abs=1e-9)

    # Blob B: favorability 0.0; volume 64/216; composite = 0.2 * (64/216)
    assert b.n_voxels == 64
    assert b.favorability_score == pytest.approx(0.0, abs=1e-9)
    assert b.diversity_score == pytest.approx(0.0, abs=1e-9)
    assert b.volume_score == pytest.approx(64.0 / 216.0, abs=1e-9)
    assert b.composite_score == pytest.approx(0.2 * (64.0 / 216.0), abs=1e-9)


def test_compute_composite_score_golden(make_hotspot):
    # favorability_score = [0.2, 0.8, 0.5]; sp_mrt = [10, 20, 35]; equal weights.
    favs = [0.2, 0.8, 0.5]
    mrts = [10.0, 20.0, 35.0]
    sites = []
    for i, (f, m) in enumerate(zip(favs, mrts)):
        s = make_hotspot(rank=i + 1, site_id=i + 1, favorability_score=f)
        s.add_property("sp_mrt", m)
        sites.append(s)

    compute_composite_score(sites, {"favorability": 1.0, "sp_mrt": 1.0})

    comp = {s.site_id: s.composite_score for s in sites}
    rank = {s.site_id: s.rank for s in sites}
    # raw weights 1.0 each -> normalized to 0.5 each internally
    # fav minmax: [0, 1, 0.5]; sp_mrt minmax: [0, 0.4, 1.0]
    assert comp[1] == pytest.approx(0.5 * 0.0 + 0.5 * 0.0, abs=1e-9)   # 0.0
    assert comp[2] == pytest.approx(0.5 * 1.0 + 0.5 * 0.4, abs=1e-9)   # 0.7
    assert comp[3] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0, abs=1e-9)   # 0.75
    assert rank == {3: 1, 2: 2, 1: 3}


def test_compute_composite_score_none_and_flat_edges(make_hotspot):
    # Key "favorability" is FLAT (all equal) -> component 1.0 for every site.
    # Key "sp_mrt" is finite for 3 sites and MISSING (None) for 1 -> that site's
    # sp_mrt component is 0.0; the others are min-maxed over the finite set.
    # Weights {favorability:1.0, sp_mrt:1.0} -> normalized 0.5 each over both
    # active keys (both keys have >=1 finite value, so neither is dropped).
    favs = [0.5, 0.5, 0.5, 0.5]          # flat -> all favorability components = 1.0
    mrts = [10.0, 20.0, 30.0, None]      # finite minmax over {10,20,30}: 0.0, 0.5, 1.0; None -> 0.0
    sites = []
    for i, (f, m) in enumerate(zip(favs, mrts)):
        s = make_hotspot(rank=i + 1, site_id=i + 1, favorability_score=f)
        if m is not None:
            s.add_property("sp_mrt", m)   # site 4 has NO sp_mrt -> treated as missing/None
        sites.append(s)

    compute_composite_score(sites, {"favorability": 1.0, "sp_mrt": 1.0})

    comp = {s.site_id: s.composite_score for s in sites}
    # composite = 0.5*fav_component + 0.5*sp_component
    # site1: 0.5*1.0 + 0.5*0.0  = 0.5
    # site2: 0.5*1.0 + 0.5*0.5  = 0.75
    # site3: 0.5*1.0 + 0.5*1.0  = 1.0
    # site4: 0.5*1.0 + 0.5*0.0  = 0.5   (sp_mrt missing -> 0.0 component)
    assert comp[1] == pytest.approx(0.5, abs=1e-9)
    assert comp[2] == pytest.approx(0.75, abs=1e-9)
    assert comp[3] == pytest.approx(1.0, abs=1e-9)
    assert comp[4] == pytest.approx(0.5, abs=1e-9)
