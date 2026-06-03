"""Tests for HotspotDetector — uses synthetic .dx files, no real trajectory.

Key insight: self.universe is only accessed inside survival_probability()
(line ~891 of hotspots_detection.py). All other methods are pure grid-math.
We pass top_n_survival=0 so survival_probability is never called, making
universe=None safe for all tests in this file.
"""

import json
import os

import numpy as np
import pytest
from gridData import Grid

from cosolvkit.analysis.hotspots_detection import BindingSite, HotspotDetector


# ---------------------------------------------------------------------------
# Helpers to write synthetic .dx maps
# ---------------------------------------------------------------------------

def _make_agfe_grid(out_dir, cosolvent, shape=(20, 20, 20), hotspot_slices=None, hotspot_val=-2.0):
    """Write map_agfe_{cosolvent}.dx to out_dir."""
    arr = np.zeros(shape, dtype=float)
    if hotspot_slices is not None:
        arr[hotspot_slices] = hotspot_val
    else:
        arr[5:10, 5:10, 5:10] = hotspot_val
    edges = [np.linspace(0, shape[i] * 0.5, shape[i] + 1) for i in range(3)]
    Grid(arr, edges=edges).export(str(out_dir / f"map_agfe_{cosolvent}.dx"))
    return arr, edges


def _make_per_type_grids(out_dir, cosolvent, hbd_hotspot=True, hba_hotspot=False):
    """Write map_agfe_HBD_{cosolvent}.dx and map_agfe_HBA_{cosolvent}.dx."""
    shape = (20, 20, 20)
    edges = [np.linspace(0, 10, 21)] * 3

    hbd = np.zeros(shape, dtype=float)
    if hbd_hotspot:
        hbd[5:10, 5:10, 5:10] = -2.0

    hba = np.zeros(shape, dtype=float)
    if hba_hotspot:
        hba[5:10, 5:10, 5:10] = -2.0

    Grid(hbd, edges=edges).export(str(out_dir / f"map_agfe_HBD_{cosolvent}.dx"))
    Grid(hba, edges=edges).export(str(out_dir / f"map_agfe_HBA_{cosolvent}.dx"))


def _make_detector(tmp_path, cosolvent="BEN", agfe_cutoff=-1.0, min_cluster_voxels=10,
                   score_weights=None):
    return HotspotDetector(
        out_path=str(tmp_path),
        cosolvent_names=[cosolvent],
        universe=None,          # safe because top_n_survival=0
        agfe_cutoff=agfe_cutoff,
        min_cluster_voxels=min_cluster_voxels,
        top_n_survival=0,
        score_weights=score_weights,
    )


# ---------------------------------------------------------------------------
# detect() — core behavior
# ---------------------------------------------------------------------------

class TestDetect:

    def test_hotspot_detected(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        sites = d.detect("BEN")
        assert len(sites) >= 1

    def test_top_site_rank_is_one(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        assert sites[0].rank == 1

    def test_top_site_agfe_below_cutoff(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN", hotspot_val=-2.0)
        d = _make_detector(tmp_path, agfe_cutoff=-1.0)
        sites = d.detect("BEN")
        assert sites[0].agfe_min < -1.0

    def test_centroid_inside_hotspot_region(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        sites = d.detect("BEN")
        c = sites[0].centroid
        # hotspot at voxels [5:10], gridsize=0.5 → Angstrom range [2.5, 5.0]
        assert 2.0 <= c[0] <= 5.5
        assert 2.0 <= c[1] <= 5.5
        assert 2.0 <= c[2] <= 5.5

    def test_no_favorable_voxels_returns_empty(self, tmp_path):
        # All AGFE = 0.0, cutoff = -1.0 → no favorable voxels
        arr = np.zeros((20, 20, 20))
        edges = [np.linspace(0, 10, 21)] * 3
        Grid(arr, edges=edges).export(str(tmp_path / "map_agfe_BEN.dx"))
        sites = _make_detector(tmp_path, agfe_cutoff=-1.0).detect("BEN")
        assert sites == []

    def test_missing_dx_raises_file_not_found(self, tmp_path):
        d = _make_detector(tmp_path, cosolvent="NMA")
        with pytest.raises(FileNotFoundError):
            d.detect("NMA")

    def test_grid_metadata_attached_to_sites(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        assert sites[0].grid_origin is not None
        assert sites[0].grid_delta is not None
        assert len(sites[0].grid_origin) == 3

    def test_cosolvent_stored_on_site(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        assert sites[0].cosolvent == "BEN"


# ---------------------------------------------------------------------------
# Scoring invariants
# ---------------------------------------------------------------------------

class TestScoring:

    def test_scores_in_unit_interval(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        for site in sites:
            assert 0.0 <= site.favorability_score <= 1.0
            assert 0.0 <= site.diversity_score <= 1.0
            assert 0.0 <= site.volume_score <= 1.0

    def test_single_cluster_has_max_scores(self, tmp_path):
        # Only one cluster → normalization collapses → favorability == 1, volume == 1
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path, min_cluster_voxels=1)
        sites = d.detect("BEN")
        # If only 1 cluster survives after filtering:
        if len(sites) == 1:
            assert sites[0].favorability_score == 1.0
            assert sites[0].volume_score == 1.0

    def test_equal_favorability_gives_ones(self, tmp_path):
        """When f_max == f_min, normalization must not divide by zero."""
        # Two identical-depth blobs: all agfe_mean_top_pct identical
        arr = np.zeros((20, 20, 20), dtype=float)
        arr[2:5, 2:5, 2:5] = -2.0   # blob A
        arr[14:17, 14:17, 14:17] = -2.0  # blob B, same depth
        edges = [np.linspace(0, 10, 21)] * 3
        Grid(arr, edges=edges).export(str(tmp_path / "map_agfe_BEN.dx"))
        sites = _make_detector(tmp_path, min_cluster_voxels=5).detect("BEN")
        # Both should have favorability_score == 1.0 (equal → ones)
        for s in sites:
            assert s.favorability_score == pytest.approx(1.0, abs=1e-6)

    def test_per_atomtype_diversity_score(self, tmp_path):
        """One favorable type out of two → diversity_score between 0 and 1."""
        _make_per_type_grids(tmp_path, "BEN", hbd_hotspot=True, hba_hotspot=False)
        d = _make_detector(tmp_path, agfe_cutoff=-1.0)
        sites = d.detect("BEN")
        assert len(sites) >= 1
        # 1 out of 2 atom types favorable → diversity in (0, 1)
        assert 0.0 < sites[0].diversity_score < 1.0

    def test_both_atomtypes_favorable_gives_max_diversity(self, tmp_path):
        _make_per_type_grids(tmp_path, "BEN", hbd_hotspot=True, hba_hotspot=True)
        d = _make_detector(tmp_path, agfe_cutoff=-1.0)
        sites = d.detect("BEN")
        assert sites[0].diversity_score == pytest.approx(1.0, abs=1e-6)

    def test_no_atomtype_map_gives_zero_diversity(self, tmp_path):
        """Single combined map (no per-type files) → diversity = 0."""
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        assert sites[0].diversity_score == 0.0

    def test_custom_score_weights_accepted(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path, score_weights={"favorability": 1.0, "diversity": 0.0, "volume": 0.0})
        sites = d.detect("BEN")
        assert len(sites) >= 1
        # Weights are normalized internally; composite should be non-negative
        assert sites[0].composite_score >= 0.0

    def test_composite_score_is_finite_and_nonnegative(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        sites = _make_detector(tmp_path).detect("BEN")
        for s in sites:
            assert np.isfinite(s.composite_score)
            assert s.composite_score >= 0.0


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

class TestExportResults:

    def test_csv_written(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        assert (tmp_path / "hotspot_sites_BEN.csv").exists()

    def test_json_written(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        assert (tmp_path / "hotspot_sites_BEN.json").exists()

    def test_combined_tsv_written(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        assert (tmp_path / "hotspot_sites_all.tsv").exists()

    def test_csv_has_expected_columns(self, tmp_path):
        import pandas as pd
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        df = pd.read_csv(tmp_path / "hotspot_sites_BEN.csv")
        for col in ("rank", "site_id", "cosolvent", "n_voxels",
                    "centroid_x", "centroid_y", "centroid_z",
                    "agfe_min", "favorability_score", "composite_score"):
            assert col in df.columns, f"Missing column: {col}"

    def test_json_is_valid(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        with open(tmp_path / "hotspot_sites_BEN.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_label_map_written(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=True)
        assert (tmp_path / "hotspot_labels_BEN.dx").exists()

    def test_empty_results_no_crash(self, tmp_path):
        d = _make_detector(tmp_path)
        d.export_results({"BEN": []}, label_map=False)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def test_roundtrip_site_count(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        assert len(loaded["BEN"]) == len(results["BEN"])

    def test_roundtrip_voxel_mask_shape(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        orig = results["BEN"][0].voxel_mask
        restored = loaded["BEN"][0].voxel_mask
        assert orig.shape == restored.shape
        assert np.array_equal(orig, restored)

    def test_roundtrip_grid_metadata(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        orig = results["BEN"][0]
        restored = loaded["BEN"][0]
        assert np.allclose(orig.grid_origin, restored.grid_origin, atol=1e-4)
        assert np.allclose(orig.grid_delta, restored.grid_delta, atol=1e-4)

    def test_roundtrip_centroid(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        orig = results["BEN"][0].centroid
        restored = loaded["BEN"][0].centroid
        assert np.allclose(orig, restored, atol=0.01)

    def test_roundtrip_custom_properties(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        results["BEN"][0].add_property("my_metric", 42.0)
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        assert loaded["BEN"][0].properties.get("my_metric") == pytest.approx(42.0)

    def test_load_missing_cosolvent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HotspotDetector.load_checkpoint(str(tmp_path), ["NOSUCHCOSOLVENT"])

    def test_npz_file_created(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        HotspotDetector.save_checkpoint(results, str(tmp_path))
        npz = tmp_path / "hotspot_checkpoints" / "hotspot_checkpoint_BEN.npz"
        assert npz.exists()

    def test_empty_results_no_checkpoint_file(self, tmp_path):
        """Empty site list → no checkpoint file written (debug-logged, not error)."""
        HotspotDetector.save_checkpoint({"BEN": []}, str(tmp_path))
        npz = tmp_path / "hotspot_checkpoints" / "hotspot_checkpoint_BEN.npz"
        assert not npz.exists()
