"""Tests for HotspotDetector — uses synthetic .dx files, no real trajectory.

Key insight: self.universe is only accessed inside survival_probability()
(in sites/detect.py). All other methods are pure grid-math.
We pass compute_survival_probability=False so survival_probability is never
called, making universe=None safe for all tests in this file.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from gridData import Grid

from cosolvkit.analysis.core.models import Hotspot
from cosolvkit.analysis.sites.detect import HotspotDetector


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


def _make_detector(tmp_path, cosolvent="BEN", agfe_cutoff=-1.0, min_cluster_voxels=10):
    return HotspotDetector(
        out_path=str(tmp_path),
        cosolvent_names=[cosolvent],
        universe=None,          # safe because compute_survival_probability=False
        agfe_cutoff=agfe_cutoff,
        min_cluster_voxels=min_cluster_voxels,
        compute_survival_probability=False,
    )


# ---------------------------------------------------------------------------
# detect() — core behavior
# ---------------------------------------------------------------------------

class TestDetect:

    def test_detect_populates_site_fields(self, tmp_path):
        """One detect() call, all the per-site fields it must fill in."""
        _make_agfe_grid(tmp_path, "BEN", hotspot_val=-2.0)
        sites = _make_detector(tmp_path, agfe_cutoff=-1.0).detect("BEN")
        assert len(sites) >= 1
        assert sites[0].rank == 1
        assert sites[0].cosolvent == "BEN"
        assert sites[0].agfe_min < -1.0
        assert sites[0].grid_origin is not None
        assert sites[0].grid_delta is not None
        assert len(sites[0].grid_origin) == 3

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

    def test_rank_by_agfe_min(self, tmp_path):
        # two blobs, different depths -> deeper (more negative) ranks first
        import numpy as np
        from gridData import Grid
        arr = np.zeros((20, 20, 20)); arr[3:9, 3:9, 3:9] = -3.0; arr[12:16, 12:16, 12:16] = -2.0
        edges = [np.linspace(0, 10, 21)] * 3
        Grid(arr, edges=edges).export(str(tmp_path / "map_agfe_BEN.dx"))
        d = _make_detector(tmp_path, min_cluster_voxels=10)
        sites = d.detect("BEN")
        assert len(sites) == 2
        assert sites[0].rank == 1 and sites[0].agfe_min == pytest.approx(-3.0)
        assert sites[1].rank == 2 and sites[1].agfe_min == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

class TestExportResults:

    def test_per_cosolvent_and_combined_files_written(self, tmp_path):
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        assert (tmp_path / "hotspot_sites_BEN.csv").exists()
        assert (tmp_path / "hotspot_sites_all.tsv").exists()
        with open(tmp_path / "hotspot_sites_BEN.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_csv_has_expected_columns(self, tmp_path):
        import pandas as pd
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        d.export_results(results, label_map=False)
        df = pd.read_csv(tmp_path / "hotspot_sites_BEN.csv")
        for col in ("rank", "site_id", "cosolvent", "n_voxels",
                    "centroid_x", "centroid_y", "centroid_z",
                    "agfe_min", "agfe_mean_top_pct"):
            assert col in df.columns, f"Missing column: {col}"

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

    def test_roundtrip_preserves_sites(self, tmp_path):
        """save -> load must return the same sites, masks, grid frame and extras."""
        _make_agfe_grid(tmp_path, "BEN")
        d = _make_detector(tmp_path)
        results = {"BEN": d.detect("BEN")}
        results["BEN"][0].add_property("my_metric", 42.0)
        HotspotDetector.save_checkpoint(results, str(tmp_path))

        npz = tmp_path / "hotspot_checkpoints" / "hotspot_checkpoint_BEN.npz"
        assert npz.exists()

        loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
        assert len(loaded["BEN"]) == len(results["BEN"])
        orig, restored = results["BEN"][0], loaded["BEN"][0]
        assert orig.voxel_mask.shape == restored.voxel_mask.shape
        assert np.array_equal(orig.voxel_mask, restored.voxel_mask)
        assert np.allclose(orig.grid_origin, restored.grid_origin, atol=1e-4)
        assert np.allclose(orig.grid_delta, restored.grid_delta, atol=1e-4)
        assert np.allclose(orig.centroid, restored.centroid, atol=0.01)
        assert restored.properties.get("my_metric") == pytest.approx(42.0)

    def test_load_missing_cosolvent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HotspotDetector.load_checkpoint(str(tmp_path), ["NOSUCHCOSOLVENT"])

    def test_empty_results_no_checkpoint_file(self, tmp_path):
        """Empty site list → no checkpoint file written (debug-logged, not error)."""
        HotspotDetector.save_checkpoint({"BEN": []}, str(tmp_path))
        npz = tmp_path / "hotspot_checkpoints" / "hotspot_checkpoint_BEN.npz"
        assert not npz.exists()


# ---------------------------------------------------------------------------
# CSV schema: geom_* columns go to sidecar
# ---------------------------------------------------------------------------

class TestCsvSlim:
    def test_geom_columns_go_to_the_sidecar_only(self, tmp_path):
        # build a detector whose sites carry geom_* via properties
        _make_agfe_grid(tmp_path, "BEN")
        det = _make_detector(tmp_path)
        sites = det.detect("BEN")
        sites[0].add_property("geom_solidity", 0.42)  # simulate a geom_* column
        det.export_results({"BEN": sites}, label_map=False)

        main = pd.read_csv(tmp_path / "hotspot_sites_BEN.csv")
        assert not any(c.startswith("geom_") for c in main.columns)
        assert "agfe_min" in main.columns

        sidecar = pd.read_csv(tmp_path / "hotspot_sites_geom_BEN.csv")
        assert "site_id" in sidecar.columns
        assert "geom_solidity" in sidecar.columns

        all_df = pd.read_csv(tmp_path / "hotspot_sites_all.tsv", sep="\t")
        assert not any(c.startswith("geom_") for c in all_df.columns)
        assert "agfe_min" in all_df.columns
