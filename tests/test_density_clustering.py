"""Tests for density_clustering.py — all four clustering strategies.

Fixtures provide a 20×20×20 boolean mask with two clearly separated 5×5×5
blobs (125 voxels each) and one tiny 2×2×2 blob (8 voxels).
"""

import numpy as np
import pytest

from cosolvkit.analysis.density_clustering import (
    ConnectedComponentsClustering,
    DBSCANClustering,
    SkimageWatershedClustering,
    WatershedClustering,
)

GRIDSIZE = 0.5  # Å per voxel — used by DBSCAN to convert to Angstroms


# ---------------------------------------------------------------------------
# ConnectedComponentsClustering
# ---------------------------------------------------------------------------

class TestConnectedComponentsClustering:

    def test_two_blobs_found_tiny_blob_filtered(self, two_blob_mask, two_blob_agfe):
        # The two 125-voxel blobs survive; the 8-voxel tiny blob is filtered at
        # either threshold.
        for min_voxels in (10, 30):
            cc = ConnectedComponentsClustering(min_cluster_voxels=min_voxels, connectivity=26)
            _, labels = cc.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
            assert len(labels) == 2, min_voxels

    def test_all_blobs_filtered_out(self, two_blob_mask, two_blob_agfe):
        cc = ConnectedComponentsClustering(min_cluster_voxels=200, connectivity=26)
        _, labels = cc.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0

    def test_connectivity_6_separates_blobs(self, two_blob_mask, two_blob_agfe):
        # Well-separated blobs: connectivity=6 should still find both
        cc = ConnectedComponentsClustering(min_cluster_voxels=10, connectivity=6)
        _, labels = cc.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 2

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError):
            ConnectedComponentsClustering(connectivity=18)

    def test_labeled_array_shape_and_background(self, two_blob_mask, two_blob_agfe):
        cc = ConnectedComponentsClustering(min_cluster_voxels=10)
        labeled, _ = cc.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert labeled.shape == two_blob_mask.shape
        assert labeled[0, 0, 0] == 0  # corner is background

    def test_empty_mask_gives_no_clusters(self, two_blob_agfe):
        empty = np.zeros((20, 20, 20), dtype=bool)
        cc = ConnectedComponentsClustering(min_cluster_voxels=1)
        _, labels = cc.cluster(empty, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0

    def test_single_voxel_cluster(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        agfe = np.zeros((10, 10, 10))
        agfe[5, 5, 5] = -1.0
        cc = ConnectedComponentsClustering(min_cluster_voxels=1)
        _, labels = cc.cluster(mask, agfe, GRIDSIZE)
        assert len(labels) == 1


# ---------------------------------------------------------------------------
# WatershedClustering
# ---------------------------------------------------------------------------

class TestWatershedClustering:

    def test_two_separated_blobs_found(self, two_blob_mask, two_blob_agfe):
        ws = WatershedClustering(min_cluster_voxels=10, min_distance=3)
        labeled, labels = ws.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) >= 2
        assert labeled.shape == two_blob_mask.shape

    def test_min_voxels_filtering(self, two_blob_mask, two_blob_agfe):
        ws = WatershedClustering(min_cluster_voxels=200)
        _, labels = ws.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0

    def test_empty_mask(self, two_blob_agfe):
        empty = np.zeros((20, 20, 20), dtype=bool)
        ws = WatershedClustering(min_cluster_voxels=1)
        _, labels = ws.cluster(empty, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# SkimageWatershedClustering
# ---------------------------------------------------------------------------

class TestSkimageWatershedClustering:

    def test_score_mode_finds_blobs(self, two_blob_mask, two_blob_agfe):
        sw = SkimageWatershedClustering(min_cluster_voxels=10, h=0.5, watershed_mode="score")
        labeled, labels = sw.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) >= 1
        assert labeled.shape == two_blob_mask.shape

    def test_distance_mode_finds_blobs(self, two_blob_mask, two_blob_agfe):
        sw = SkimageWatershedClustering(min_cluster_voxels=10, h=0.5, watershed_mode="distance")
        _, labels = sw.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) >= 1

    def test_invalid_watershed_mode_raises(self):
        with pytest.raises(ValueError, match="watershed_mode"):
            SkimageWatershedClustering(watershed_mode="bogus")

    def test_large_h_reduces_cluster_count(self, two_blob_mask, two_blob_agfe):
        """Very large h suppresses all maxima → 0 or 1 cluster."""
        sw = SkimageWatershedClustering(min_cluster_voxels=10, h=100.0)
        _, labels = sw.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) <= 1

    def test_min_voxels_filtering(self, two_blob_mask, two_blob_agfe):
        sw = SkimageWatershedClustering(min_cluster_voxels=200, h=0.5)
        _, labels = sw.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0

    def test_smoothing_sigma_does_not_crash(self, two_blob_mask, two_blob_agfe):
        sw = SkimageWatershedClustering(min_cluster_voxels=10, h=0.1, smoothing_sigma=1.0)
        labeled, labels = sw.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert labeled.shape == two_blob_mask.shape


# ---------------------------------------------------------------------------
# DBSCANClustering
# ---------------------------------------------------------------------------

class TestDBSCANClustering:

    def test_two_separated_blobs(self, two_blob_mask, two_blob_agfe):
        # Blobs A and B are ~3.5 Å apart; eps=1.5 Å keeps them separate
        db = DBSCANClustering(min_cluster_voxels=10, eps_angstrom=1.5)
        labeled, labels = db.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        assert len(labels) >= 2
        assert labeled.shape == two_blob_mask.shape

    def test_empty_mask_gives_no_clusters(self, two_blob_agfe):
        empty = np.zeros((20, 20, 20), dtype=bool)
        db = DBSCANClustering(min_cluster_voxels=1, eps_angstrom=1.5)
        labeled, labels = db.cluster(empty, two_blob_agfe, GRIDSIZE)
        assert len(labels) == 0
        assert labeled.shape == two_blob_agfe.shape

    def test_background_voxels_are_zero_in_labeled(self, two_blob_mask, two_blob_agfe):
        db = DBSCANClustering(min_cluster_voxels=5, eps_angstrom=1.5)
        labeled, _ = db.cluster(two_blob_mask, two_blob_agfe, GRIDSIZE)
        # Voxels not in the mask must be 0 (background)
        assert np.all(labeled[~two_blob_mask] == 0)
