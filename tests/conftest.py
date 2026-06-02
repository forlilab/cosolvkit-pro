"""Shared fixtures for the CosolvKit test suite."""

import os
import contextlib

import numpy as np
import pytest
from gridData import Grid

from cosolvkit.hotspots_detection import BindingSite


# ---------------------------------------------------------------------------
# BindingSite factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_binding_site():
    """Return a factory that builds a BindingSite with controlled geometry."""
    def _factory(
        rank=1,
        site_id=1,
        cosolvent="BEN",
        shape=(20, 20, 20),
        blob_slices=None,
        centroid=None,
        agfe_min=-2.0,
        agfe_mean_top_pct=-1.5,
        favorability_score=1.0,
        diversity_score=0.5,
        volume_score=0.5,
        composite_score=0.75,
        favorable_atomtypes=None,
        per_type_agfe=None,
        grid_origin=None,
        grid_delta=None,
    ):
        voxel_mask = np.zeros(shape, dtype=bool)
        if blob_slices is not None:
            voxel_mask[blob_slices] = True
        else:
            voxel_mask[5:10, 5:10, 5:10] = True

        _centroid = centroid if centroid is not None else np.array([3.75, 3.75, 3.75])
        site = BindingSite(
            rank=rank,
            site_id=site_id,
            cosolvent=cosolvent,
            n_voxels=int(voxel_mask.sum()),
            centroid=np.asarray(_centroid, dtype=float),
            agfe_min=agfe_min,
            agfe_mean_top_pct=agfe_mean_top_pct,
            voxel_mask=voxel_mask,
            favorability_score=favorability_score,
            diversity_score=diversity_score,
            volume_score=volume_score,
            composite_score=composite_score,
            favorable_atomtypes=favorable_atomtypes if favorable_atomtypes is not None else ["HBD"],
            per_type_agfe=per_type_agfe if per_type_agfe is not None else {"HBD": -2.0},
        )
        site.grid_origin = np.asarray(grid_origin if grid_origin is not None else [0.0, 0.0, 0.0])
        site.grid_delta = np.asarray(grid_delta if grid_delta is not None else [0.5, 0.5, 0.5])
        return site

    return _factory


# ---------------------------------------------------------------------------
# Synthetic AGFE array / grid helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_agfe_array():
    """20x20x20 AGFE array: hotspot blob at [5:10, 5:10, 5:10] = -2.0, background = 0."""
    arr = np.zeros((20, 20, 20), dtype=float)
    arr[5:10, 5:10, 5:10] = -2.0
    return arr


@pytest.fixture
def synthetic_edges():
    """Grid edges for a 20x20x20 grid, 0.5 Å spacing, starting at origin."""
    return [np.linspace(0.0, 10.0, 21)] * 3


# ---------------------------------------------------------------------------
# Two-blob clustering fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blob_mask():
    """Boolean 20x20x20 mask: two 5x5x5 blobs (125 vox each) + one tiny 2x2x2 blob (8 vox)."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[1:6, 1:6, 1:6] = True          # blob A
    mask[13:18, 13:18, 13:18] = True     # blob B
    mask[8:10, 8:10, 8:10] = True        # small blob, below typical min_cluster_voxels
    return mask


@pytest.fixture
def two_blob_agfe(two_blob_mask):
    """AGFE array matching two_blob_mask: -2.0 where favorable, 0 elsewhere."""
    arr = np.zeros((20, 20, 20), dtype=float)
    arr[two_blob_mask] = -2.0
    return arr


# ---------------------------------------------------------------------------
# Temporary working-directory helper (avoids solvent_accessible_map.dx pollution)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _tmp_cwd(path):
    orig = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(orig)


@pytest.fixture
def tmp_cwd(tmp_path):
    """Change CWD to tmp_path for the duration of the test, then restore."""
    with _tmp_cwd(tmp_path):
        yield tmp_path
