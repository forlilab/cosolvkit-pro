#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Pluggable clustering strategies for hotspot detection
#

import numpy as np
from scipy.ndimage import label

try:
    from skimage.segmentation import watershed as _skimage_watershed
    from skimage.feature import peak_local_max as _peak_local_max
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ConnectedComponentsClustering:
    """Cluster favorable voxels with connected-components labeling.

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum number of voxels a cluster must contain to be retained.
    connectivity : {6, 26}
        Voxel adjacency rule.  6 (default) connects only face-sharing voxels;
        26 also connects edge- and corner-sharing voxels, which reduces
        fragmentation across narrow bridges.
    """

    def __init__(self, min_cluster_voxels=10, connectivity=26):
        if connectivity not in (6, 26):
            raise ValueError("connectivity must be 6 or 26")
        self.min_cluster_voxels = min_cluster_voxels
        self.connectivity = connectivity

    def cluster(self, favorable_mask, agfe_array, gridsize):
        structure = np.ones((3, 3, 3)) if self.connectivity == 26 else None
        labeled_array, n_raw = label(favorable_mask, structure=structure)
        site_labels = [
            lbl for lbl in range(1, n_raw + 1)
            if int((labeled_array == lbl).sum()) >= self.min_cluster_voxels
        ]
        return labeled_array, site_labels


class WatershedClustering:
    """Cluster favorable voxels with a watershed transform on the AGFE values.

    Treats the AGFE map as a height field and floods from local minima
    (most-favorable voxels) outward.  This separates touching pockets that
    connected-components would merge into a single large cluster.

    Requires ``scikit-image`` (``pip install scikit-image``).

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum number of voxels a cluster must contain to be retained.
    min_distance : int
        Minimum distance (in voxels) between seed local minima (default 3).
    compactness : float
        Compactness parameter forwarded to ``skimage.segmentation.watershed``
        (default 0).  Larger values produce more compact, ball-shaped regions.
    """

    def __init__(self, min_cluster_voxels=10, min_distance=3, compactness=0.0):
        self.min_cluster_voxels = min_cluster_voxels
        self.min_distance = min_distance
        self.compactness = compactness

    def cluster(self, favorable_mask, agfe_array, gridsize):
        if not _SKIMAGE_AVAILABLE:
            raise ImportError(
                "WatershedClustering requires scikit-image. "
                "Install it with: pip install scikit-image"
            )
        # Seeds: local minima of AGFE (most-negative = most-favorable)
        # peak_local_max on the negated AGFE finds local minima
        neg_agfe = -agfe_array
        masked_neg = np.where(favorable_mask, neg_agfe, -np.inf)
        coords = _peak_local_max(
            masked_neg,
            min_distance=self.min_distance,
            labels=favorable_mask,
        )
        seed_mask = np.zeros(agfe_array.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = label(seed_mask)

        labeled_array = _skimage_watershed(
            agfe_array,
            markers=markers,
            mask=favorable_mask,
            compactness=self.compactness,
        )
        n_raw = int(labeled_array.max())
        site_labels = [
            lbl for lbl in range(1, n_raw + 1)
            if int((labeled_array == lbl).sum()) >= self.min_cluster_voxels
        ]
        return labeled_array, site_labels


class DBSCANClustering:
    """Cluster favorable voxels with DBSCAN on their Angstrom coordinates.

    Purely spatial: ignores AGFE intensity, depends only on whether favorable
    voxels are within ``eps_angstrom`` of each other.  Independent of voxel
    adjacency rules.

    Requires ``scikit-learn`` (``pip install scikit-learn``).

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum number of voxels a cluster must contain to be retained
        (maps to DBSCAN ``min_samples``).
    eps_angstrom : float
        Neighbourhood radius in Angstroms (default 1.5).  Roughly 2–3× the
        grid spacing works well for a 0.5 Å grid.
    """

    def __init__(self, min_cluster_voxels=10, eps_angstrom=1.5):
        self.min_cluster_voxels = min_cluster_voxels
        self.eps_angstrom = eps_angstrom

    def cluster(self, favorable_mask, agfe_array, gridsize):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "DBSCANClustering requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            )
        vox_coords = np.argwhere(favorable_mask).astype(float)
        ang_coords = vox_coords * gridsize

        db = _DBSCAN(
            eps=self.eps_angstrom,
            min_samples=self.min_cluster_voxels,
            n_jobs=-1,
        ).fit(ang_coords)
        raw_labels = db.labels_  # -1 = noise

        labeled_array = np.zeros(agfe_array.shape, dtype=int)
        for i, vox in enumerate(np.argwhere(favorable_mask)):
            lbl = int(raw_labels[i])
            if lbl >= 0:
                labeled_array[tuple(vox)] = lbl + 1  # shift so 0 = background

        n_raw = int(labeled_array.max())
        site_labels = [
            lbl for lbl in range(1, n_raw + 1)
            if int((labeled_array == lbl).sum()) >= self.min_cluster_voxels
        ]
        return labeled_array, site_labels
