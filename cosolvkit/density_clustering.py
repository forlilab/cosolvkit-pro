#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Pluggable clustering strategies for hotspot detection
#

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.filters import gaussian
from sklearn.cluster import DBSCAN


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
        # Seeds: local minima of AGFE (most-negative = most-favorable)
        # peak_local_max on the negated AGFE finds local minima
        neg_agfe = -agfe_array
        masked_neg = np.where(favorable_mask, neg_agfe, -np.inf)
        coords = peak_local_max(
            masked_neg,
            min_distance=self.min_distance,
            labels=favorable_mask,
        )
        seed_mask = np.zeros(agfe_array.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = label(seed_mask)

        labeled_array = watershed(
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


class SkimageWatershedClustering:
    """Marker-controlled watershed using h_maxima seeds on the AGFE score field.

    Builds a positive score image from the AGFE map, finds local maxima with
    h_maxima (or peak_local_max for distance mode), then runs a
    marker-controlled watershed.  Handles merged pockets better than plain
    connected-components and is more robust than generic peak-detection when
    the density map is smooth.

    Parameters
    ----------
    min_cluster_voxels : int
        Clusters smaller than this are discarded after watershed (default 10).
    h : float
        h-maxima suppression height in score units.  Only local maxima that
        are at least *h* above their surrounding baseline become markers
        (default 0.5).  Increase to merge nearby sub-peaks; decrease to split
        them more aggressively.
    smoothing_sigma : float or None
        If given, apply a Gaussian filter with this sigma (in voxels) to the
        score image before computing h_maxima.  Useful to suppress salt-and-
        pepper noise before seeding (default None = disabled).
    min_distance : int
        Minimum distance (in voxels) between seed maxima.  Only used when
        ``watershed_mode="distance"`` (default 3).
    watershed_mode : {"score", "distance"}
        ``"score"`` (default): watershed on the clipped AGFE score field
        ``clip(-agfe, 0, None)``.  Seeds from h_maxima.
        ``"distance"``: watershed on the Euclidean distance transform of the
        favorable mask.  Seeds from local maxima in the distance field.  This
        tends to split blobs more evenly than score-based seeding.
    """

    def __init__(self, min_cluster_voxels=10, h=0.5,
                 smoothing_sigma=None, min_distance=3,
                 watershed_mode="score"):
        if watershed_mode not in ("score", "distance"):
            raise ValueError("watershed_mode must be 'score' or 'distance'")
        self.min_cluster_voxels = min_cluster_voxels
        self.h = h
        self.smoothing_sigma = smoothing_sigma
        self.min_distance = min_distance
        self.watershed_mode = watershed_mode

    def cluster(self, favorable_mask, agfe_array, gridsize):
        if self.watershed_mode == "score":
            score = np.clip(-agfe_array, 0, None)
            # Zero out voxels outside the favorable mask so h_maxima only
            # finds peaks inside the region of interest.
            score_masked = score * favorable_mask

            if self.smoothing_sigma is not None:
                score_masked = gaussian(score_masked, sigma=self.smoothing_sigma)
                # Re-apply mask after smoothing to avoid bleed-out artifacts
                score_masked = score_masked * favorable_mask

            maxima_mask = h_maxima(score_masked, h=self.h)
            # Restrict markers to the favorable region
            maxima_mask = maxima_mask & favorable_mask
            markers, _ = label(maxima_mask)

            labeled_array = watershed(
                -score,
                markers=markers,
                mask=favorable_mask,
            )

        else:  # watershed_mode == "distance"
            dt = distance_transform_edt(favorable_mask)
            coords = peak_local_max(
                dt,
                min_distance=self.min_distance,
                labels=favorable_mask,
            )
            seed_mask = np.zeros(agfe_array.shape, dtype=bool)
            if len(coords):
                seed_mask[tuple(coords.T)] = True
            markers, _ = label(seed_mask)

            labeled_array = watershed(
                -dt,
                markers=markers,
                mask=favorable_mask,
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
        vox_coords = np.argwhere(favorable_mask).astype(float)
        ang_coords = vox_coords * gridsize

        db = DBSCAN(
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
