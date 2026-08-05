#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Pluggable clustering strategies for hotspot detection
#

import warnings

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.filters import gaussian


def min_cluster_voxels_for_volume(volume_ang3, gridsize):
    """Voxel count matching a physical minimum cluster volume, at least 1.

    ``min_cluster_voxels`` is a raw count, so its physical meaning scales as
    ``1 / gridsize**3``; a volume threshold is grid-independent.

    Parameters
    ----------
    volume_ang3 : float
        Minimum cluster volume in A^3.
    gridsize : float
        Voxel edge length in A; must be positive.
    """
    if gridsize is None or float(gridsize) <= 0:
        raise ValueError(f"gridsize must be positive, got {gridsize!r}")
    return max(1, int(round(float(volume_ang3) / float(gridsize) ** 3)))


class ConnectedComponentsClustering:
    """Cluster favorable voxels with connected-components labeling.

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum voxels a cluster must contain to be retained.
    connectivity : {6, 26}
        Voxel adjacency: 6 links face-sharing voxels only; 26 also links
        edge- and corner-sharing ones, reducing fragmentation across bridges.
    """

    def __init__(self, min_cluster_voxels, connectivity=26):
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


class SkimageWatershedClustering:
    """Marker-controlled watershed using h_maxima seeds on the AGFE score field.

    More robust than generic peak detection on smooth maps, and unlike plain
    connected-components it splits merged pockets.

    Parameters
    ----------
    min_cluster_voxels : int
        Clusters smaller than this are discarded after watershed.
    h : float
        h-maxima suppression height, in score units: a local maximum seeds a
        region only if it rises at least *h* above its surrounding baseline.
        Increase to merge nearby sub-peaks, decrease to split them.
    smoothing_sigma : float or None
        Gaussian sigma, in voxels, applied to the score image before seeding
        to suppress speckle noise.  ``None`` disables smoothing.
    min_distance : int or None
        Minimum separation, in voxels, between seed maxima. ``"distance"`` mode ONLY -- in
        ``"score"`` mode seeds come from h_maxima and this is ignored, so passing it there
        warns rather than silently doing nothing. Defaults to 3 in distance mode.
    watershed_mode : {"score", "distance"}
        ``"score"``: watershed the clipped AGFE score ``clip(-agfe, 0, None)``,
        seeded by h_maxima.  ``"distance"``: watershed the Euclidean distance
        transform of the favorable mask, seeded by its local maxima, which
        splits blobs more evenly.
    """

    def __init__(self, min_cluster_voxels, h=0.5,
                 smoothing_sigma=None, min_distance=None,
                 watershed_mode="score"):
        if watershed_mode not in ("score", "distance"):
            raise ValueError("watershed_mode must be 'score' or 'distance'")
        if watershed_mode == "score" and min_distance is not None:
            warnings.warn(
                "min_distance is ignored in watershed_mode='score' (seeds come from h_maxima); "
                "set `h` to control splitting, or use watershed_mode='distance'.",
                RuntimeWarning, stacklevel=2,
            )
        self.min_cluster_voxels = min_cluster_voxels
        self.h = h
        self.smoothing_sigma = smoothing_sigma
        self.min_distance = 3 if min_distance is None else min_distance
        self.watershed_mode = watershed_mode

    def cluster(self, favorable_mask, agfe_array, gridsize):
        if self.watershed_mode == "score":
            score = np.clip(-agfe_array, 0, None)
            # h_maxima takes no mask, so zero the outside to confine peaks.
            score_masked = score * favorable_mask

            if self.smoothing_sigma is not None:
                score_masked = gaussian(score_masked, sigma=self.smoothing_sigma)
                # The Gaussian bleeds past the mask edge; re-apply it.
                score_masked = score_masked * favorable_mask

            maxima_mask = h_maxima(score_masked, h=self.h)
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


def build_clustering_strategy(clustering_cfg, gridsize):
    """Build a clustering-strategy instance from a ``ClusteringConfig``.

    Constructs the class named by ``clustering_cfg.strategy`` with the resolved
    ``min_cluster_voxels`` plus any ``strategy_kwargs``.

    *gridsize* is REQUIRED: the size threshold is configured as a volume, so it cannot be turned
    into a voxel count without it. It used to default to None, and passing nothing silently ignored
    ``min_cluster_volume_ang3`` and fell back to a raw count.

    Raises
    ------
    ValueError
        If ``clustering_cfg.strategy`` is not a known strategy name.
    """
    # Two strategies, deliberately. `skimage_watershed` splits merged pockets using an
    # h_maxima contrast criterion; `connected_components` is the no-splitting baseline.
    # Removed: `watershed` (seeded by peak_local_max with a grid-dependent `min_distance`
    # rather than a contrast threshold, so it was strictly worse than skimage_watershed) and
    # `dbscan` (ignored both voxel adjacency and AGFE intensity). Neither was ever a default.
    registry = {
        "skimage_watershed": SkimageWatershedClustering,
        "connected_components": ConnectedComponentsClustering,
    }
    strategy = clustering_cfg.strategy
    if strategy not in registry:
        raise ValueError(
            f"Unknown clustering strategy {strategy!r}; "
            f"valid options: {sorted(registry)}"
        )
    kwargs = dict(clustering_cfg.strategy_kwargs or {})
    n_vox = clustering_cfg.resolve_min_cluster_voxels(gridsize)
    return registry[strategy](min_cluster_voxels=n_vox, **kwargs)
