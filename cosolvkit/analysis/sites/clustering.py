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

    Floods the AGFE map as a height field from its local minima, separating
    touching pockets that connected-components would merge.

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum voxels a cluster must contain to be retained.
    min_distance : int
        Minimum separation, in voxels, between seed minima.
    compactness : float
        Forwarded to ``skimage.segmentation.watershed``; larger values give
        more compact, ball-shaped regions.
    """

    def __init__(self, min_cluster_voxels=10, min_distance=3, compactness=0.0):
        self.min_cluster_voxels = min_cluster_voxels
        self.min_distance = min_distance
        self.compactness = compactness

    def cluster(self, favorable_mask, agfe_array, gridsize):
        # Seeds are AGFE minima (most-negative = most favorable), i.e. maxima of -AGFE.
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
    min_distance : int
        Minimum separation, in voxels, between seed maxima.  ``"distance"``
        mode only.
    watershed_mode : {"score", "distance"}
        ``"score"``: watershed the clipped AGFE score ``clip(-agfe, 0, None)``,
        seeded by h_maxima.  ``"distance"``: watershed the Euclidean distance
        transform of the favorable mask, seeded by its local maxima, which
        splits blobs more evenly.
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


class DBSCANClustering:
    """Cluster favorable voxels with DBSCAN on their Angstrom coordinates.

    Purely spatial: ignores AGFE intensity and voxel adjacency rules, keying
    only on whether favorable voxels lie within ``eps_angstrom`` of each other.

    Parameters
    ----------
    min_cluster_voxels : int
        Minimum voxels a cluster must contain to be retained (DBSCAN
        ``min_samples``).
    eps_angstrom : float
        Neighbourhood radius in Angstroms; a small multiple of the grid spacing.
    """

    def __init__(self, min_cluster_voxels=10, eps_angstrom=1.5):
        self.min_cluster_voxels = min_cluster_voxels
        self.eps_angstrom = eps_angstrom

    def cluster(self, favorable_mask, agfe_array, gridsize):
        vox_coords = np.argwhere(favorable_mask).astype(float)
        if len(vox_coords) == 0:
            return np.zeros(agfe_array.shape, dtype=int), []
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


def build_clustering_strategy(clustering_cfg, gridsize=None):
    """Build a clustering-strategy instance from a ``ClusteringConfig``.

    Constructs the class named by ``clustering_cfg.strategy`` with
    ``min_cluster_voxels`` plus any ``strategy_kwargs``.  If *gridsize* is given
    and the config sets ``min_cluster_volume_ang3``, the voxel threshold is
    derived from that volume instead.

    Raises
    ------
    ValueError
        If ``clustering_cfg.strategy`` is not a known strategy name.
    """
    registry = {
        "skimage_watershed": SkimageWatershedClustering,
        "connected_components": ConnectedComponentsClustering,
        "watershed": WatershedClustering,
        "dbscan": DBSCANClustering,
    }
    strategy = clustering_cfg.strategy
    if strategy not in registry:
        raise ValueError(
            f"Unknown clustering strategy {strategy!r}; "
            f"valid options: {sorted(registry)}"
        )
    kwargs = dict(clustering_cfg.strategy_kwargs or {})
    n_vox = clustering_cfg.min_cluster_voxels
    resolve = getattr(clustering_cfg, "resolve_min_cluster_voxels", None)
    if resolve is not None and gridsize is not None:
        n_vox = resolve(gridsize)
    return registry[strategy](min_cluster_voxels=n_vox, **kwargs)
