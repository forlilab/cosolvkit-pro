#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Cross-probe consensus hotspot detection
#

import os
import json
import logging
from functools import reduce

import numpy as np
import pandas as pd


class ConsensusSite:
    """A consensus binding site formed by overlapping hotspots from multiple probes.

    Created by :class:`CrossProbeConsensusDetector` from the output of
    :meth:`HotspotDetector.detect_all`. Do not construct directly.

    Attributes
    ----------
    consensus_rank : int
        1 = highest consensus_score.
    community_id : int
        Internal community index from the overlap graph.
    member_sites : list[BindingSite]
        All per-probe BindingSite objects belonging to this community.
    member_cosolvents : list[str]
        Unique cosolvent names that contribute at least one site.
    n_probes : int
        Number of distinct probes in this community.
    total_probes : int
        Total number of probes analysed (denominator for probe_coverage).
    probe_coverage : float
        ``n_probes / total_probes`` in [0, 1].
    consensus_centroid : np.ndarray (3,)
        AGFE-weighted mean of member centroids, in Angstroms.
    union_voxel_count : int
        Number of voxels in the union of all member voxel masks.
    min_agfe : float
        Most favourable AGFE across all member sites, in kcal/mol.
    mean_agfe : float
        Mean of per-member ``agfe_min`` values, in kcal/mol.
    pharmacophore : dict[str, dict[str, float]]
        Nested pharmacophore profile: ``{cosolvent: {atomtype: min_agfe}}``.
        Only contains probes with per-atom-type AGFE maps.
    favorable_atomtypes_union : set[str]
        Union of ``favorable_atomtypes`` across all member sites.
    consensus_score : float
        Weighted combination of probe_coverage, normalised favourability,
        and normalised union volume.
    """

    def __init__(self, consensus_rank, community_id, member_sites,
                 member_cosolvents, n_probes, total_probes, probe_coverage,
                 consensus_centroid, union_voxel_count,
                 min_agfe, mean_agfe,
                 pharmacophore, favorable_atomtypes_union,
                 consensus_score):
        self.consensus_rank = consensus_rank
        self.community_id = community_id
        self.member_sites = member_sites
        self.member_cosolvents = member_cosolvents
        self.n_probes = n_probes
        self.total_probes = total_probes
        self.probe_coverage = probe_coverage
        self.consensus_centroid = consensus_centroid
        self.union_voxel_count = union_voxel_count
        self.min_agfe = min_agfe
        self.mean_agfe = mean_agfe
        self.pharmacophore = pharmacophore
        self.favorable_atomtypes_union = favorable_atomtypes_union
        self.consensus_score = consensus_score

    def to_dict(self):
        """Flat dict suitable for CSV/JSON export."""
        d = {
            "consensus_rank": self.consensus_rank,
            "community_id": self.community_id,
            "n_probes": self.n_probes,
            "total_probes": self.total_probes,
            "probe_coverage": round(float(self.probe_coverage), 4),
            "member_cosolvents": ",".join(self.member_cosolvents),
            "consensus_centroid_x": round(float(self.consensus_centroid[0]), 3),
            "consensus_centroid_y": round(float(self.consensus_centroid[1]), 3),
            "consensus_centroid_z": round(float(self.consensus_centroid[2]), 3),
            "union_voxel_count": self.union_voxel_count,
            "min_agfe": round(float(self.min_agfe), 4),
            "mean_agfe": round(float(self.mean_agfe), 4),
            "favorable_atomtypes_union": ",".join(sorted(self.favorable_atomtypes_union)),
            "consensus_score": round(float(self.consensus_score), 4),
        }
        for site in self.member_sites:
            prefix = f"probe_{site.cosolvent}"
            d[f"{prefix}_rank"] = site.rank
            d[f"{prefix}_agfe_min"] = round(float(site.agfe_min), 4)
            d[f"{prefix}_composite_score"] = round(float(site.composite_score), 4)
        return d

    def __repr__(self):
        return (
            f"ConsensusSite(rank={self.consensus_rank}, "
            f"probes={self.member_cosolvents}, "
            f"coverage={self.probe_coverage:.2f}, "
            f"min_agfe={self.min_agfe:.3f} kcal/mol, "
            f"score={self.consensus_score:.3f})"
        )


class CrossProbeConsensusDetector:
    """Detect consensus binding sites from overlapping per-probe hotspots.

    Takes the ``Dict[str, List[BindingSite]]`` output of
    :meth:`HotspotDetector.detect_all` and groups sites from different
    cosolvents that share favorable voxels into communities. Each community
    becomes a :class:`ConsensusSite` with a pharmacophore profile describing
    which atom types from which probes are favorable.

    The algorithm:

    1. Build an overlap graph: nodes = (cosolvent, site) pairs; edges added
       when ``Jaccard(mask_i, mask_j) >= jaccard_threshold`` for sites from
       **different** cosolvents.  Bounding-box pre-filtering avoids O(N²)
       Jaccard computations for spatially distant pairs.
    2. Run community detection on the graph (connected components by default).
    3. Score each community on probe coverage, aggregate AGFE, and union volume.

    Parameters
    ----------
    probe_results : dict[str, list[BindingSite]]
        Output of :meth:`HotspotDetector.detect_all`.
    jaccard_threshold : float
        Minimum Jaccard similarity to add an edge (default 0.05).  Lower values
        merge sites that barely touch; higher values require strong overlap.
    community_method : str
        ``'connected_components'`` (default, no extra deps) or
        ``'greedy_modularity'`` (finer splitting, requires networkx ≥ 2.6).
    score_weights : dict, optional
        Keys: ``coverage``, ``favorability``, ``volume``.  Normalised to sum 1.
        Default: ``{coverage: 0.4, favorability: 0.4, volume: 0.2}``.
    """

    _DEFAULT_WEIGHTS = {"coverage": 0.4, "favorability": 0.4, "volume": 0.2}

    def __init__(self, probe_results, jaccard_threshold=0.05,
                 community_method="connected_components",
                 score_weights=None):
        self.logger = logging.getLogger(__name__)
        self.probe_results = probe_results
        self.total_probes = len(probe_results)
        self.jaccard_threshold = jaccard_threshold
        self.community_method = community_method

        weights = dict(self._DEFAULT_WEIGHTS)
        if score_weights is not None:
            weights.update(score_weights)
        total = sum(weights.values())
        self.score_weights = {k: v / total for k, v in weights.items()}

        # Flat list of (cosolvent, BindingSite) for indexed graph nodes
        self._all_sites = [
            (cosolvent, site)
            for cosolvent, sites in probe_results.items()
            for site in sites
        ]

        self.logger.info(
            f"CrossProbeConsensusDetector: {self.total_probes} probe(s), "
            f"{len(self._all_sites)} total site(s), "
            f"jaccard_threshold={jaccard_threshold}, method={community_method}"
        )

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _compute_jaccard(self, site_a, site_b):
        """Jaccard similarity of two sites' voxel masks.

        When both sites share the same grid shape, does a direct boolean
        comparison.  When shapes differ (e.g. probes from different simulations
        with different box sizes), resamples ``site_b``'s mask onto ``site_a``'s
        coordinate system using nearest-neighbour interpolation via
        ``scipy.ndimage.map_coordinates``, which requires ``grid_origin`` and
        ``grid_delta`` to be set on both sites (done by
        :meth:`HotspotDetector.detect`).
        """
        mask_a = site_a.voxel_mask
        mask_b = site_b.voxel_mask

        if mask_a.shape == mask_b.shape:
            intersection = int(np.logical_and(mask_a, mask_b).sum())
            if intersection == 0:
                return 0.0
            union = int(np.logical_or(mask_a, mask_b).sum())
            return intersection / union

        # Different shapes — need Angstrom-space mapping.
        origin_a = getattr(site_a, "grid_origin", None)
        delta_a  = getattr(site_a, "grid_delta",  None)
        origin_b = getattr(site_b, "grid_origin", None)
        delta_b  = getattr(site_b, "grid_delta",  None)

        if any(x is None for x in (origin_a, delta_a, origin_b, delta_b)):
            # No grid metadata available — fall back to zoom approximation.
            self.logger.debug(
                "Grid metadata missing for Jaccard; using zoom approximation "
                f"({mask_a.shape} vs {mask_b.shape})"
            )
            from scipy.ndimage import zoom
            factors = tuple(a / b for a, b in zip(mask_a.shape, mask_b.shape))
            mask_b_r = zoom(mask_b.astype(np.float32), factors, order=1) >= 0.5
            # Trim or pad each axis to match mask_a exactly
            slices = tuple(slice(0, min(s, mask_a.shape[i])) for i, s in enumerate(mask_b_r.shape))
            mask_b_r = mask_b_r[slices]
            pad = [(0, max(0, mask_a.shape[i] - mask_b_r.shape[i])) for i in range(3)]
            mask_b_r = np.pad(mask_b_r, pad)
        else:
            # Sample mask_b at the Angstrom positions of mask_a's voxels.
            from scipy.ndimage import map_coordinates
            i_a, j_a, k_a = np.mgrid[
                0:mask_a.shape[0], 0:mask_a.shape[1], 0:mask_a.shape[2]
            ]
            # Angstrom positions of mask_a voxels
            pos_ang = origin_a + np.stack([i_a, j_a, k_a], axis=-1) * delta_a
            # Fractional indices in mask_b's grid
            frac_b = (pos_ang - origin_b) / delta_b
            coords_b = [frac_b[..., i].ravel() for i in range(3)]
            mask_b_r = (
                map_coordinates(
                    mask_b.astype(np.float32), coords_b,
                    order=0, mode="constant", cval=0.0,
                ).reshape(mask_a.shape) >= 0.5
            )

        intersection = int(np.logical_and(mask_a, mask_b_r).sum())
        if intersection == 0:
            return 0.0
        union = int(np.logical_or(mask_a, mask_b_r).sum())
        return intersection / union

    def _bboxes_overlap(self, site_a, site_b):
        """Return False when the sites' spatial bounding boxes are disjoint.

        Uses Angstrom-space comparison when both sites have ``grid_origin`` /
        ``grid_delta`` metadata (converts voxel bbox to Angstroms).  Falls
        back to True (always compute Jaccard) when the metadata is absent or
        the grids have different shapes without metadata.
        """
        bbox_a = site_a.properties.get("geom_bbox")
        bbox_b = site_b.properties.get("geom_bbox")
        if bbox_a is None or bbox_b is None:
            return True

        origin_a = getattr(site_a, "grid_origin", None)
        delta_a  = getattr(site_a, "grid_delta",  None)
        origin_b = getattr(site_b, "grid_origin", None)
        delta_b  = getattr(site_b, "grid_delta",  None)

        if any(x is None for x in (origin_a, delta_a, origin_b, delta_b)):
            return True  # can't compare reliably — let Jaccard decide

        # Convert voxel bbox to Angstrom: [min0,min1,min2,max0,max1,max2]
        lo_a = origin_a + np.array(bbox_a[:3]) * delta_a
        hi_a = origin_a + np.array(bbox_a[3:]) * delta_a
        lo_b = origin_b + np.array(bbox_b[:3]) * delta_b
        hi_b = origin_b + np.array(bbox_b[3:]) * delta_b

        for i in range(3):
            if hi_a[i] <= lo_b[i] or hi_b[i] <= lo_a[i]:
                return False
        return True

    def _resample_mask_to(self, site_src, ref_shape, ref_origin, ref_delta):
        """Resample ``site_src.voxel_mask`` onto a reference grid.

        Uses nearest-neighbour ``map_coordinates`` when grid metadata is
        available, otherwise falls back to ``zoom``.

        Returns a boolean array with shape ``ref_shape``.
        """
        mask_src = site_src.voxel_mask
        if mask_src.shape == ref_shape:
            return mask_src

        origin_src = getattr(site_src, "grid_origin", None)
        delta_src  = getattr(site_src, "grid_delta",  None)

        if origin_src is not None and delta_src is not None:
            from scipy.ndimage import map_coordinates
            i_r, j_r, k_r = np.mgrid[0:ref_shape[0], 0:ref_shape[1], 0:ref_shape[2]]
            pos_ang = ref_origin + np.stack([i_r, j_r, k_r], axis=-1) * ref_delta
            frac_src = (pos_ang - origin_src) / delta_src
            coords = [frac_src[..., i].ravel() for i in range(3)]
            return (
                map_coordinates(
                    mask_src.astype(np.float32), coords,
                    order=0, mode="constant", cval=0.0,
                ).reshape(ref_shape) >= 0.5
            )

        # Fallback: zoom approximation
        from scipy.ndimage import zoom
        factors = tuple(r / s for r, s in zip(ref_shape, mask_src.shape))
        resampled = zoom(mask_src.astype(np.float32), factors, order=1) >= 0.5
        slices = tuple(slice(0, min(resampled.shape[i], ref_shape[i])) for i in range(3))
        resampled = resampled[slices]
        pad = [(0, max(0, ref_shape[i] - resampled.shape[i])) for i in range(3)]
        return np.pad(resampled, pad)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_overlap_graph(self):
        """Build the site-overlap graph.

        Returns
        -------
        graph : networkx.Graph
            Nodes are integer indices into ``self._all_sites``.  Edges carry
            a ``weight`` equal to the Jaccard similarity.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for consensus detection. "
                "Install with: pip install networkx"
            )

        n = len(self._all_sites)
        graph = nx.Graph()
        for i in range(n):
            cosolvent, site = self._all_sites[i]
            graph.add_node(i, cosolvent=cosolvent, site=site)

        n_edges = 0
        n_compared = 0
        n_skipped_bbox = 0

        for i in range(n):
            cosolvent_i, site_i = self._all_sites[i]
            for j in range(i + 1, n):
                cosolvent_j, site_j = self._all_sites[j]

                # No intra-probe edges
                if cosolvent_i == cosolvent_j:
                    continue

                # Bounding-box pre-filter
                if not self._bboxes_overlap(site_i, site_j):
                    n_skipped_bbox += 1
                    continue

                n_compared += 1
                jaccard = self._compute_jaccard(site_i, site_j)
                if jaccard >= self.jaccard_threshold:
                    graph.add_edge(i, j, weight=jaccard)
                    n_edges += 1

        self.logger.info(
            f"Overlap graph: {n} nodes, {n_edges} edges "
            f"(compared {n_compared} pairs, {n_skipped_bbox} skipped by bbox)"
        )
        return graph

    # ------------------------------------------------------------------
    # Community detection and scoring
    # ------------------------------------------------------------------

    def detect_communities(self):
        """Run community detection and return ranked :class:`ConsensusSite` list.

        Returns
        -------
        list[ConsensusSite]
            Sorted by ``consensus_score`` descending; rank 1 = highest.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for consensus detection. "
                "Install with: pip install networkx"
            )

        graph = self.build_overlap_graph()

        if self.community_method == "greedy_modularity":
            from networkx.community import greedy_modularity_communities
            raw_communities = list(greedy_modularity_communities(graph))
        else:
            raw_communities = list(nx.connected_components(graph))

        self.logger.info(
            f"Community detection ({self.community_method}): "
            f"{len(raw_communities)} communities"
        )

        raw_data = []
        for comm_id, node_set in enumerate(raw_communities):
            node_set = list(node_set)
            members = [self._all_sites[i][1] for i in node_set]
            member_cosolvents = sorted({self._all_sites[i][0] for i in node_set})

            n_probes = len(member_cosolvents)
            probe_coverage = n_probes / self.total_probes

            # AGFE-weighted consensus centroid (more negative = higher weight)
            w_arr = np.array([max(-s.agfe_min, 1e-6) for s in members])
            consensus_centroid = np.average(
                [s.centroid for s in members], axis=0, weights=w_arr
            )

            # Union of voxel masks — resample onto the first member's grid
            ref = members[0]
            ref_shape  = ref.voxel_mask.shape
            ref_origin = getattr(ref, "grid_origin", None)
            ref_delta  = getattr(ref, "grid_delta",  None)
            resampled = [
                self._resample_mask_to(s, ref_shape, ref_origin, ref_delta)
                for s in members
            ]
            union_mask = reduce(np.logical_or, resampled)
            union_voxel_count = int(union_mask.sum())

            # Aggregate AGFE
            min_agfe = float(min(s.agfe_min for s in members))
            mean_agfe = float(np.mean([s.agfe_min for s in members]))

            # Pharmacophore: best (most negative) per-type AGFE per probe
            pharmacophore = {}
            for i in node_set:
                cosolvent, site = self._all_sites[i]
                if not site.per_type_agfe:
                    continue
                if cosolvent not in pharmacophore:
                    pharmacophore[cosolvent] = {}
                for atype, val in site.per_type_agfe.items():
                    if atype not in pharmacophore[cosolvent] or val < pharmacophore[cosolvent][atype]:
                        pharmacophore[cosolvent][atype] = round(float(val), 4)

            favorable_atomtypes_union = set()
            for site in members:
                favorable_atomtypes_union.update(site.favorable_atomtypes)

            raw_data.append({
                "community_id": comm_id,
                "members": members,
                "member_cosolvents": member_cosolvents,
                "n_probes": n_probes,
                "probe_coverage": probe_coverage,
                "consensus_centroid": consensus_centroid,
                "union_voxel_count": union_voxel_count,
                "min_agfe": min_agfe,
                "mean_agfe": mean_agfe,
                "pharmacophore": pharmacophore,
                "favorable_atomtypes_union": favorable_atomtypes_union,
            })

        # Normalise favorability and volume for scoring
        mean_agfe_arr = np.array([d["mean_agfe"] for d in raw_data])
        vol_arr = np.array([d["union_voxel_count"] for d in raw_data], dtype=float)

        f_min, f_max = mean_agfe_arr.min(), mean_agfe_arr.max()
        if (f_max - f_min) < 1e-20:
            f_norm_arr = np.ones(len(raw_data))
        else:
            # Most negative → 1 (best); least negative → 0
            f_norm_arr = (f_max - mean_agfe_arr) / (f_max - f_min)

        v_max = vol_arr.max()
        v_norm_arr = vol_arr / v_max if v_max > 0 else np.zeros(len(raw_data))

        w = self.score_weights
        scores = (
            w["coverage"] * np.array([d["probe_coverage"] for d in raw_data])
            + w["favorability"] * f_norm_arr
            + w["volume"] * v_norm_arr
        )

        order = np.argsort(-scores)
        consensus_sites = []
        for rank, idx in enumerate(order, start=1):
            d = raw_data[idx]
            consensus_sites.append(ConsensusSite(
                consensus_rank=rank,
                community_id=d["community_id"],
                member_sites=d["members"],
                member_cosolvents=d["member_cosolvents"],
                n_probes=d["n_probes"],
                total_probes=self.total_probes,
                probe_coverage=d["probe_coverage"],
                consensus_centroid=d["consensus_centroid"],
                union_voxel_count=d["union_voxel_count"],
                min_agfe=d["min_agfe"],
                mean_agfe=d["mean_agfe"],
                pharmacophore=d["pharmacophore"],
                favorable_atomtypes_union=d["favorable_atomtypes_union"],
                consensus_score=float(scores[idx]),
            ))

        if consensus_sites:
            top = consensus_sites[0]
            self.logger.info(
                f"Top consensus site: {top.n_probes} probe(s) {top.member_cosolvents}, "
                f"coverage={top.probe_coverage:.2f}, "
                f"min_agfe={top.min_agfe:.3f} kcal/mol, "
                f"score={top.consensus_score:.3f}"
            )

        return consensus_sites

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, sites, out_path):
        """Write consensus results to CSV, JSON, and pharmacophore JSON.

        Files written to ``out_path``:

        - ``consensus_sites.csv`` — flat table, one row per consensus site
        - ``consensus_sites.json`` — same content in JSON format
        - ``consensus_sites_pharmacophore.json`` — nested pharmacophore profile
          ``{rank, cosolvents, {cosolvent: {atomtype: min_agfe}}}``

        Parameters
        ----------
        sites : list[ConsensusSite]
        out_path : str
            Output directory (created if needed).
        """
        os.makedirs(out_path, exist_ok=True)
        rows = [s.to_dict() for s in sites]

        csv_path = os.path.join(out_path, "consensus_sites.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        self.logger.info(f"Consensus CSV written: {csv_path}")

        json_path = os.path.join(out_path, "consensus_sites.json")
        with open(json_path, "w") as fh:
            json.dump(rows, fh, indent=2)
        self.logger.info(f"Consensus JSON written: {json_path}")

        pharma_data = [
            {
                "consensus_rank": s.consensus_rank,
                "community_id": s.community_id,
                "member_cosolvents": s.member_cosolvents,
                "probe_coverage": round(float(s.probe_coverage), 4),
                "favorable_atomtypes_union": sorted(s.favorable_atomtypes_union),
                "pharmacophore": s.pharmacophore,
            }
            for s in sites
        ]
        pharma_path = os.path.join(out_path, "consensus_sites_pharmacophore.json")
        with open(pharma_path, "w") as fh:
            json.dump(pharma_data, fh, indent=2)
        self.logger.info(f"Pharmacophore JSON written: {pharma_path}")
