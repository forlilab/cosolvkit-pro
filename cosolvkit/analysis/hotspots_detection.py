#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Hotspot detection and ranking from cosolvent MD density maps
#

import os
import json
import glob as glob_module
import logging
import numpy as np
import pandas as pd
from gridData import Grid
from scipy.ndimage import center_of_mass

from .density_clustering import (
    ConnectedComponentsClustering,
    WatershedClustering,
    DBSCANClustering,
    SkimageWatershedClustering,
)
from . import hotspot_visualization as viz
from .pocket_properties import PocketPropertyCalculator, PocketResidue, compute_composite_score


from cosolvkit.analysis.core.models import BindingSite


class HotspotDetector:
    """Detect and rank binding hotspots from cosolvent AGFE density maps.

    Reads the AGFE ``.dx`` maps produced by :meth:`Report.generate_density_maps`,
    clusters favorable voxels using a pluggable clustering strategy, scores each
    cluster on four independent axes, and exports ranked results.

    **Composite score** = w_fav × favorability + w_div × diversity + w_vol × volume

    Parameters
    ----------
    out_path : str
        Directory containing the ``.dx`` map files.
    cosolvent_names : list[str]
        Cosolvent residue names to analyse.
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    agfe_cutoff : float
        AGFE threshold in kcal/mol (default -0.5).  Only voxels strictly below
        this value are considered favorable.
    min_cluster_voxels : int
        Minimum cluster size to retain (default 5).  Scale this with gridsize
        (e.g. use 3 for a 1.0 Å grid).
    top_percentile : float
        Top percentage of most-favorable voxels used for favorability scoring
        (default 10.0).
    score_weights : dict, optional
        Weights for composite score components.  Keys: ``favorability``,
        ``diversity``, ``volume``.  Will be normalised to sum 1.0.
    gridsize : float
        Voxel size in Angstroms (default 0.5).  Should match the value used
        in :meth:`Report.generate_density_maps`.
    clustering_strategy : ClusteringStrategy, optional
        Object with a ``cluster(favorable_mask, agfe_array, gridsize)`` method
        that returns ``(labeled_array, site_labels)``.  Defaults to
        :class:`ConnectedComponentsClustering` with 6-connectivity.
        Built-in strategies: :class:`ConnectedComponentsClustering`,
        :class:`WatershedClustering`, :class:`DBSCANClustering`.
    compute_survival_probability : bool
        If ``True``, :meth:`detect_all` runs survival probability analysis for
        all detected sites (default ``False``).
    survival_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :meth:`PocketPropertyCalculator.run_survival_probability`
        (e.g. ``radius``, ``max_tau``, ``intermittency``).
    """

    _DEFAULT_WEIGHTS = {
        "favorability": 0.5,
        "diversity": 0.3,
        "volume": 0.2,
    }

    def __init__(self, out_path, cosolvent_names, universe,
                 agfe_cutoff=-0.5, 
                 min_cluster_voxels=1,
                 top_percentile=10.0,
                 score_weights=None, gridsize=0.5,
                 clustering_strategy=None,
                 compute_survival_probability=False, survival_kwargs=None,
                 use_skimage_cleanup=False,
                 cleanup_min_size=1,
                 cleanup_hole_size=2,
                 cleanup_opening_radius=None,
                 cleanup_closing_radius=None,
                 compute_regionprops=True,
                 regionprops_properties=None,
                 regionprops_extra_properties=None,
                 ):
        self.logger = logging.getLogger(__name__)
        self._out_path = out_path
        self.cosolvent_names = cosolvent_names
        self._universe = universe
        self.agfe_cutoff = agfe_cutoff
        self.min_cluster_voxels = min_cluster_voxels
        self.top_percentile = top_percentile
        self.gridsize = gridsize
        self.clustering_strategy = (
            clustering_strategy
            if clustering_strategy is not None
            # else ConnectedComponentsClustering(min_cluster_voxels=min_cluster_voxels)
            else SkimageWatershedClustering(min_cluster_voxels=min_cluster_voxels,
                                            h=0.5)
        )
        self.compute_survival_probability = compute_survival_probability
        self.survival_kwargs = survival_kwargs or {}
        self.use_skimage_cleanup = use_skimage_cleanup
        self.cleanup_min_size = cleanup_min_size
        self.cleanup_hole_size = cleanup_hole_size
        self.cleanup_opening_radius = cleanup_opening_radius
        self.cleanup_closing_radius = cleanup_closing_radius
        self.compute_regionprops = compute_regionprops
        self.regionprops_properties = regionprops_properties
        self.regionprops_extra_properties = regionprops_extra_properties

        weights = dict(self._DEFAULT_WEIGHTS)
        if score_weights is not None:
            weights.update(score_weights)
        total = sum(weights.values())
        self.score_weights = {k: v / total for k, v in weights.items()}

        # Caches populated during detect() for use in export_results()
        self._labeled_arrays = {}
        self._combined_grids = {}

        self.property_calculator = PocketPropertyCalculator(
            out_path=self._out_path,
            universe=self._universe,
            gridsize=self.gridsize,
            regionprops_properties=self.regionprops_properties,
            regionprops_extra_properties=self.regionprops_extra_properties,
        )

    # ------------------------------------------------------------------
    # out_path and universe properties — setters propagate to property_calculator
    # so that multi_report.py can mutate them between SP calls without breaking
    # the calculator's reference.
    # ------------------------------------------------------------------

    @property
    def out_path(self):
        return self._out_path

    @out_path.setter
    def out_path(self, value):
        self._out_path = value
        if hasattr(self, "property_calculator"):
            self.property_calculator.out_path = value

    @property
    def universe(self):
        return self._universe

    @universe.setter
    def universe(self, value):
        self._universe = value
        if hasattr(self, "property_calculator"):
            self.property_calculator.universe = value

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_dx(self, filepath):
        return Grid(str(filepath))

    def _load_combined_agfe(self, cosolvent):
        """Load and combine per-atom-type AGFE maps into one grid.

        For ``use_atomtypes=True`` runs: takes the element-wise minimum across
        all per-type maps (most favorable signal at each voxel).  This finds
        any voxel favorable for *any* part of the cosolvent.
        Atom-type composition is then read back per-site for diversity scoring.

        Returns
        -------
        combined_grid : gridData.Grid
        per_type_grids : dict[str, gridData.Grid]
            Empty when only a single combined map exists (use_atomtypes=False).
        """
        pattern = os.path.join(self.out_path, f"map_agfe_*_{cosolvent}.dx")
        candidates = sorted(
            p for p in glob_module.glob(pattern)
            if "_raw_" not in os.path.basename(p)
        )

        if candidates:
            per_type = {}
            arrays = []
            first_grid = None
            for path in candidates:
                basename = os.path.basename(path)
                prefix = "map_agfe_"
                suffix = f"_{cosolvent}.dx"
                if basename.startswith(prefix) and basename.endswith(suffix):
                    atype = basename[len(prefix):-len(suffix)]
                else:
                    atype = basename
                g = self._load_dx(path)
                if first_grid is None:
                    first_grid = g
                per_type[atype] = g
                arrays.append(g.grid)

            combined_array = np.minimum.reduce(arrays)
            combined_grid = Grid(combined_array, first_grid.edges)
            return combined_grid, per_type

        single_path = os.path.join(self.out_path, f"map_agfe_{cosolvent}.dx")
        if os.path.exists(single_path):
            self.logger.info(
                f"No per-atom-type AGFE maps found for '{cosolvent}'; using combined map. "
                "Diversity score will be 0."
            )
            return self._load_dx(single_path), {}

        available = sorted(os.listdir(self.out_path))
        raise FileNotFoundError(
            f"No AGFE maps found for cosolvent '{cosolvent}' in {self.out_path!r}.\n"
            f"Available files: {available}"
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _voxel_to_angstrom(self, grid, vox_idx):
        """Convert fractional voxel index (i, j, k) to Angstrom coordinates.

        ``grid.origin`` is the centre of voxel 0 and ``grid.delta`` is a 1D
        array ``[dx, dy, dz]`` (gridData always stores delta as a 1D array).
        For fractional indices from ``center_of_mass`` this gives a continuous
        linear interpolation.
        """
        return np.array(grid.origin) + np.array(vox_idx) * np.array(grid.delta)

    def _cluster_voxels(self, favorable_mask, agfe_array):
        """Delegate clustering to ``self.clustering_strategy``.

        Returns
        -------
        labeled_array : np.ndarray of int
            3-D array; 0 = background, positive integers = cluster ids.
        site_labels : list[int]
            Cluster ids that survived the minimum-size filter.
        """
        return self.clustering_strategy.cluster(
            favorable_mask, agfe_array, self.gridsize
        )

    def _preprocess_favorable_mask(self, favorable_mask):
        """Optionally clean the favorable mask using scikit-image morphology.

        Only active when ``use_skimage_cleanup=True``.  Each step is
        individually gated by its corresponding parameter — set only the ones
        you need.

        Returns the (possibly modified) boolean mask.
        """
        if not self.use_skimage_cleanup:
            return favorable_mask

        from skimage.morphology import (
            remove_small_objects,
            remove_small_holes,
            binary_opening,
            binary_closing,
            ball,
        )

        mask = favorable_mask.copy()

        if self.cleanup_min_size is not None:
            mask = remove_small_objects(mask, max_size=self.cleanup_min_size)

        if self.cleanup_hole_size is not None:
            mask = remove_small_holes(mask, max_size=self.cleanup_hole_size)

        if self.cleanup_opening_radius is not None:
            mask = binary_opening(mask, footprint=ball(self.cleanup_opening_radius))

        if self.cleanup_closing_radius is not None:
            mask = binary_closing(mask, footprint=ball(self.cleanup_closing_radius))

        return mask

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, cosolvent):
        """Detect and score hotspots for one cosolvent.

        Returns a list of :class:`BindingSite` objects sorted by composite
        score (rank 1 = highest score).
        """
        self.logger.info(f"Detecting hotspots for {cosolvent}...")

        combined_grid, per_type_grids = self._load_combined_agfe(cosolvent)
        agfe_array = combined_grid.grid
        shape = agfe_array.shape

        # --- Threshold ---
        favorable_mask = agfe_array < self.agfe_cutoff
        n_favorable = int(favorable_mask.sum())
        if n_favorable == 0:
            self.logger.warning(
                f"No favorable voxels for '{cosolvent}' at cutoff "
                f"{self.agfe_cutoff} kcal/mol. Try a less strict cutoff."
            )
            return []

        # --- Optional mask preprocessing ---
        favorable_mask = self._preprocess_favorable_mask(favorable_mask)
        n_favorable = int(favorable_mask.sum())
        if n_favorable == 0:
            self.logger.warning(
                f"No favorable voxels remain for '{cosolvent}' after mask "
                "preprocessing. Try relaxing the cleanup parameters."
            )
            return []

        # --- Cluster favorable voxels ---
        labeled_array, site_labels = self._cluster_voxels(favorable_mask, agfe_array)

        if not site_labels:
            self.logger.warning(
                f"No clusters survived size filtering for '{cosolvent}'. "
                "Try reducing min_cluster_voxels or adjusting the clustering strategy."
            )
            return []

        # Sanity: warn if one giant cluster dominates
        largest = max(int((labeled_array == lbl).sum()) for lbl in site_labels)
        if largest > 0.5 * n_favorable:
            self.logger.warning(
                f"Largest cluster for '{cosolvent}' contains "
                f"{largest}/{n_favorable} favorable voxels (>50%). "
                "The map may be degenerate — consider a stricter agfe_cutoff."
            )

        # --- AGFE-weighted centroids (voxel space) ---
        # center_of_mass with a list index always returns a list of tuples
        coms = center_of_mass(np.abs(agfe_array), labeled_array, site_labels)

        # --- Compute raw scores ---
        raw_f, raw_d, raw_v = [], [], []
        site_data = []

        for lbl, com_vox in zip(site_labels, coms):
            site_mask = labeled_array == lbl
            voxel_agfe = agfe_array[site_mask]
            n_vox = int(site_mask.sum())

            # Favorability: mean of top-N% most-negative voxels
            n_top = max(1, int(n_vox * self.top_percentile / 100.0))
            f_raw = float(np.mean(np.sort(voxel_agfe)[:n_top]))

            # Centroid in Angstroms
            centroid_ang = self._voxel_to_angstrom(combined_grid, com_vox)

            # Diversity: fraction of atom types favorable at this site
            if per_type_grids:
                n_fav_types = sum(
                    1 for tg in per_type_grids.values()
                    if np.any(tg.grid[site_mask] < self.agfe_cutoff)
                )
                d_raw = float(n_fav_types) / len(per_type_grids)
            else:
                d_raw = 0.0

            # Per-type min AGFE (only types with at least one favorable voxel)
            per_type_agfe = {
                atype: float(np.min(tg.grid[site_mask]))
                for atype, tg in per_type_grids.items()
                if np.any(tg.grid[site_mask] < self.agfe_cutoff)
            }
            favorable_atomtypes = sorted(per_type_agfe.keys())

            raw_f.append(f_raw)
            raw_d.append(d_raw)
            raw_v.append(n_vox)
            site_data.append({
                "lbl": lbl,
                "n_voxels": n_vox,
                "centroid_ang": centroid_ang,
                "agfe_min": float(np.min(voxel_agfe)),
                "agfe_mean_top_pct": f_raw,
                "diversity": d_raw,
                "voxel_mask": site_mask,
                "favorable_atomtypes": favorable_atomtypes,
                "per_type_agfe": per_type_agfe,
            })

        # --- Normalise favorability and volume ---
        raw_f = np.array(raw_f)
        raw_v = np.array(raw_v, dtype=float)

        f_min, f_max = raw_f.min(), raw_f.max()
        if (f_max - f_min) < 1e-20:
            f_norm = np.ones_like(raw_f)
        else:
            # Most negative (f_min) → f_norm = 1 (best); least negative → 0
            f_norm = (f_max - raw_f) / (f_max - f_min)

        v_max = raw_v.max()
        v_norm = raw_v / v_max if v_max > 0 else np.zeros_like(raw_v)

        # --- Composite score ---
        w = self.score_weights
        composite = (
            w["favorability"] * f_norm
            + w["diversity"] * np.array(raw_d)
            + w["volume"] * v_norm
        )

        # --- Sort descending and build BindingSite objects ---
        order = np.argsort(-composite)
        sites = []
        for rank, idx in enumerate(order, start=1):
            sd = site_data[idx]
            sites.append(BindingSite(
                rank=rank,
                site_id=int(sd["lbl"]),
                cosolvent=cosolvent,
                n_voxels=sd["n_voxels"],
                centroid=sd["centroid_ang"],
                agfe_min=sd["agfe_min"],
                agfe_mean_top_pct=sd["agfe_mean_top_pct"],
                voxel_mask=sd["voxel_mask"],
                favorability_score=float(f_norm[idx]),
                diversity_score=float(raw_d[idx]),
                volume_score=float(v_norm[idx]),
                composite_score=float(composite[idx]),
                favorable_atomtypes=sd["favorable_atomtypes"],
                per_type_agfe=sd["per_type_agfe"],
            ))

        # --- Optional geometry descriptor extraction ---
        if self.compute_regionprops:
            score_image = np.clip(-agfe_array, 0, None)
            self.property_calculator.compute_regionprops(
                sites, labeled_array, score_image
            )

        # Attach grid spatial metadata so CrossProbeConsensusDetector can compute
        # Jaccard in Angstrom space when probes live on different-shaped grids.
        grid_origin = np.array(combined_grid.origin)
        grid_delta = np.array(combined_grid.delta)
        for site in sites:
            site.grid_origin = grid_origin
            site.grid_delta = grid_delta

        # Cache for export_results()
        self._labeled_arrays[cosolvent] = labeled_array
        self._combined_grids[cosolvent] = combined_grid

        self.logger.info(
            f"Found {len(sites)} hotspot(s) for '{cosolvent}'. "
            f"Top site: agfe_min={sites[0].agfe_min:.3f} kcal/mol, "
            f"composite={sites[0].composite_score:.3f}."
        )
        return sites

    def detect_all(self):
        """Run hotspot detection for all cosolvents.

        If ``compute_survival_probability=True``, runs survival probability
        analysis for all detected sites and attaches kinetic metrics to each
        :class:`BindingSite`.  If ``sp_*`` keys appear in ``score_weights``,
        the composite score is recomputed after SP metrics are attached.

        Returns
        -------
        dict[str, list[BindingSite]]
            ``{cosolvent: [site, ...]}`` sorted by composite score per cosolvent.
        """
        results = {cosolvent: self.detect(cosolvent) for cosolvent in self.cosolvent_names}

        if self.compute_survival_probability:
            for cosolvent, sites in results.items():
                if not sites:
                    continue
                candidate_zones = [
                    [float(v) for v in site.centroid] for site in sites
                ]
                self.logger.info(
                    f"Running survival probability for {len(sites)} "
                    f"site(s) of '{cosolvent}'."
                )
                self.property_calculator.run_survival_probability(
                    cosolvent_names=[cosolvent],
                    candidate_zones=candidate_zones,
                    **self.survival_kwargs,
                )
            self.property_calculator.fit_survival_probability(results)

            if any(k.startswith("sp_") for k in self.score_weights):
                for cosolvent_sites in results.values():
                    if cosolvent_sites:
                        compute_composite_score(cosolvent_sites, self.score_weights)

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, results, label_map=False):
        """Export hotspot results to CSV, JSON, and a label DX map.

        Parameters
        ----------
        results : dict[str, list[BindingSite]]
            Output of :meth:`detect_all`.
        label_map : bool
            If True, export ``hotspot_labels_{cosolvent}.dx`` where voxel
            value equals the site rank (0 = background, 1 = top site).
        """
        all_rows = []

        for cosolvent, sites in results.items():
            if not sites:
                self.logger.warning(f"No hotspots to export for '{cosolvent}'.")
                continue

            rows = [s.to_dict() for s in sites]
            df = pd.DataFrame(rows)

            csv_path = os.path.join(self.out_path, f"hotspot_sites_{cosolvent}.csv")
            json_path = os.path.join(self.out_path, f"hotspot_sites_{cosolvent}.json")
            df.to_csv(csv_path, index=False)
            with open(json_path, "w") as fh:
                json.dump(rows, fh, indent=2)

            self.logger.info(
                f"Exported {len(sites)} hotspot(s) for '{cosolvent}': "
                f"{csv_path}, {json_path}"
            )
            all_rows.extend(rows)

            if label_map and cosolvent in self._labeled_arrays:
                self._export_label_map(cosolvent, sites)

        if all_rows:
            all_df = (
                pd.DataFrame(all_rows)
                .sort_values("composite_score", ascending=False)
                .reset_index(drop=True)
            )
            tsv_path = os.path.join(self.out_path, "hotspot_sites_all.tsv")
            all_df.to_csv(tsv_path, sep="\t", index=False)
            self.logger.info(f"Exported combined hotspot table: {tsv_path}")

    def _export_label_map(self, cosolvent, sites):
        """Write a DX grid where voxel value = site rank (0 = background)."""
        labeled_array = self._labeled_arrays[cosolvent]
        combined_grid = self._combined_grids[cosolvent]

        rank_array = np.zeros_like(labeled_array, dtype=float)
        for site in sites:
            rank_array[labeled_array == site.site_id] = float(site.rank)

        Grid(rank_array, combined_grid.edges).export(
            os.path.join(self.out_path, f"hotspot_labels_{cosolvent}.dx")
        )
        self.logger.info(
            f"Exported label map: hotspot_labels_{cosolvent}.dx "
            "(voxel value = rank; isosurface at 0.5 shows all sites)"
        )

    # ------------------------------------------------------------------
    # Visualisation — thin wrappers; implementation in hotspot_visualization.py
    # ------------------------------------------------------------------

    def plot_hotspot_clustering_3d(self, cosolvent, sites, output_path=None,
                                   max_voxels_per_cluster=3000, top_n=10):
        """See :func:`hotspot_visualization.plot_hotspot_clustering_3d`.

        Requires :meth:`detect` to have been called for *cosolvent* so that
        the labeled array and combined grid are cached.

        Parameters
        ----------
        cosolvent : str
        sites : list[BindingSite]
            Output of :meth:`detect` for this cosolvent.
        output_path : str, optional
            If given, write an interactive HTML file to this path.
        max_voxels_per_cluster : int
            Subsampling cap per cluster (default 3000).
        top_n : int
            Maximum number of sites to plot, in rank order (default 10).
        """
        if cosolvent not in self._labeled_arrays:
            raise RuntimeError(
                f"No cached clustering for '{cosolvent}'. "
                "Call detect() first."
            )
        return viz.plot_hotspot_clustering_3d(
            labeled_array=self._labeled_arrays[cosolvent],
            agfe_array=self._combined_grids[cosolvent].grid,
            sites=sites,
            combined_grid=self._combined_grids[cosolvent],
            cosolvent=cosolvent,
            agfe_cutoff=self.agfe_cutoff,
            output_path=output_path,
            max_voxels_per_cluster=max_voxels_per_cluster,
            top_n=top_n,
        )

    def visualise_clustering(self, cosolvent, results=None, reference_pdb=None):
        """See :func:`hotspot_visualization.visualise_clustering`."""
        if cosolvent not in self._labeled_arrays or results is None:
            results = self.detect(cosolvent)
        return viz.visualise_clustering(
            cosolvent=cosolvent,
            labeled_array=self._labeled_arrays[cosolvent],
            combined_grid=self._combined_grids[cosolvent],
            results=results,
            out_path=self.out_path,
            voxel_to_angstrom_fn=self._voxel_to_angstrom,
            reference_pdb=reference_pdb,
        )

    def add_hotspots_to_pymol_session(self, results, pse_path, top_n=10):
        """See :func:`hotspot_visualization.add_hotspots_to_pymol_session`."""
        viz.add_hotspots_to_pymol_session(results, pse_path, self.out_path, top_n=top_n)

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(results, out_path):
        """Save hotspot detection results to compressed NPZ checkpoint files.

        One file per cosolvent is written to
        ``{out_path}/hotspot_checkpoints/hotspot_checkpoint_{cosolvent}.npz``.
        Each file stores the voxel masks (as a stacked boolean array), centroids,
        grid spatial metadata, and all scalar/string/list fields as JSON.

        The checkpoint can be reloaded with :meth:`load_checkpoint` to skip
        re-running the full hotspot detection step when only consensus
        parameters need to change.

        Parameters
        ----------
        results : dict[str, list[BindingSite]]
            Output of :meth:`detect_all`.
        out_path : str
            Directory where the ``hotspot_checkpoints/`` sub-directory will be
            created.
        """
        logger = logging.getLogger(__name__)
        chk_dir = os.path.join(out_path, "hotspot_checkpoints")
        os.makedirs(chk_dir, exist_ok=True)

        for cosolvent, sites in results.items():
            if not sites:
                logger.debug(f"No sites for '{cosolvent}' — skipping checkpoint.")
                continue

            voxel_masks = np.stack([s.voxel_mask for s in sites])  # (n, nx, ny, nz) bool
            centroids = np.array([s.centroid for s in sites], dtype=float)
            grid_origin = (
                np.asarray(sites[0].grid_origin, dtype=float)
                if sites[0].grid_origin is not None
                else np.zeros(3, dtype=float)
            )
            grid_delta = (
                np.asarray(sites[0].grid_delta, dtype=float)
                if sites[0].grid_delta is not None
                else np.zeros(3, dtype=float)
            )

            meta = []
            for s in sites:
                m = s.to_dict()
                m["_properties"] = s.properties
                meta.append(m)

            npz_path = os.path.join(chk_dir, f"hotspot_checkpoint_{cosolvent}.npz")
            np.savez_compressed(
                npz_path,
                voxel_masks=voxel_masks,
                centroids=centroids,
                grid_origin=grid_origin,
                grid_delta=grid_delta,
                metadata=np.array([json.dumps(meta)]),
            )
            logger.info(
                f"Saved hotspot checkpoint for '{cosolvent}': {npz_path} "
                f"({len(sites)} site(s))"
            )

    @staticmethod
    def load_checkpoint(out_path, cosolvent_names):
        """Load hotspot detection results from NPZ checkpoint files.

        Reconstructs :class:`BindingSite` objects (including ``voxel_mask``
        and grid metadata) previously saved by :meth:`save_checkpoint`.

        Parameters
        ----------
        out_path : str
            Directory that contains the ``hotspot_checkpoints/`` sub-directory.
        cosolvent_names : list[str]
            Cosolvents to load.  A :class:`FileNotFoundError` is raised if the
            checkpoint file for any requested cosolvent is missing.

        Returns
        -------
        dict[str, list[BindingSite]]
            Same structure as the output of :meth:`detect_all`.
        """
        logger = logging.getLogger(__name__)
        chk_dir = os.path.join(out_path, "hotspot_checkpoints")
        results = {}

        for cosolvent in cosolvent_names:
            npz_path = os.path.join(chk_dir, f"hotspot_checkpoint_{cosolvent}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Hotspot checkpoint not found for '{cosolvent}': {npz_path}\n"
                    "Run the full hotspot detection first (save_checkpoint=True) "
                    "before using load_checkpoint."
                )

            data = np.load(npz_path, allow_pickle=True)
            meta = json.loads(str(data["metadata"][0]))
            voxel_masks = data["voxel_masks"]
            grid_origin = data["grid_origin"]
            grid_delta = data["grid_delta"]

            sites = [
                BindingSite.from_dict(m, voxel_masks[i].astype(bool), grid_origin, grid_delta)
                for i, m in enumerate(meta)
            ]
            results[cosolvent] = sites
            logger.info(
                f"Loaded hotspot checkpoint for '{cosolvent}': {npz_path} "
                f"({len(sites)} site(s))"
            )

        return results
