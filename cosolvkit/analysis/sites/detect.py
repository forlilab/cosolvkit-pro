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
import warnings
import numpy as np
import pandas as pd
from gridData import Grid
from scipy.ndimage import center_of_mass

from cosolvkit.analysis.sites.clustering import build_clustering_strategy
from cosolvkit.analysis.sites.properties import PocketPropertyCalculator
from cosolvkit.analysis.core.models import Hotspot


class HotspotDetector:
    """Detect and rank binding hotspots from cosolvent AGFE density maps.

    Reads the AGFE ``.dx`` maps written by :meth:`Report.generate_density_maps`,
    clusters favorable voxels with a pluggable strategy, computes raw per-site
    features, and ranks the resulting :class:`Hotspot` objects by ``agfe_min``
    ascending (most-negative = rank 1).

    Parameters
    ----------
    out_path : str
        Directory containing the ``.dx`` map files.
    cosolvent_names : list[str]
        Cosolvent residue names to analyse.
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    agfe_cutoff : float
        AGFE threshold in kcal/mol; voxels strictly below it are favorable.
    min_cluster_volume_ang3 : float
        Minimum cluster size to retain, as a VOLUME. Grid-independent, which a voxel count is
        not. Default 20 A^3 = one heavy atom's van der Waals volume; see ClusteringConfig for
        the measured ground-truth-coverage tradeoff behind that number.
    min_cluster_voxels : int, optional
        Low-level override in raw voxels, for callers that need an exact count on a synthetic or
        non-standard grid (tests, sweeps). Deliberately NOT a ClusteringConfig field: at the
        config level a grid-dependent count is a foot-gun, and having both invites a precedence
        rule to get wrong.
    top_percentile : float
        Percentage of most-favorable voxels averaged for favorability scoring.
    gridsize : float
        Voxel size in Angstroms; must match the value used to generate the maps.
    clustering_strategy : ClusteringStrategy, optional
        Object with a ``cluster(favorable_mask, agfe_array, gridsize)`` method
        returning ``(labeled_array, site_labels)``.  See
        :mod:`cosolvkit.analysis.sites.clustering` for the built-in strategies.
    compute_survival_probability : bool
        If ``True``, :meth:`detect_all` also runs survival probability analysis
        for all detected sites.
    survival_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :meth:`PocketPropertyCalculator.run_survival_probability`.
    """

    def __init__(self, out_path, cosolvent_names, universe,
                 agfe_cutoff=-0.5,
                 min_cluster_volume_ang3=20.0,
                 min_cluster_voxels=None,
                 top_percentile=10.0,
                 gridsize=0.5,
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
        self.top_percentile = top_percentile
        self.gridsize = gridsize
        # One construction path. This used to hand-build the strategy here while
        # `build_clustering_strategy` built it from config elsewhere, so the two could drift.
        if clustering_strategy is not None:
            self.clustering_strategy = clustering_strategy
        else:
            from cosolvkit.analysis.config import ClusteringConfig
            self.clustering_strategy = build_clustering_strategy(
                ClusteringConfig(min_cluster_volume_ang3=min_cluster_volume_ang3),
                gridsize=self.gridsize,
            )
            if min_cluster_voxels is not None:
                self.clustering_strategy.min_cluster_voxels = int(min_cluster_voxels)
        # Read the threshold back off the strategy that will actually apply it. Keeping a separate
        # copy let the detector report one number while a supplied strategy filtered on another.
        self.min_cluster_voxels = getattr(self.clustering_strategy, "min_cluster_voxels", None)
        self.min_cluster_volume_ang3 = min_cluster_volume_ang3
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
    # Setters must propagate to property_calculator: callers may retarget the
    # detector between runs, and the calculator holds its own reference.
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

        Combines by element-wise minimum, so a voxel is favorable if it is
        favorable for any atom type.  Which types those were is recovered
        per-site later as ``favorable_atomtypes``.

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
        """Convert a (possibly fractional) voxel index (i, j, k) to Angstroms.

        ``grid.origin`` is the centre of voxel 0 and gridData stores
        ``grid.delta`` as the 1D array ``[dx, dy, dz]``.  Fractional indices,
        e.g. from ``center_of_mass``, interpolate linearly.
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
        """Optionally clean the favorable mask with scikit-image morphology.

        Inactive unless ``use_skimage_cleanup=True``; each step is gated on its
        own parameter being non-``None``.  Returns the boolean mask.
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

        Returns a list of :class:`Hotspot` objects ranked by ``agfe_min``
        ascending (rank 1 = most-negative / most favorable).
        """
        self.logger.info(f"Detecting hotspots for {cosolvent}...")

        combined_grid, per_type_grids = self._load_combined_agfe(cosolvent)
        agfe_array = combined_grid.grid
        shape = agfe_array.shape

        favorable_mask = agfe_array < self.agfe_cutoff
        n_favorable = int(favorable_mask.sum())
        if n_favorable == 0:
            self.logger.warning(
                f"No favorable voxels for '{cosolvent}' at cutoff "
                f"{self.agfe_cutoff} kcal/mol. Try a less strict cutoff."
            )
            return []

        favorable_mask = self._preprocess_favorable_mask(favorable_mask)
        n_favorable = int(favorable_mask.sum())
        if n_favorable == 0:
            self.logger.warning(
                f"No favorable voxels remain for '{cosolvent}' after mask "
                "preprocessing. Try relaxing the cleanup parameters."
            )
            return []

        labeled_array, site_labels = self._cluster_voxels(favorable_mask, agfe_array)

        if not site_labels:
            self.logger.warning(
                f"No clusters survived size filtering for '{cosolvent}'. "
                "Try lowering min_cluster_volume_ang3 or adjusting the clustering strategy."
            )
            return []

        # A single cluster swallowing the map means the threshold is too loose.
        largest = max(int((labeled_array == lbl).sum()) for lbl in site_labels)
        if largest > 0.5 * n_favorable:
            self.logger.warning(
                f"Largest cluster for '{cosolvent}' contains "
                f"{largest}/{n_favorable} favorable voxels (>50%). "
                "The map may be degenerate — consider a stricter agfe_cutoff."
            )

        # AGFE-weighted centroids, in voxel space.  Passing a label list makes
        # center_of_mass return a list of tuples even for a single label.
        coms = center_of_mass(np.abs(agfe_array), labeled_array, site_labels)

        # --- Compute raw site features ---
        site_data = []

        for lbl, com_vox in zip(site_labels, coms):
            site_mask = labeled_array == lbl
            voxel_agfe = agfe_array[site_mask]
            n_vox = int(site_mask.sum())

            # Favorability: mean of top-N% most-negative voxels
            n_top = max(1, int(n_vox * self.top_percentile / 100.0))
            f_raw = float(np.mean(np.sort(voxel_agfe)[:n_top]))

            centroid_ang = self._voxel_to_angstrom(combined_grid, com_vox)

            # Per-type min AGFE, keyed only on types with a favorable voxel here,
            # so the key set is itself the site's favorable atom types.
            per_type_agfe = {
                atype: float(np.min(tg.grid[site_mask]))
                for atype, tg in per_type_grids.items()
                if np.any(tg.grid[site_mask] < self.agfe_cutoff)
            }
            favorable_atomtypes = sorted(per_type_agfe.keys())

            site_data.append({
                "lbl": lbl,
                "n_voxels": n_vox,
                "centroid_ang": centroid_ang,
                "agfe_min": float(np.min(voxel_agfe)),
                "agfe_mean_top_pct": f_raw,
                "voxel_mask": site_mask,
                "favorable_atomtypes": favorable_atomtypes,
                "per_type_agfe": per_type_agfe,
            })

        # --- Rank by affinity: ascending agfe_min, most-negative = rank 1 ---
        agfe_mins = np.array([sd["agfe_min"] for sd in site_data], dtype=float)
        order = np.argsort(agfe_mins)
        sites = []
        for rank, idx in enumerate(order, start=1):
            sd = site_data[idx]
            sites.append(Hotspot(
                rank=rank,
                site_id=int(sd["lbl"]),
                cosolvent=cosolvent,
                n_voxels=sd["n_voxels"],
                centroid=sd["centroid_ang"],
                agfe_min=sd["agfe_min"],
                agfe_mean_top_pct=sd["agfe_mean_top_pct"],
                voxel_mask=sd["voxel_mask"],
                favorable_atomtypes=sd["favorable_atomtypes"],
                per_type_agfe=sd["per_type_agfe"],
            ))

        # --- Field descriptors, read from the map, not the thresholded blob ---
        from cosolvkit.analysis.core.field import (
            attach_accessible_fraction, attach_field_descriptors,
        )
        attach_field_descriptors(sites, agfe_array, combined_grid.origin,
                                 combined_grid.delta)
        # `accessible_fraction` is the enclosure feature (normalised, bounded, plateaus with
        # radius); it replaced `buriedness`, which was removed. It needs the accessible-volume
        # mask that GridAnalysis writes next to the AGFE maps. Attach it when that map is
        # present; stay silent when it is not, so a run that never generated one behaves
        # exactly as before.
        # `solvent_accessible_map.dx` is the replica-merged mask written by
        # MultiReport._merge_accessible_masks. A single-simulation run has no merged mask, only the
        # per-probe `solvent_accessible_map_<RES>.dx` that GridAnalysis wrote; combine those by the
        # same majority vote so both layouts behave identically.
        merged_mask = os.path.join(self.out_path, "solvent_accessible_map.dx")
        mask_paths = ([merged_mask] if os.path.isfile(merged_mask)
                      else sorted(glob_module.glob(os.path.join(self.out_path,
                                                    "solvent_accessible_map_*.dx"))))
        if mask_paths:
            try:
                from cosolvkit.analysis.core.grid import combine_accessible_masks
                if len(mask_paths) == 1:
                    mg = self._load_dx(mask_paths[0])
                    mask = np.asarray(mg.grid) > 0.5
                    origin = np.asarray(mg.origin, dtype=float)
                    delta = np.asarray(mg.delta, dtype=float)
                else:
                    mask = combine_accessible_masks(mask_paths)
                    ref = self._load_dx(max(mask_paths,
                                            key=lambda p: self._load_dx(p).grid.size))
                    origin = np.asarray(ref.origin, dtype=float)
                    delta = np.asarray(ref.delta, dtype=float)
                attach_accessible_fraction(sites, mask, origin, delta)
            except Exception as exc:
                warnings.warn(
                    f"could not attach accessible_fraction from {mask_paths}: {exc}",
                    RuntimeWarning)
        else:
            # Loud, because `accessible_fraction` carries a non-zero default weight: a run without
            # the mask scores every site as if the enclosure term did not exist.
            warnings.warn(
                f"no solvent_accessible_map*.dx in {self.out_path}; `accessible_fraction` will be "
                "absent and its default weight will have no effect. It is written by GridAnalysis "
                "when constructed with out_dir=<map directory>.", RuntimeWarning)

        if self.compute_regionprops:
            score_image = np.clip(-agfe_array, 0, None)
            self.property_calculator.compute_regionprops(
                sites, labeled_array, score_image
            )

        # Grid metadata lets downstream grouping compute overlap in Angstrom
        # space, which is required when probes live on different-shaped grids.
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
            f"Top site: agfe_min={sites[0].agfe_min:.3f} kcal/mol."
        )
        return sites

    def detect_all(self):
        """Run hotspot detection for all cosolvents.

        With ``compute_survival_probability=True``, also attaches kinetic
        metrics to each :class:`Hotspot`.

        Returns
        -------
        dict[str, list[Hotspot]]
            ``{cosolvent: [site, ...]}`` ranked by ``agfe_min`` per cosolvent.
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

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, results, label_map=False):
        """Export hotspot results to CSV, JSON, and a label DX map.

        Parameters
        ----------
        results : dict[str, list[Hotspot]]
            Output of :meth:`detect_all`.
        label_map : bool
            If True, export ``hotspot_labels_{cosolvent}.dx`` where voxel
            value equals the site rank (0 = background, 1 = top site).
        """
        GEOM_PREFIX = "geom_"
        all_rows = []

        for cosolvent, sites in results.items():
            if not sites:
                self.logger.warning(f"No hotspots to export for '{cosolvent}'.")
                continue

            rows = [s.to_dict() for s in sites]
            df = pd.DataFrame(rows)

            geom_cols = [c for c in df.columns if c.startswith(GEOM_PREFIX)]
            main_df = df.drop(columns=geom_cols)

            csv_path = os.path.join(self.out_path, f"hotspot_sites_{cosolvent}.csv")
            json_path = os.path.join(self.out_path, f"hotspot_sites_{cosolvent}.json")
            main_df.to_csv(csv_path, index=False)
            with open(json_path, "w") as fh:
                json.dump(rows, fh, indent=2)            # JSON keeps the full record

            if geom_cols:
                geom_path = os.path.join(self.out_path, f"hotspot_sites_geom_{cosolvent}.csv")
                df[["site_id"] + geom_cols].to_csv(geom_path, index=False)
                self.logger.info(f"Exported geometry sidecar: {geom_path}")

            self.logger.info(
                f"Exported {len(sites)} hotspot(s) for '{cosolvent}': "
                f"{csv_path}, {json_path}"
            )
            all_rows.extend(rows)

            if label_map and cosolvent in self._labeled_arrays:
                self._export_label_map(cosolvent, sites)

        if all_rows:
            all_df = pd.DataFrame(all_rows)
            all_df = all_df.drop(columns=[c for c in all_df.columns if c.startswith(GEOM_PREFIX)])
            all_df = all_df.sort_values("agfe_min", ascending=True).reset_index(drop=True)
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

        Requires a prior :meth:`detect` call for *cosolvent* to populate the
        labeled-array and grid caches.

        Parameters
        ----------
        cosolvent : str
        sites : list[Hotspot]
            Output of :meth:`detect` for this cosolvent.
        output_path : str, optional
            If given, write an interactive HTML file here.
        max_voxels_per_cluster : int
            Subsampling cap per cluster.
        top_n : int
            Maximum number of sites to plot, in rank order.
        """
        if cosolvent not in self._labeled_arrays:
            raise RuntimeError(
                f"No cached clustering for '{cosolvent}'. "
                "Call detect() first."
            )
        from cosolvkit.analysis import hotspot_visualization as viz
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
        from cosolvkit.analysis import hotspot_visualization as viz
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
        from cosolvkit.analysis import hotspot_visualization as viz
        viz.add_hotspots_to_pymol_session(results, pse_path, self.out_path, top_n=top_n)

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(results, out_path):
        """Save hotspot detection results to compressed NPZ checkpoint files.

        Writes one file per cosolvent to
        ``{out_path}/hotspot_checkpoints/hotspot_checkpoint_{cosolvent}.npz``
        holding stacked voxel masks, centroids, grid metadata, and the
        remaining fields as JSON.  Reload with :meth:`load_checkpoint` to
        re-run downstream steps without repeating detection.

        Parameters
        ----------
        results : dict[str, list[Hotspot]]
            Output of :meth:`detect_all`.
        out_path : str
            Directory in which ``hotspot_checkpoints/`` is created.
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

        Rebuilds the :class:`Hotspot` objects, voxel masks and grid metadata
        written by :meth:`save_checkpoint`.

        Parameters
        ----------
        out_path : str
            Directory containing ``hotspot_checkpoints/``.
        cosolvent_names : list[str]
            Cosolvents to load; a missing checkpoint raises
            :class:`FileNotFoundError`.

        Returns
        -------
        dict[str, list[Hotspot]]
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
                Hotspot.from_dict(m, voxel_masks[i].astype(bool), grid_origin, grid_delta)
                for i, m in enumerate(meta)
            ]
            results[cosolvent] = sites
            logger.info(
                f"Loaded hotspot checkpoint for '{cosolvent}': {npz_path} "
                f"({len(sites)} site(s))"
            )

        return results


# ---------------------------------------------------------------------------
# Module-level aliases for the static methods (for back-compat shim export)
# ---------------------------------------------------------------------------
save_checkpoint = HotspotDetector.save_checkpoint
load_checkpoint = HotspotDetector.load_checkpoint
