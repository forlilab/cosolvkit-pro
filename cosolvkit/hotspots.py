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
from scipy.ndimage import label, center_of_mass
from scipy.spatial import cKDTree

try:
    from pymol import cmd as _pymol_cmd
    _PYMOL_AVAILABLE = True
except ImportError:
    _PYMOL_AVAILABLE = False


class BindingSite:
    """A binding hotspot detected from cosolvent AGFE density maps.

    Stores all computed scores and an extensible ``properties`` dict so that
    downstream analyses (e.g. residence time, pharmacophore annotation) can
    attach extra data without subclassing.

    Parameters are set by :class:`HotspotDetector` — do not construct directly.
    """

    def __init__(self, rank, site_id, cosolvent, n_voxels, centroid,
                 agfe_min, agfe_mean_top_pct, voxel_mask,
                 favorability_score, burial_score, diversity_score,
                 volume_score, composite_score,
                 favorable_atomtypes, per_type_agfe):
        self.rank = rank                            # int; 1 = highest composite score
        self.site_id = site_id                      # label from scipy.ndimage.label
        self.cosolvent = cosolvent                  # str residue name
        self.n_voxels = n_voxels                    # int
        self.centroid = centroid                    # np.ndarray (3,), Angstroms
        self.agfe_min = agfe_min                    # float, kcal/mol
        self.agfe_mean_top_pct = agfe_mean_top_pct  # float, kcal/mol
        self.voxel_mask = voxel_mask                # boolean 3D ndarray, same shape as AGFE grid
        self.favorability_score = favorability_score  # float [0, 1]
        self.burial_score = burial_score              # float [0, 1]
        self.diversity_score = diversity_score        # float [0, 1]
        self.volume_score = volume_score              # float [0, 1]
        self.composite_score = composite_score        # float, weighted sum
        self.favorable_atomtypes = favorable_atomtypes  # List[str]
        self.per_type_agfe = per_type_agfe            # Dict[str, float]: min AGFE per type
        self.properties = {}                          # extensible user properties

    def add_property(self, name, value):
        """Attach an arbitrary property (e.g. ``site.add_property('residence_time_ns', 12.4)``)."""
        self.properties[name] = value

    def to_dict(self):
        """Flat dict for CSV/JSON export. Includes base scores and ``properties``."""
        d = {
            "rank": self.rank,
            "site_id": self.site_id,
            "cosolvent": self.cosolvent,
            "n_voxels": self.n_voxels,
            "centroid_x": round(float(self.centroid[0]), 3),
            "centroid_y": round(float(self.centroid[1]), 3),
            "centroid_z": round(float(self.centroid[2]), 3),
            "agfe_min": round(float(self.agfe_min), 4),
            "agfe_mean_top_pct": round(float(self.agfe_mean_top_pct), 4),
            "favorability_score": round(float(self.favorability_score), 4),
            "burial_score": round(float(self.burial_score), 4),
            "diversity_score": round(float(self.diversity_score), 4),
            "volume_score": round(float(self.volume_score), 4),
            "composite_score": round(float(self.composite_score), 4),
            "favorable_atomtypes": ",".join(self.favorable_atomtypes),
        }
        d.update({f"agfe_{k}": round(float(v), 4) for k, v in self.per_type_agfe.items()})
        d.update(self.properties)
        return d

    def __repr__(self):
        return (
            f"BindingSite(rank={self.rank}, cosolvent={self.cosolvent!r}, "
            f"n_voxels={self.n_voxels}, agfe_min={self.agfe_min:.3f} kcal/mol, "
            f"composite={self.composite_score:.3f})"
        )


class HotspotDetector:
    """Detect and rank binding hotspots from cosolvent AGFE density maps.

    Reads the AGFE ``.dx`` maps produced by :meth:`Report.generate_density_maps`,
    clusters favorable voxels with connected-components analysis, scores each
    cluster on four independent axes, and exports ranked results.

    **Composite score** = w_fav × favorability + w_bur × burial
                        + w_div × diversity   + w_vol × volume

    Parameters
    ----------
    out_path : str
        Directory containing the ``.dx`` map files.
    cosolvent_names : list[str]
        Cosolvent residue names to analyse.
    universe : MDAnalysis.Universe
        Loaded trajectory universe (used for burial scoring fallback).
    agfe_cutoff : float
        AGFE threshold in kcal/mol (default -0.5).  Only voxels strictly below
        this value are considered favorable.
    min_cluster_voxels : int
        Minimum cluster size to retain (default 5).  Scale this with gridsize
        (e.g. use 3 for a 1.0 Å grid).
    burial_radius : float
        Radius in Angstroms for the burial-score sphere (default 6.0).
    top_percentile : float
        Top percentage of most-favorable voxels used for favorability scoring
        (default 10.0).
    score_weights : dict, optional
        Weights for composite score components.  Keys: ``favorability``,
        ``burial``, ``diversity``, ``volume``.  Will be normalised to sum 1.0.
    gridsize : float
        Voxel size in Angstroms (default 0.5).  Should match the value used
        in :meth:`Report.generate_density_maps`.
    """

    _DEFAULT_WEIGHTS = {
        "favorability": 0.5,
        "burial": 0.2,
        "diversity": 0.2,
        "volume": 0.1,
    }

    def __init__(self, out_path, cosolvent_names, universe,
                 agfe_cutoff=-0.5, min_cluster_voxels=5,
                 burial_radius=6.0, top_percentile=10.0,
                 score_weights=None, gridsize=0.5):
        self.logger = logging.getLogger(__name__)
        self.out_path = out_path
        self.cosolvent_names = cosolvent_names
        self.universe = universe
        self.agfe_cutoff = agfe_cutoff
        self.min_cluster_voxels = min_cluster_voxels
        self.burial_radius = burial_radius
        self.top_percentile = top_percentile
        self.gridsize = gridsize

        weights = dict(self._DEFAULT_WEIGHTS)
        if score_weights is not None:
            weights.update(score_weights)
        total = sum(weights.values())
        self.score_weights = {k: v / total for k, v in weights.items()}

        # Caches populated during detect() for use in export_results()
        self._labeled_arrays = {}
        self._combined_grids = {}

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_dx(self, filepath):
        return Grid(str(filepath))

    def _load_combined_agfe(self, cosolvent):
        """Load and combine per-atom-type AGFE maps into one grid.

        For ``use_atomtypes=True`` runs: takes the element-wise minimum across
        all per-type maps (most favorable signal at each voxel).  This finds
        any voxel favorable for *any* part of the cosolvent, which is the
        right signal for pocket detection.  Atom-type composition is then
        read back per-site for diversity scoring.

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

    def _load_accessible_mask(self, reference_shape):
        """Load ``solvent_accessible_map.dx`` (1 = accessible, 0 = buried).

        The file is written to CWD by :meth:`Analysis._build_accessible_mask`
        (not to out_path — a pre-existing path inconsistency).  Searches both
        locations.
        """
        search_paths = [
            os.path.join(self.out_path, "solvent_accessible_map.dx"),
            os.path.join(os.getcwd(), "solvent_accessible_map.dx"),
        ]
        for path in search_paths:
            if os.path.exists(path):
                grid = self._load_dx(path)
                mask = grid.grid
                if mask.shape != reference_shape:
                    self.logger.warning(
                        f"solvent_accessible_map.dx shape {mask.shape} differs from "
                        f"AGFE grid shape {reference_shape}. "
                        "Falling back to protein-atom distance for burial scoring."
                    )
                    return None
                return mask
        self.logger.warning(
            "solvent_accessible_map.dx not found in:\n"
            + "\n".join(f"  {p}" for p in search_paths)
            + "\nFalling back to protein-atom distance for burial scoring."
        )
        return None

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

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _burial_from_mask(self, accessible_mask, center_vox, r_vox, shape):
        """Burial score using the solvent-accessible mask.

        Returns the fraction of voxels inside a sphere of ``r_vox`` voxels
        around ``center_vox`` that are *not* accessible (buried in protein).
        """
        cv = np.array(center_vox)
        i0 = max(0, int(cv[0] - r_vox))
        i1 = min(shape[0], int(cv[0] + r_vox) + 1)
        j0 = max(0, int(cv[1] - r_vox))
        j1 = min(shape[1], int(cv[1] + r_vox) + 1)
        k0 = max(0, int(cv[2] - r_vox))
        k1 = min(shape[2], int(cv[2] + r_vox) + 1)

        ii, jj, kk = np.mgrid[i0:i1, j0:j1, k0:k1]
        in_sphere = (ii - cv[0])**2 + (jj - cv[1])**2 + (kk - cv[2])**2 <= r_vox**2
        if not in_sphere.any():
            return 0.0
        sub = accessible_mask[i0:i1, j0:j1, k0:k1]
        return float(1.0 - sub[in_sphere].mean())

    def _burial_from_protein(self, protein_tree, centroid_ang):
        """Burial score from nearest protein heavy-atom distance (fallback)."""
        dist, _ = protein_tree.query(centroid_ang.reshape(1, 3), k=1)
        d = float(dist[0])
        return float(1.0 / (1.0 + d / self.burial_radius))

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

        accessible_mask = self._load_accessible_mask(shape)
        r_vox = self.burial_radius / self.gridsize

        # Build protein KD-tree once for the fallback burial scorer
        protein_tree = None
        if accessible_mask is None:
            try:
                protein = self.universe.select_atoms("protein and not name H*")
                if len(protein) > 0:
                    protein_tree = cKDTree(protein.positions)
            except Exception as exc:
                self.logger.debug(f"Could not build protein KD-tree: {exc}")

        # --- Threshold ---
        favorable_mask = agfe_array < self.agfe_cutoff
        n_favorable = int(favorable_mask.sum())
        if n_favorable == 0:
            self.logger.warning(
                f"No favorable voxels for '{cosolvent}' at cutoff "
                f"{self.agfe_cutoff} kcal/mol. Try a less strict cutoff."
            )
            return []

        # --- Connected-components clustering (6-connectivity) ---
        labeled_array, n_raw_sites = label(favorable_mask)

        # --- Filter small clusters ---
        site_labels = [
            lbl for lbl in range(1, n_raw_sites + 1)
            if int((labeled_array == lbl).sum()) >= self.min_cluster_voxels
        ]
        if not site_labels:
            self.logger.warning(
                f"All clusters for '{cosolvent}' are smaller than "
                f"min_cluster_voxels={self.min_cluster_voxels}. "
                "Try reducing min_cluster_voxels."
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
        raw_f, raw_b, raw_d, raw_v = [], [], [], []
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

            # Burial
            if accessible_mask is not None:
                b_raw = self._burial_from_mask(
                    accessible_mask, np.array(com_vox), r_vox, shape
                )
            elif protein_tree is not None:
                b_raw = self._burial_from_protein(protein_tree, centroid_ang)
            else:
                b_raw = 0.0

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
            raw_b.append(b_raw)
            raw_d.append(d_raw)
            raw_v.append(n_vox)
            site_data.append({
                "lbl": lbl,
                "n_voxels": n_vox,
                "centroid_ang": centroid_ang,
                "agfe_min": float(np.min(voxel_agfe)),
                "agfe_mean_top_pct": f_raw,
                "burial": b_raw,
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
            + w["burial"] * np.array(raw_b)
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
                burial_score=float(raw_b[idx]),
                diversity_score=float(raw_d[idx]),
                volume_score=float(v_norm[idx]),
                composite_score=float(composite[idx]),
                favorable_atomtypes=sd["favorable_atomtypes"],
                per_type_agfe=sd["per_type_agfe"],
            ))

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

        Returns
        -------
        dict[str, list[BindingSite]]
            ``{cosolvent: [site, ...]}`` sorted by composite score per cosolvent.
        """
        return {cosolvent: self.detect(cosolvent) for cosolvent in self.cosolvent_names}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, results, label_map=True):
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
    # PyMol visualisation
    # ------------------------------------------------------------------

    def add_hotspots_to_pymol_session(self, results, pse_path, top_n=10):
        """Add hotspot pseudoatom spheres to an existing PyMol session file.

        The ``.pse`` file is overwritten in-place.  Pseudoatom commands are
        also appended to the ``.pml`` script (if it exists).

        Parameters
        ----------
        results : dict[str, list[BindingSite]]
        pse_path : str
            Path to existing ``.pse`` file.
        top_n : int
            Maximum sites per cosolvent to add (default 10).
        """
        if not _PYMOL_AVAILABLE:
            self.logger.warning(
                "PyMol is not available — skipping hotspot session update."
            )
            return

        _RANK_COLORS = {1: "yellow", 2: "orange", 3: "tv_red"}
        _DEFAULT_COLOR = "salmon"

        _pymol_cmd.load(pse_path)
        pml_lines = ["\n# Hotspot sites added by HotspotDetector\n"]

        for cosolvent, sites in results.items():
            group_members = []
            for site in sites[:top_n]:
                name = f"hotspot_{cosolvent}_rank{site.rank}"
                color = _RANK_COLORS.get(site.rank, _DEFAULT_COLOR)
                vdw = min(site.n_voxels / 50.0, 4.0)
                cx, cy, cz = float(site.centroid[0]), float(site.centroid[1]), float(site.centroid[2])

                _pymol_cmd.pseudoatom(name, pos=[cx, cy, cz], vdw=vdw)
                _pymol_cmd.color(color, name)
                _pymol_cmd.show("spheres", name)
                group_members.append(name)

                pml_lines.append(
                    f"pseudoatom {name}, pos=[{cx:.3f},{cy:.3f},{cz:.3f}], vdw={vdw:.2f}\n"
                    f"color {color}, {name}\n"
                    f"show spheres, {name}\n"
                )

            if group_members:
                group_name = f"hotspots_{cosolvent}"
                _pymol_cmd.group(group_name, " ".join(group_members))
                pml_lines.append(f"group {group_name}, {' '.join(group_members)}\n")

        _pymol_cmd.save(pse_path)
        self.logger.info(f"Updated PyMol session: {pse_path}")

        pml_path = pse_path.replace(".pse", ".pml")
        if os.path.exists(pml_path):
            with open(pml_path, "a") as fh:
                fh.writelines(pml_lines)
            self.logger.info(f"Appended hotspot commands to: {pml_path}")
