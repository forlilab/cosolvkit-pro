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
import matplotlib.pyplot as plt
import seaborn as sns
from gridData import Grid
from scipy.ndimage import label, center_of_mass

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
                 favorability_score, diversity_score,
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
    top_n_survival : int
        Number of top-ranked hotspots (per cosolvent) for which survival
        probability analysis is run automatically by :meth:`detect_all`.
        ``0`` (default) disables automatic survival probability analysis.
    survival_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`survival_probability`
        (e.g. ``radius``, ``max_tau``, ``intermittency``).
    """

    _DEFAULT_WEIGHTS = {
        "favorability": 0.5,
        "diversity": 0.2,
        "volume": 0.1,
        "sp_mrt": 0.2
    }

    def __init__(self, out_path, cosolvent_names, universe,
                 agfe_cutoff=-0.5, min_cluster_voxels=5,
                 top_percentile=10.0,
                 score_weights=None, gridsize=0.5,
                 top_n_survival=10, survival_kwargs=None):
        self.logger = logging.getLogger(__name__)
        self.out_path = out_path
        self.cosolvent_names = cosolvent_names
        self.universe = universe
        self.agfe_cutoff = agfe_cutoff
        self.min_cluster_voxels = min_cluster_voxels
        self.top_percentile = top_percentile
        self.gridsize = gridsize
        self.top_n_survival = top_n_survival
        self.survival_kwargs = survival_kwargs or {}

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

        If ``top_n_survival > 0``, automatically runs :meth:`survival_probability`
        for the top ``top_n_survival`` hotspots (by composite score) per cosolvent
        and then :meth:`fit_survival_probability` to attach kinetic metrics to each
        :class:`BindingSite`.

        Returns
        -------
        dict[str, list[BindingSite]]
            ``{cosolvent: [site, ...]}`` sorted by composite score per cosolvent.
        """
        results = {cosolvent: self.detect(cosolvent) for cosolvent in self.cosolvent_names}

        if self.top_n_survival > 0:
            candidate_zones = {}
            for cosolvent, sites in results.items():
                top_sites = sites[:self.top_n_survival]
                if not top_sites:
                    continue
                candidate_zones[cosolvent] = [
                    list(float(v) for v in site.centroid) for site in top_sites
                ]
                self.logger.info(
                    f"Running survival probability for top {len(top_sites)} "
                    f"hotspot(s) of '{cosolvent}'."
                )
                self.survival_probability(
                    cosolvent_names=[cosolvent],
                    candidate_zones=candidate_zones[cosolvent],
                    **self.survival_kwargs,
                )
            self.fit_survival_probability(results)

        return results

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
    # Survival probability
    # ------------------------------------------------------------------

    def survival_probability(self,
                             cosolvent_names: list = None,
                             candidate_zones: list = None,
                             radius: float = 6.0,
                             max_tau: int = 100,
                             intermittency: int = 2):
        """Compute the survival probability of cosolvents inside spherical zones.

        Each zone can be defined either as a group of residue IDs *or* as an
        explicit XYZ coordinate (an arbitrary point in space).  The two forms
        can be mixed freely within the same ``candidate_zones`` list.

        **Zone formats** — each element of ``candidate_zones`` is one zone:

        * ``[resid1, resid2, ...]`` / ``(resid1, resid2, ...)``  — sphere centred
          at the COM of the listed residues (original behaviour).
        * ``[x, y, z]`` / ``(x, y, z)``  — sphere centred at the explicit
          Angstrom coordinate.  Detected when the group contains exactly 3
          float-like values.
        * A bare ``int`` — treated as ``[resid]``.

        Results are saved as ``survival_probability_{cosolvent}.csv`` and
        ``survival_probability_{cosolvent}.png`` under ``self.out_path``.

        Parameters
        ----------
        cosolvent_names : list[str], optional
            Cosolvent residue names to analyse.  Defaults to all cosolvents.
        candidate_zones : list, required
            Zones to analyse (see format description above).
        radius : float
            Sphere radius in Angstroms (default 5.0).
        max_tau : int
            Maximum lag time for the survival-probability calculation (default 100).
        intermittency : int
            Intermittency for the waterdynamics SurvivalProbability (default 1).

        More info: https://www.mdanalysis.org/waterdynamics/api.html#waterdynamics.SurvivalProbability
        """
        try:
            from waterdynamics import SurvivalProbability as SP
        except ImportError:
            raise ImportError(
                "waterdynamics package is required for survival probability analysis. "
                "Please install it."
            )

        if candidate_zones is None:
            raise ValueError("candidate_zones must be provided.")
        if cosolvent_names is None:
            self.logger.warning(
                "No cosolvent specified for survival probability analysis. "
                "Using all cosolvents..."
            )
            cosolvent_names = self.cosolvent_names

        def _is_xyz(group):
            """Return True if group encodes an XYZ point (3 float-like values)."""
            return (
                len(group) == 3
                and all(isinstance(v, float) for v in group)
            )

        def _build_selection(cosolvent_name, group):
            if isinstance(group, int):
                return (
                    f"resname {cosolvent_name} and sphzone {radius} resid {group}",
                    str(group),
                )
            group = list(group)
            if _is_xyz(group):
                x, y, z = group
                return (
                    f"resname {cosolvent_name} and point {x} {y} {z} {radius}",
                    f"({x:.2f}, {y:.2f}, {z:.2f})",
                )
            # residue-ID group
            resids = " or ".join(f"resid {r}" for r in group)
            return (
                f"resname {cosolvent_name} and sphzone {radius} ({resids})",
                " ".join(str(r) for r in group),
            )

        for cosolvent_name in cosolvent_names:
            data = []
            zone_labels = []

            for zone_idx, zone in enumerate(candidate_zones):
                select, label_str = _build_selection(cosolvent_name, zone)
                zone_labels.append(label_str)
                self.logger.info(
                    f"Zone {zone_idx} [{label_str}] — cosolvent {cosolvent_name}"
                )

                sp = SP(self.universe, select, verbose=True)
                sp.run(tau_max=max_tau, residues=False, intermittency=intermittency)

                for tau, sp_value in zip(sp.tau_timeseries, sp.sp_timeseries):
                    data.append({
                        "Group": zone_idx,
                        "Zone": label_str,
                        "Time": tau,
                        "SP": sp_value,
                        "Cosolvent": cosolvent_name,
                    })

            df_sp = pd.DataFrame(data)
            df_sp.to_csv(
                os.path.join(self.out_path, f"survival_probability_{cosolvent_name}.csv"),
                index=False,
            )

            self._plot_sp_raw(cosolvent_name, df_sp)

    def _plot_sp_raw(self, cosolvent_name, df_sp):
        """Plot raw SP curves with hotspot rank as legend labels."""
        n_groups = df_sp["Group"].nunique()
        palette = sns.color_palette("flare", n_colors=max(n_groups, 1))
        fig, ax = plt.subplots()
        for zone_idx, group_df in df_sp.groupby("Group"):
            rank = int(zone_idx) + 1
            ax.plot(group_df["Time"], group_df["SP"],
                    label=f"Rank {rank}", color=palette[int(zone_idx)])
        ax.set_xlabel("Lag time (frames)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"{cosolvent_name} — Survival Probability")
        ax.legend(title="Hotspot")
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.out_path, f"survival_probability_{cosolvent_name}.png")
        )
        plt.close(fig)

    def _plot_sp_fits(self, cosolvent, sites, df):
        """Overlay fitted decay curves on SP data — one figure per model.

        Writes ``survival_probability_fit_{model}_{cosolvent}.png`` for each
        of the three models: single-exp, bi-exponential, and KWW.
        """
        def _single_exp(t, tau):
            return np.exp(-t / tau)

        def _bi_exp(t, A, tau1, tau2):
            return A * np.exp(-t / tau1) + (1.0 - A) * np.exp(-t / tau2)

        def _kww(t, tau, beta):
            return np.exp(-(t / tau) ** beta)

        site_by_rank = {site.rank: site for site in sites}
        n_groups = df["Group"].nunique()
        palette = sns.color_palette("flare", n_colors=max(n_groups, 1))

        models = [
            (
                "single", "Single-exponential",
                _single_exp,
                lambda p: (p.get("sp_tau_single"),),
                lambda p: f"τ={p['sp_tau_single']:.1f}, R²={p.get('sp_r2_single', 0):.3f}",
            ),
            (
                "biexp", "Bi-exponential",
                _bi_exp,
                lambda p: (p.get("sp_amplitude_fast"), p.get("sp_tau_fast"), p.get("sp_tau_slow")),
                lambda p: (
                    f"A={p['sp_amplitude_fast']:.2f}, "
                    f"τ_fast={p['sp_tau_fast']:.1f}, "
                    f"τ_slow={p['sp_tau_slow']:.1f}, "
                    f"R²={p.get('sp_r2_biexp', 0):.3f}"
                ),
            ),
            (
                "kww", "KWW (stretched-exp)",
                _kww,
                lambda p: (p.get("sp_tau_kww"), p.get("sp_beta_kww")),
                lambda p: (
                    f"τ={p['sp_tau_kww']:.1f}, "
                    f"β={p['sp_beta_kww']:.3f}, "
                    f"R²={p.get('sp_r2_kww', 0):.3f}"
                ),
            ),
        ]

        for model_key, model_title, model_fn, param_getter, label_fn in models:
            fig, ax = plt.subplots()
            for zone_idx, group_df in df.groupby("Group"):
                rank = int(zone_idx) + 1
                site = site_by_rank.get(rank)
                color = palette[int(zone_idx)]
                tau_arr = group_df["Time"].values.astype(float)
                sp_arr = group_df["SP"].values.astype(float)

                ax.scatter(tau_arr, sp_arr, color=color, s=10, alpha=0.5, zorder=2)

                if site is not None:
                    params = param_getter(site.properties)
                    if all(v is not None for v in params):
                        t_fine = np.linspace(tau_arr[0], tau_arr[-1], 300)
                        ax.plot(
                            t_fine, model_fn(t_fine, *params),
                            color=color,
                            label=f"Rank {rank} — {label_fn(site.properties)}",
                        )
                    else:
                        ax.plot([], [], color=color, label=f"Rank {rank} (fit failed)")

            ax.set_xlabel("Lag time (frames)")
            ax.set_ylabel("Survival Probability")
            ax.set_title(f"{cosolvent} — {model_title} fit")
            ax.legend(title="Hotspot", fontsize="small")
            fig.tight_layout()
            out = os.path.join(
                self.out_path,
                f"survival_probability_fit_{model_key}_{cosolvent}.png",
            )
            fig.savefig(out)
            plt.close(fig)
            self.logger.info(f"Saved {model_title} fit plot: {os.path.basename(out)}")

    def fit_survival_probability(self, results, zone_to_site_rank=None):
        """Fit SP decay curves and store kinetic metrics in each :class:`BindingSite`.

        Reads the ``survival_probability_{cosolvent}.csv`` files written by
        :meth:`survival_probability`, fits three decay models to each zone's
        curve, and stores the derived metrics in ``BindingSite.properties``
        via :meth:`BindingSite.add_property`.

        **Stored properties** (prefixed ``sp_``):

        * ``sp_mrt``            — mean residence time (trapezoid integral of SP)
        * ``sp_half_life``      — time at SP = 0.5
        * ``sp_plateau``        — mean SP over the last 10 % of timepoints
        * ``sp_tau_single``     — single-exponential time constant τ
        * ``sp_r2_single``      — R² of single-exp fit
        * ``sp_amplitude_fast`` — fraction in the fast population (bi-exp A)
        * ``sp_tau_fast``       — fast time constant τ₁ (bi-exp)
        * ``sp_tau_slow``       — slow time constant τ₂ (bi-exp)
        * ``sp_r2_biexp``       — R² of bi-exponential fit
        * ``sp_tau_kww``        — KWW (stretched-exp) characteristic time τ_c
        * ``sp_beta_kww``       — KWW stretching exponent β (1 = simple exp, <1 = heterogeneous)
        * ``sp_r2_kww``         — R² of KWW fit

        Parameters
        ----------
        results : dict[str, list[BindingSite]]
            Output of :meth:`detect_all`.
        zone_to_site_rank : dict[int, int], optional
            Maps zone index (``Group`` column in CSV) to site rank.
            If *None*, zone 0 → rank 1, zone 1 → rank 2, etc.
        """
        from scipy.optimize import curve_fit
        from scipy.interpolate import interp1d

        def _single_exp(t, tau):
            return np.exp(-t / tau)

        def _bi_exp(t, A, tau1, tau2):
            return A * np.exp(-t / tau1) + (1.0 - A) * np.exp(-t / tau2)

        def _kww(t, tau, beta):
            return np.exp(-(t / tau) ** beta)

        def _r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else 0.0

        for cosolvent, sites in results.items():
            csv_path = os.path.join(
                self.out_path, f"survival_probability_{cosolvent}.csv"
            )
            if not os.path.exists(csv_path):
                self.logger.warning(
                    f"No SP CSV found for '{cosolvent}': {csv_path}. "
                    "Run survival_probability() first."
                )
                continue

            df = pd.read_csv(csv_path)
            site_by_rank = {site.rank: site for site in sites}

            for zone_idx, group_df in df.groupby("Group"):
                rank = (
                    zone_to_site_rank.get(int(zone_idx))
                    if zone_to_site_rank is not None
                    else int(zone_idx) + 1
                )
                site = site_by_rank.get(rank)
                if site is None:
                    self.logger.debug(
                        f"Zone {zone_idx} → rank {rank}: no matching site, skipping."
                    )
                    continue

                tau_arr = group_df["Time"].values.astype(float)
                sp_arr = group_df["SP"].values.astype(float)

                if len(tau_arr) < 3:
                    continue

                props = {}

                # MRT — trapezoidal integral
                props["sp_mrt"] = round(float(np.trapz(sp_arr, tau_arr)), 4)

                # Half-life — interpolate SP = 0.5
                try:
                    f_interp = interp1d(
                        sp_arr[::-1], tau_arr[::-1],
                        bounds_error=False, fill_value=np.nan
                    )
                    hl = float(f_interp(0.5))
                    props["sp_half_life"] = round(hl, 4) if np.isfinite(hl) else None
                except Exception:
                    props["sp_half_life"] = None

                # Late-time plateau (mean of last 10 % of timepoints)
                n_tail = max(1, len(sp_arr) // 10)
                props["sp_plateau"] = round(float(np.mean(sp_arr[-n_tail:])), 4)

                # Single-exponential fit
                try:
                    p0 = [max(props["sp_mrt"], 1.0)]
                    popt, _ = curve_fit(
                        _single_exp, tau_arr, sp_arr,
                        p0=p0, bounds=(0, np.inf), maxfev=5000
                    )
                    props["sp_tau_single"] = round(float(popt[0]), 4)
                    props["sp_r2_single"] = round(
                        _r2(sp_arr, _single_exp(tau_arr, *popt)), 4
                    )
                except Exception as exc:
                    self.logger.debug(f"Single-exp fit failed (zone {zone_idx}): {exc}")

                # Bi-exponential fit (requires at least 6 points)
                if len(tau_arr) >= 6:
                    try:
                        mrt = props["sp_mrt"]
                        p0 = [0.5, max(mrt * 0.1, 1.0), max(mrt, 1.0)]
                        popt, _ = curve_fit(
                            _bi_exp, tau_arr, sp_arr, p0=p0,
                            bounds=([0, 0, 0], [1, np.inf, np.inf]),
                            maxfev=10000
                        )
                        A, tau1, tau2 = float(popt[0]), float(popt[1]), float(popt[2])
                        if tau1 > tau2:          # enforce fast < slow convention
                            A, tau1, tau2 = 1.0 - A, tau2, tau1
                        props["sp_amplitude_fast"] = round(A, 4)
                        props["sp_tau_fast"] = round(tau1, 4)
                        props["sp_tau_slow"] = round(tau2, 4)
                        props["sp_r2_biexp"] = round(
                            _r2(sp_arr, _bi_exp(tau_arr, *popt)), 4
                        )
                    except Exception as exc:
                        self.logger.debug(f"Bi-exp fit failed (zone {zone_idx}): {exc}")

                # KWW (stretched-exponential) fit
                try:
                    mrt = props["sp_mrt"]
                    p0 = [max(mrt, 1.0), 1.0]
                    popt, _ = curve_fit(
                        _kww, tau_arr, sp_arr, p0=p0,
                        bounds=([0, 0.1], [np.inf, 2.0]),
                        maxfev=5000
                    )
                    props["sp_tau_kww"] = round(float(popt[0]), 4)
                    props["sp_beta_kww"] = round(float(popt[1]), 4)
                    props["sp_r2_kww"] = round(
                        _r2(sp_arr, _kww(tau_arr, *popt)), 4
                    )
                except Exception as exc:
                    self.logger.debug(f"KWW fit failed (zone {zone_idx}): {exc}")

                for k, v in props.items():
                    site.add_property(k, v)

                self.logger.info(
                    f"Site rank {rank} ({cosolvent}): "
                    f"MRT={props['sp_mrt']:.2f}, "
                    f"plateau={props['sp_plateau']:.3f}, "
                    f"τ_single={props.get('sp_tau_single', 'N/A')}, "
                    f"β_kww={props.get('sp_beta_kww', 'N/A')}"
                )

            self._plot_sp_fits(cosolvent, sites, df)

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
