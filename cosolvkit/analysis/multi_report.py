#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Multi-trajectory analysis orchestrator
#

import os
import re
import logging
from glob import glob
from typing import Dict, List, Optional

from cosolvkit.analysis.report import Report
from cosolvkit.analysis.config import resolve_agfe_cutoff
from cosolvkit.analysis.sites.clustering import build_clustering_strategy
from .analysis_config import AnalysisConfig, SimulationEntry
from .density_analysis import combine_dx_maps_with_resampling
from .hotspots_detection import HotspotDetector


DEFAULT_PLOT_TOP_N = 10


def _sp_candidate_zones(sites, sp_top_n, zones=None):
    """Zones to profile kinetics in, as ``[[x, y, z], ...]``.

    Defaults to the top ``sp_top_n`` hotspot centroids. Pass *zones* to profile SUPPLIED
    points instead.

    Why supplying points matters: hotspot centroids are thresholded-and-clustered objects.
    Benchmarked against crystallographic ligand positions they sit a median 1.4 A (max 3.8 A)
    from the ligand they represent, and only 9 of 38 land within 1 A. Since the zone radius is
    only a few Angstrom, that offset is large enough to change the answer — profiling the same
    trajectories at ligand-centred points instead moved ``sp_half_life`` from AUC ~0.5 with
    between-replica rho ~0.15 to AUC 0.90 with rho 0.66. Kinetics measured at centroids is not
    the same measurement.
    """
    if zones is not None:
        return [[float(v) for v in z] for z in zones]
    return [[float(v) for v in site.centroid] for site in sites[:sp_top_n]]


def _load_zones_csv(path):
    """Read supplied kinetics zones from a CSV with x/y/z columns (case-insensitive)."""
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    missing = [a for a in ("x", "y", "z") if a not in cols]
    if missing:
        raise ValueError(
            f"{path}: zones CSV needs x, y, z columns; missing {missing} "
            f"(found {list(df.columns)})")
    return df[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=float).tolist()


def _zone_to_site_rank(zones, sites, max_dist_ang):
    """Map each supplied zone to the RANK of the nearest hotspot within *max_dist_ang*.

    Needed because ``fit_survival_probability`` otherwise attaches zone *i*'s kinetics to the
    site of rank *i+1* — correct for centroid-derived zones, but arbitrary for supplied ones.
    Zones with no hotspot in range are left unmapped, so their curves are still written to CSV
    but no hotspot receives misattributed ``sp_*`` values.
    """
    import numpy as np

    if not sites:
        return {}
    cents = np.asarray([s.centroid for s in sites], dtype=float)
    ranks = [s.rank for s in sites]
    out = {}
    for i, z in enumerate(zones):
        d = np.sqrt(((cents - np.asarray(z, dtype=float)) ** 2).sum(axis=1))
        j = int(np.argmin(d))
        if d[j] <= max_dist_ang:
            out[i] = ranks[j]
    return out


class MultiReport:
    """Orchestrate analysis of one or more cosolvent MD simulations.

    Takes an :class:`~cosolvkit.analysis_config.AnalysisConfig` (loaded from a
    YAML file) and runs the full pipeline:

    1. Per-simulation: structural report + density-map generation, written to
       ``out_path/<label>/`` subdirectories.
    2. Merge: AGFE ``.dx`` maps from different simulations are resampled onto a
       common grid and combined into ``out_path/merged/``.
    3. Joint hotspot detection on the merged maps using the protein reference
       from the first simulation (or an explicit ``reference_pdb``).
    4. PyMol session generation pointing at the merged maps.

    Each step is a separate public method so they can be called independently
    when partial re-runs are needed.

    Parameters
    ----------
    config : AnalysisConfig
    sp_zones : list of [x, y, z], optional
        Explicit points at which to profile kinetics, overriding the default of hotspot
        centroids (and overriding ``survival_kwargs.zones_csv`` if both are given). Use this
        to measure residence at externally-defined locations — crystallographic ligand
        positions, docking poses, a grid — rather than at cluster centroids, which are a
        median 1.4 A away from the ligands they represent. See :func:`_sp_candidate_zones`.
    """

    def __init__(self, config: AnalysisConfig, sp_zones=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.out_path = config.out_path
        self.sp_zones = (None if sp_zones is None
                         else [[float(v) for v in z] for z in sp_zones])
        os.makedirs(self.out_path, exist_ok=True)

        self._reports: List[Report] = []
        self._merged_dir: Optional[str] = None
        self._reference_pdb: Optional[str] = None
        self._binding_sites: List = []

    # ------------------------------------------------------------------
    # Step 1 — per-simulation processing
    # ------------------------------------------------------------------

    def run_per_simulation(self):
        """Build a :class:`Report` per simulation and run structural analysis
        and density-map generation, writing results to per-simulation subdirs."""
        cfg = self.config

        for i, sim in enumerate(cfg.simulations):
            label = sim.label or f"sim_{i}"
            sim_out = os.path.join(self.out_path, label)
            os.makedirs(sim_out, exist_ok=True)

            self.logger.info(f"Processing simulation '{label}' → {sim_out}")

            report = Report(
                traj_file=sim.trajectory,
                top_file=sim.topology,
                cosolvent_names=sim.cosolvents,
                out_path=sim_out,
            )

            report.generate_report(
                rmsf=cfg.report.rmsf,
                rdf=cfg.report.rdf,
                rmsf_avg_selec=cfg.report.rmsf_avg_selec,
                align_selection=cfg.report.align_selection,
            )

            report.generate_density_maps(
                cosolvent_names=sim.cosolvents,
                use_atomtypes=cfg.density_maps.use_atomtypes,
                atomtypes_definitions=cfg.density_maps.atomtypes_file,
                gridsize=cfg.density_maps.gridsize,
                temperature=cfg.density_maps.temperature,
            )

            self._reports.append(report)

        # Determine protein reference for hotspot detection
        if self._reports:
            self._reference_pdb = (
                cfg.report.reference_pdb
                or self._reports[0].avg_pdb_path
            )
            if not os.path.exists(self._reference_pdb):
                self.logger.warning(
                    f"Reference PDB not found at '{self._reference_pdb}'. "
                    "Hotspot visualisation may be incomplete. "
                    "Set 'rmsf: true' in report config or provide an explicit "
                    "'report.reference_pdb'."
                )

    # ------------------------------------------------------------------
    # Step 2 — merge density maps
    # ------------------------------------------------------------------

    def merge_density_maps(self):
        """Merge per-simulation AGFE maps into ``out_path/merged/``.

        For each cosolvent that appears in only one simulation the maps are
        copied/linked without aggregation.  For cosolvents that appear in
        multiple simulations, ``combine_dx_maps_with_resampling`` is used to
        resample all maps to a common grid and combine them.
        """
        merged_dir = os.path.join(self.out_path, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        self._merged_dir = merged_dir

        cfg = self.config
        merge_cfg = cfg.misc

        all_cosolvents = _collect_all_cosolvents(cfg.simulations)

        for cosolvent in all_cosolvents:
            dx_paths = _find_dx_paths_for_cosolvent(
                cosolvent, self._reports, cfg.density_maps.use_atomtypes
            )

            if not dx_paths:
                self.logger.warning(
                    f"No .dx maps found for cosolvent '{cosolvent}' — skipping merge."
                )
                continue

            # Group by atom-type key (e.g. 'HBD', 'HBA', 'Car', or 'total')
            groups = _group_dx_by_atomtype(cosolvent, dx_paths)

            for group_key, paths in groups.items():
                if group_key == "total":
                    out_fname = os.path.join(merged_dir, f"map_agfe_{cosolvent}.dx")
                else:
                    out_fname = os.path.join(merged_dir, f"map_agfe_{group_key}_{cosolvent}.dx")

                if len(paths) == 1:
                    # Only one simulation contributes this map — no aggregation needed
                    import shutil
                    shutil.copy(paths[0], out_fname)
                    self.logger.info(
                        f"Cosolvent '{cosolvent}' ({group_key}): single-sim map "
                        f"copied to {out_fname}"
                    )
                else:
                    self.logger.info(
                        f"Merging {len(paths)} maps for '{cosolvent}' ({group_key}) "
                        f"using method='{merge_cfg.merge_method}', "
                        f"resample_to='{merge_cfg.merge_resampling_to}' → {out_fname}"
                    )
                    combine_dx_maps_with_resampling(
                        filepaths=paths,
                        method=merge_cfg.merge_method,
                        resample_to=merge_cfg.merge_resampling_to,
                        out_fname=out_fname,
                    )

    # ------------------------------------------------------------------
    # Step 3 — joint hotspot detection
    # ------------------------------------------------------------------

    def run_joint_hotspot_detection(self) -> dict:
        """Run :class:`HotspotDetector` on merged maps.

        Uses the protein reference from the first simulation unless
        ``config.report.reference_pdb`` is specified.

        Returns
        -------
        dict
            ``{cosolvent: List[Hotspot]}`` sorted by composite score.
        """
        if not self._reports:
            raise RuntimeError(
                "run_per_simulation() must be called before run_joint_hotspot_detection()."
            )
        if self._merged_dir is None:
            raise RuntimeError(
                "merge_density_maps() must be called before run_joint_hotspot_detection()."
            )

        hs = self.config.hotspots
        cl = hs.clustering
        all_cosolvents = _collect_all_cosolvents(self.config.simulations)

        effective_cutoff = resolve_agfe_cutoff(hs, self.config.density_maps.temperature)
        self.logger.info(
            f"Hotspot AGFE cutoff: {effective_cutoff:.3f} kcal/mol "
            f"(n_kt={hs.n_kt}, T={self.config.density_maps.temperature} K)."
        )

        # Always disable auto-survival inside detect_all so we can run it
        # per-cosolvent with the correct universe below.
        detector = HotspotDetector(
            out_path=self._merged_dir,
            cosolvent_names=all_cosolvents,
            universe=self._reports[0].universe,
            agfe_cutoff=effective_cutoff,
            top_percentile=hs.top_percentile,
            gridsize=self.config.density_maps.gridsize,
            clustering_strategy=build_clustering_strategy(cl),
            compute_survival_probability=False,
            use_skimage_cleanup=cl.use_skimage_cleanup,
            cleanup_min_size=cl.cleanup_min_size,
            cleanup_hole_size=cl.cleanup_hole_size,
        )

        ck = self.config.checkpoint
        if ck.load_hotspots:
            self.logger.info(
                "Loading hotspot checkpoint (skipping hotspot detection)..."
            )
            results = HotspotDetector.load_checkpoint(
                self._merged_dir,
                all_cosolvents,
            )
        else:
            results = detector.detect_all()
            detector.export_results(results, label_map=True)
            if ck.save_hotspots:
                HotspotDetector.save_checkpoint(results, self._merged_dir)

        # Run survival probability per-cosolvent using the simulation universe
        # that actually contains each probe.  Outputs (CSV + PNG) are written
        # to the probe's own simulation subfolder, not the merged directory.
        survival_kwargs = dict(hs.survival_kwargs or {})
        sp_top_n = int(survival_kwargs.pop("sp_top_n", 5))
        # Supplied zones override the hotspot centroids: explicit argument first, then a
        # zones_csv in the config. See _sp_candidate_zones for why this matters.
        zones_csv = survival_kwargs.pop("zones_csv", None)
        supplied_zones = self.sp_zones
        if supplied_zones is None and zones_csv:
            supplied_zones = _load_zones_csv(zones_csv)
            self.logger.info(
                f"Kinetics zones supplied from {zones_csv}: {len(supplied_zones)} points")
        zone_match_ang = float(survival_kwargs.pop("zone_match_ang", 4.0))
        if supplied_zones is not None or sp_top_n > 0:
            cosolvent_to_universe = _build_cosolvent_universe_map(
                self.config.simulations, self._reports
            )
            # Survival probability averages over replicas rather than using one.
            cosolvent_to_universes = _build_cosolvent_universes_map(
                self.config.simulations, self._reports
            )
            cosolvent_to_out_path = _build_cosolvent_out_path_map(
                self.config.simulations, self._reports
            )
            ran_any = False
            for cosolvent, sites in results.items():
                if not sites:
                    continue
                if cosolvent not in cosolvent_to_universe:
                    self.logger.warning(
                        f"No universe found for cosolvent '{cosolvent}'; "
                        "skipping survival probability."
                    )
                    continue
                candidate_zones = _sp_candidate_zones(sites, sp_top_n,
                                                      zones=supplied_zones)
                sim_out = cosolvent_to_out_path[cosolvent]
                replicas = cosolvent_to_universes.get(
                    cosolvent, [cosolvent_to_universe[cosolvent]])
                self.logger.info(
                    f"Running survival probability for {len(candidate_zones)} "
                    f"zone(s) of '{cosolvent}' over {len(replicas)} replica(s) "
                    f"({'supplied points' if supplied_zones is not None else f'sp_top_n={sp_top_n}'})"
                    f" → {sim_out}"
                )
                detector.universe = cosolvent_to_universe[cosolvent]
                detector.out_path = sim_out
                detector.property_calculator.run_survival_probability(
                    cosolvent_names=[cosolvent],
                    candidate_zones=candidate_zones,
                    universes=replicas,
                    **survival_kwargs,
                )
                ran_any = True

            if ran_any:
                for cosolvent in results:
                    if cosolvent in cosolvent_to_out_path:
                        detector.out_path = cosolvent_to_out_path[cosolvent]
                        z2r = None
                        if supplied_zones is not None:
                            z2r = _zone_to_site_rank(
                                _sp_candidate_zones(results[cosolvent], sp_top_n,
                                                    zones=supplied_zones),
                                results[cosolvent], zone_match_ang)
                            self.logger.info(
                                f"'{cosolvent}': {len(z2r)} of {len(supplied_zones)} supplied "
                                f"zones matched a hotspot within {zone_match_ang} A; "
                                "unmatched zones keep their curves but attach to no site."
                            )
                        detector.property_calculator.fit_survival_probability(
                            {cosolvent: results[cosolvent]}, zone_to_site_rank=z2r
                        )

            # Restore out_path to merged dir for subsequent operations
            detector.out_path = self._merged_dir

        for cosolvent, sites in results.items():
            if sites:
                detector.plot_hotspot_clustering_3d(
                    cosolvent,
                    sites=sites,
                    output_path=os.path.join(
                        self.out_path, f"clustering_3d_{cosolvent}.html"
                    ),
                    top_n=DEFAULT_PLOT_TOP_N,
                )

        bs_cfg = self.config.binding_sites
        if bs_cfg.enabled:
            from cosolvkit.analysis.sites.binding_sites import (
                identify_binding_sites, export_binding_sites,
            )
            binding_sites = identify_binding_sites(
                results, connectivity=bs_cfg.connectivity,
                weights=bs_cfg.weights, merge_tolerance_ang=bs_cfg.merge_tolerance_ang,
                probe_chemotype_overrides=bs_cfg.probe_chemotypes,
            )
            export_binding_sites(binding_sites, self.out_path)
            self._binding_sites = binding_sites
            self.logger.info(f"Identified {len(binding_sites)} binding site(s).")

        return results

    # ------------------------------------------------------------------
    # Step 4 — session generation
    # ------------------------------------------------------------------

    def generate_sessions(self):
        """Generate a binding-site PyMol session from the detected sites."""
        pm = self.config.pymol
        if not pm.enabled:
            self.logger.info("pymol.enabled is false — skipping PyMol session.")
            return
        if not self._binding_sites:
            self.logger.info("No binding sites available — skipping PyMol session.")
            return

        from cosolvkit.analysis.viz.pymol import generate_binding_site_session

        generate_binding_site_session(
            binding_sites=self._binding_sites,
            reference_pdb=self._reference_pdb,
            density_dir=self._merged_dir or self.out_path,
            out_path=self.out_path,
            top_n_sites=pm.top_n_sites,
        )

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline in order:
        per-simulation → merge → hotspot detection → sessions."""
        self.run_per_simulation()
        self.merge_density_maps()
        self.run_joint_hotspot_detection()
        self.generate_sessions()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _collect_all_cosolvents(simulations: List[SimulationEntry]) -> List[str]:
    """Return an ordered deduplicated list of all cosolvent names."""
    seen = []
    for sim in simulations:
        for c in sim.cosolvents:
            if c not in seen:
                seen.append(c)
    return seen


def _find_dx_paths_for_cosolvent(
    cosolvent: str,
    reports: List[Report],
    use_atomtypes: bool,
) -> List[str]:
    """Collect all .dx paths for *cosolvent* across all sim subdirectories."""
    paths = []
    for report in reports:
        if use_atomtypes:
            per_type = sorted(
                f for f in glob(os.path.join(report.out_path, f"map_agfe_*_{cosolvent}.dx"))
                if "raw" not in os.path.basename(f)
            )
            paths.extend(per_type)
        else:
            p = os.path.join(report.out_path, f"map_agfe_{cosolvent}.dx")
            if os.path.exists(p):
                paths.append(p)
    return paths


def _build_cosolvent_universe_map(
    simulations: List[SimulationEntry],
    reports: List[Report],
) -> Dict[str, object]:
    """Return ``{cosolvent_name: universe}`` using the first simulation that has it."""
    mapping: Dict[str, object] = {}
    for sim, report in zip(simulations, reports):
        for cosolvent in sim.cosolvents:
            if cosolvent not in mapping:
                mapping[cosolvent] = report.universe
    return mapping


def _build_cosolvent_universes_map(
    simulations: List[SimulationEntry],
    reports: List[Report],
) -> Dict[str, List[object]]:
    """Return ``{cosolvent_name: [universe, ...]}`` — EVERY replica of each cosolvent.

    Survival probability must be computed per replica and averaged (concatenating
    replicas would invent departure events at each join), so it needs all of them,
    unlike the single-universe map used for residue featurisation.
    """
    mapping: Dict[str, List[object]] = {}
    for sim, report in zip(simulations, reports):
        for cosolvent in sim.cosolvents:
            mapping.setdefault(cosolvent, []).append(report.universe)
    return mapping


def _build_cosolvent_out_path_map(
    simulations: List[SimulationEntry],
    reports: List[Report],
) -> Dict[str, str]:
    """Return ``{cosolvent_name: out_path}`` pointing to the simulation subdir
    that owns each cosolvent.  Survival probability outputs are written there
    so they stay alongside the per-probe density maps and plots."""
    mapping: Dict[str, str] = {}
    for sim, report in zip(simulations, reports):
        for cosolvent in sim.cosolvents:
            if cosolvent not in mapping:
                mapping[cosolvent] = report.out_path
    return mapping


def _group_dx_by_atomtype(cosolvent: str, dx_paths: List[str]) -> Dict[str, List[str]]:
    """Group .dx paths by their atom-type prefix (e.g. 'HBD', 'HBA', 'Car')
    or by 'total' when no atom-type prefix is present."""
    groups: Dict[str, List[str]] = {}
    for p in dx_paths:
        name = os.path.basename(p)
        m = re.match(rf"map_agfe_(.+)_{re.escape(cosolvent)}\.dx$", name)
        key = m.group(1) if m else "total"
        groups.setdefault(key, []).append(p)
    return groups
