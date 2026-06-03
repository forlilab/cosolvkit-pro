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

from .analysis import Report
from .analysis_config import AnalysisConfig, SimulationEntry
from .density_analysis import combine_dx_maps_with_resampling, generate_pymol_session
from .hotspots_detection import HotspotDetector
from .consensus_detection import CrossProbeConsensusDetector
from .hotspot_visualization import (
    generate_consensus_pockets_session,
    generate_pharmacophore_session,
)


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
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.out_path = config.out_path
        os.makedirs(self.out_path, exist_ok=True)

        self._reports: List[Report] = []
        self._merged_dir: Optional[str] = None
        self._reference_pdb: Optional[str] = None

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
                statistics_file=sim.statistics,
                traj_file=sim.trajectory,
                top_file=sim.topology,
                cosolvent_names=sim.cosolvents,
                out_path=sim_out,
            )

            report.generate_report(
                equilibration=cfg.report.equilibration,
                rmsf=cfg.report.rmsf,
                rdf=cfg.report.rdf,
                avg_selection=cfg.report.avg_selection,
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
                cfg.reference_pdb
                or self._reports[0].avg_pdb_path
            )
            if not os.path.exists(self._reference_pdb):
                self.logger.warning(
                    f"Reference PDB not found at '{self._reference_pdb}'. "
                    "Hotspot visualisation may be incomplete. "
                    "Set 'rmsf: true' in report config or provide an explicit 'reference_pdb'."
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
        merge_cfg = cfg.merge

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
                        f"using method='{merge_cfg.method}', "
                        f"resample_to='{merge_cfg.resample_to}' → {out_fname}"
                    )
                    combine_dx_maps_with_resampling(
                        filepaths=paths,
                        method=merge_cfg.method,
                        resample_to=merge_cfg.resample_to,
                        out_fname=out_fname,
                    )

    # ------------------------------------------------------------------
    # Step 3 — joint hotspot detection
    # ------------------------------------------------------------------

    def run_joint_hotspot_detection(self) -> dict:
        """Run :class:`HotspotDetector` on merged maps.

        Uses the protein reference from the first simulation unless
        ``config.reference_pdb`` is specified.

        Returns
        -------
        dict
            ``{cosolvent: List[BindingSite]}`` sorted by composite score.
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

        # Always disable auto-survival inside detect_all so we can run it
        # per-cosolvent with the correct universe below.
        detector = HotspotDetector(
            out_path=self._merged_dir,
            cosolvent_names=all_cosolvents,
            universe=self._reports[0].universe,
            agfe_cutoff=hs.agfe_cutoff,
            min_cluster_voxels=hs.min_cluster_voxels,
            top_percentile=hs.top_percentile,
            score_weights=hs.score_weights,
            gridsize=hs.gridsize,
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
            detector.export_results(results, label_map=hs.export_label_map)
            if ck.save_hotspots:
                HotspotDetector.save_checkpoint(results, self._merged_dir)

        # Run survival probability per-cosolvent using the simulation universe
        # that actually contains each probe.  Outputs (CSV + PNG) are written
        # to the probe's own simulation subfolder, not the merged directory.
        if hs.compute_survival_probability:
            cosolvent_to_universe = _build_cosolvent_universe_map(
                self.config.simulations, self._reports
            )
            cosolvent_to_out_path = _build_cosolvent_out_path_map(
                self.config.simulations, self._reports
            )
            survival_kwargs = hs.survival_kwargs or {}
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
                candidate_zones = [
                    [float(v) for v in site.centroid] for site in sites
                ]
                sim_out = cosolvent_to_out_path[cosolvent]
                self.logger.info(
                    f"Running survival probability for {len(sites)} "
                    f"site(s) of '{cosolvent}' → {sim_out}"
                )
                detector.universe = cosolvent_to_universe[cosolvent]
                detector.out_path = sim_out
                detector.property_calculator.run_survival_probability(
                    cosolvent_names=[cosolvent],
                    candidate_zones=candidate_zones,
                    **survival_kwargs,
                )
                ran_any = True

            if ran_any:
                for cosolvent in results:
                    if cosolvent in cosolvent_to_out_path:
                        detector.out_path = cosolvent_to_out_path[cosolvent]
                        detector.property_calculator.fit_survival_probability(
                            {cosolvent: results[cosolvent]}
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
                    top_n=hs.top_n_plot,
                )

        cs = self.config.consensus
        if cs.enabled:
            self.logger.info("Running cross-probe consensus detection...")
            consensus_detector = CrossProbeConsensusDetector(
                probe_results=results,
                jaccard_threshold=cs.jaccard_threshold,
                community_method=cs.community_method,
                score_weights=cs.score_weights,
            )
            consensus_sites = consensus_detector.detect_communities()
            consensus_detector.export_results(consensus_sites, out_path=self.out_path)

            generate_consensus_pockets_session(
                consensus_sites=consensus_sites,
                out_path=self.out_path,
                reference_pdb=self._reference_pdb,
            )
            generate_pharmacophore_session(
                consensus_sites=consensus_sites,
                out_path=self.out_path,
                reference_pdb=self._reference_pdb,
                top_n=3,
            )

        return results

    # ------------------------------------------------------------------
    # Step 4 — session generation
    # ------------------------------------------------------------------

    def generate_sessions(self):
        """Generate a PyMol session pointing at merged density maps."""
        pm = self.config.pymol
        all_cosolvents = _collect_all_cosolvents(self.config.simulations)
        merged_dir = self._merged_dir or self.out_path

        generate_pymol_session(
            out_path=self.out_path,
            cosolvent_names=all_cosolvents,
            avg_pdb_path=self._reference_pdb,
            density_files=merged_dir,
            selection_string=pm.selection_string,
            reference_pdb=pm.reference_pdb or self.config.reference_pdb,
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
        # self.generate_sessions()


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
