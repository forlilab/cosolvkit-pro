#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# YAML-based configuration for the analysis pipeline
#

import os
import shutil
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import yaml


# ---------------------------------------------------------------------------
# Leaf dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class SimulationEntry:
    """A single MD simulation to include in the analysis."""
    trajectory:  str
    topology:    str
    cosolvents:  List[str]
    statistics:  Optional[str] = None
    label:       Optional[str] = None


@dataclass
class ReportConfig:
    equilibration:   bool = False
    rmsf:            bool = True
    rdf:             bool = False
    avg_selection:   str  = "protein"
    align_selection: str  = "protein and name CA"


@dataclass
class DensityMapsConfig:
    use_atomtypes:  bool          = True
    atomtypes_file: Optional[str] = None
    gridsize:       float         = 0.5
    temperature:    float         = 300.0


@dataclass
class MergeConfig:
    method:      str = "mean"   # mean | min | max | sum | median
    resample_to: str = "first"  # first | largest | smallest


@dataclass
class ClusteringConfig:
    use_skimage_cleanup: bool = False
    cleanup_min_size:    int  = 1
    cleanup_hole_size:   int  = 2


@dataclass
class HotspotsConfig:
    agfe_cutoff:        float                = -1.0
    min_cluster_voxels: int                  = 20
    top_percentile:     float                = 10.0
    score_weights:      Optional[Dict]       = None
    export_label_map:   bool                 = True
    add_to_pymol:       bool                 = True
    gridsize:           float                = 0.5
    top_n_plot:         int                  = 10
    top_n_survival:     int                  = 0
    survival_kwargs:    Optional[Dict]       = field(default_factory=dict)
    clustering:         ClusteringConfig     = field(default_factory=ClusteringConfig)


@dataclass
class ConsensusConfig:
    enabled:          bool          = False
    jaccard_threshold: float        = 0.05
    community_method: str           = "connected_components"
    score_weights:    Optional[Dict] = None


@dataclass
class PyMolConfig:
    selection_string: Optional[str] = None
    reference_pdb:    Optional[str] = None


@dataclass
class CheckpointConfig:
    """Controls checkpoint save/load for the hotspot detection step.

    Set ``save_hotspots: true`` (default) to write compressed NPZ files under
    ``out_path/hotspot_checkpoints/`` after each hotspot detection run.

    Set ``load_hotspots: true`` to skip hotspot detection entirely and reload
    the previously saved checkpoint instead — useful when you only want to
    re-run consensus with different parameters.
    """
    save_hotspots: bool = True
    load_hotspots: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Full analysis configuration loaded from a YAML file.

    Instantiate via :meth:`from_yaml` rather than directly.
    """
    out_path:      str
    simulations:   List[SimulationEntry]
    reference_pdb: Optional[str]    = None
    report:        ReportConfig     = field(default_factory=ReportConfig)
    density_maps:  DensityMapsConfig = field(default_factory=DensityMapsConfig)
    merge:         MergeConfig      = field(default_factory=MergeConfig)
    hotspots:      HotspotsConfig   = field(default_factory=HotspotsConfig)
    consensus:     ConsensusConfig  = field(default_factory=ConsensusConfig)
    pymol:         PyMolConfig      = field(default_factory=PyMolConfig)
    checkpoint:    CheckpointConfig = field(default_factory=CheckpointConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        """Load and validate an analysis config YAML file.

        All relative paths inside the YAML are resolved against the directory
        that contains the YAML file, so configs are portable.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If required keys are missing or unknown keys are present.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Analysis config file not found: {path}")

        base_dir = os.path.dirname(path)

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raw = {}

        # --- validate top-level keys ---
        known_top = {"out_path", "simulations", "reference_pdb",
                     "report", "density_maps", "merge", "hotspots", "consensus", "pymol",
                     "checkpoint"}
        bad = set(raw) - known_top
        if bad:
            raise ValueError(
                f"Unknown keys in analysis config: {sorted(bad)}. "
                f"Valid keys are: {sorted(known_top)}"
            )

        # --- required fields ---
        if "out_path" not in raw:
            raise ValueError("'out_path' is required in the analysis config.")
        if "simulations" not in raw or not raw["simulations"]:
            raise ValueError("'simulations' must be a non-empty list.")

        def resolve(p):
            """Resolve a path relative to the config file's directory."""
            if p is None:
                return None
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        # --- parse simulations ---
        sims = []
        for i, s in enumerate(raw["simulations"]):
            missing = [k for k in ("trajectory", "topology", "cosolvents") if k not in s]
            if missing:
                raise ValueError(
                    f"simulations[{i}] is missing required keys: {missing}"
                )
            unknown_sim = set(s) - {"trajectory", "topology", "cosolvents", "statistics", "label"}
            if unknown_sim:
                raise ValueError(
                    f"simulations[{i}] has unknown keys: {sorted(unknown_sim)}"
                )
            sims.append(SimulationEntry(
                trajectory=resolve(s["trajectory"]),
                topology=resolve(s["topology"]),
                cosolvents=list(s["cosolvents"]),
                statistics=resolve(s.get("statistics")),
                label=s.get("label"),
            ))

        # --- generic section parser ---
        def _parse(section_cls, raw_dict):
            valid = {f.name for f in dataclasses.fields(section_cls)}
            bad_keys = set(raw_dict) - valid
            if bad_keys:
                raise ValueError(
                    f"Unknown keys in {section_cls.__name__}: {sorted(bad_keys)}. "
                    f"Valid keys: {sorted(valid)}"
                )
            return section_cls(**raw_dict)

        r_raw  = dict(raw.get("report",       {}))
        dm_raw = dict(raw.get("density_maps", {}))
        mg_raw = dict(raw.get("merge",        {}))
        hs_raw = dict(raw.get("hotspots",     {}))
        cs_raw = dict(raw.get("consensus",    {}))
        pm_raw = dict(raw.get("pymol",        {}))
        ck_raw = dict(raw.get("checkpoint",   {}))

        # clustering is nested inside hotspots
        cl_raw = dict(hs_raw.pop("clustering", {}))
        clustering = _parse(ClusteringConfig, cl_raw)

        # hotspots needs clustering injected as a dataclass instance
        hotspots = _parse(HotspotsConfig, {**hs_raw, "clustering": clustering})

        # resolve paths inside density_maps
        if dm_raw.get("atomtypes_file"):
            dm_raw["atomtypes_file"] = resolve(dm_raw["atomtypes_file"])

        # resolve paths inside pymol
        if pm_raw.get("reference_pdb"):
            pm_raw["reference_pdb"] = resolve(pm_raw["reference_pdb"])

        return cls(
            out_path=resolve(raw["out_path"]),
            simulations=sims,
            reference_pdb=resolve(raw.get("reference_pdb")),
            report=_parse(ReportConfig, r_raw),
            density_maps=_parse(DensityMapsConfig, dm_raw),
            merge=_parse(MergeConfig, mg_raw),
            hotspots=hotspots,
            consensus=_parse(ConsensusConfig, cs_raw),
            pymol=_parse(PyMolConfig, pm_raw),
            checkpoint=_parse(CheckpointConfig, ck_raw),
        )

    @classmethod
    def generate_template(cls, path: str) -> None:
        """Write a fully-commented YAML template to *path*.

        Parameters
        ----------
        path : str
            Destination path for the template YAML file.
        """
        template_src = os.path.join(
            os.path.dirname(__file__), "data", "analysis_config_template.yaml"
        )
        shutil.copy(template_src, path)
