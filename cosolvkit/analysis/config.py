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


def resolve_agfe_cutoff(hotspots_cfg, temperature):
    """Effective AGFE cutoff in kcal/mol: ``-n_kt * kB * T`` (an e^n_kt enrichment over bulk)."""
    from cosolvkit.analysis.core.grid import BOLTZMANN_CONSTANT_KB
    return -float(hotspots_cfg.n_kt) * BOLTZMANN_CONSTANT_KB * float(temperature)


# ---------------------------------------------------------------------------
# Leaf dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class SimulationEntry:
    """A single MD simulation to include in the analysis."""
    trajectory:  str
    topology:    str
    cosolvents:  List[str]
    label:       Optional[str] = None


@dataclass
class ReportConfig:
    rmsf:            bool          = True
    rdf:             bool          = False
    rmsf_avg_selec:  str           = "protein"
    align_selection: str           = "protein and name CA"
    reference_pdb:   Optional[str] = None


@dataclass
class DensityMapsConfig:
    use_atomtypes:  bool          = True
    atomtypes_file: Optional[str] = None
    gridsize:       float         = 0.5    # voxel edge, Angstrom
    temperature:    float         = 300.0  # K
    # Also write unclamped map_agfe_raw_*.dx; map_agfe_*.dx zeroes voxels >= 0, losing depletion.
    export_raw:     bool          = True


@dataclass
class MiscConfig:
    merge_method:        str = "mean"   # mean | min | max | sum | median
    merge_resampling_to: str = "first"  # first | largest | smallest


@dataclass
class ClusteringConfig:
    strategy:            str  = "skimage_watershed"  # skimage_watershed | connected_components
    strategy_kwargs:     Dict = field(default_factory=dict)
    # THE size threshold, and the only default for it. Expressed as a volume because a voxel count
    # means different physical things at different gridsizes: 10 voxels is 1.25 A^3 at 0.5 A but
    # 5.12 A^3 at 0.8 A, and that silent 4x change once dropped a real hotspot (see
    # tests/test_min_cluster_volume.py).
    #
    # 20 A^3 is the van der Waals volume of one heavy atom (C ~ 20.4 A^3) = ~39 voxels at 0.8 A:
    # the smallest blob that can plausibly hold an atom rather than being shot noise.
    #
    # Measured on FosAKP (18 probes, scripts/sweep_min_cluster_volume.py). Ground-truth site
    # coverage and spurious-hotspot count vs this threshold:
    #     1.25 A^3 (deployed)  7/7 sites   325 novel hotspots  (100%)
    #     5.0                  7/7         247                 ( 76%)
    #    10.0                  7/7         200                 ( 62%)
    #    20.0  <- default      7/7         148                 ( 46%)
    #    40.0                  6/7 (!)      96                 ( 30%)
    # 20 A^3 is the largest threshold that still recovers every ground-truth site, and it removes
    # more than half the spurious hotspots. 40 A^3 looks defensible on atom volume alone but loses
    # a real site, so it is not.
    min_cluster_volume_ang3: float = 20.0
    # Escape hatch: a literal voxel count that WINS over the volume when set. Leave None unless you
    # need grid-dependent behaviour, e.g. reproducing an older run.
    min_cluster_voxels:  Optional[int] = None
    use_skimage_cleanup: bool = False
    cleanup_min_size:    int  = 1
    cleanup_hole_size:   int  = 2

    def resolve_min_cluster_voxels(self, gridsize):
        """Voxel threshold: the explicit count if one was given, else derived from the volume."""
        if self.min_cluster_voxels is not None:
            return int(self.min_cluster_voxels)
        from cosolvkit.analysis.sites.clustering import min_cluster_voxels_for_volume
        return min_cluster_voxels_for_volume(self.min_cluster_volume_ang3, gridsize)


@dataclass
class HotspotsConfig:
    n_kt:            float            = 1.0   # cutoff = -n_kt * kB * T
    top_percentile:  float            = 10.0
    survival_kwargs: Dict             = field(default_factory=dict)
    clustering:      ClusteringConfig = field(default_factory=ClusteringConfig)


@dataclass
class BindingSitesConfig:
    enabled:             bool          = True
    connectivity:        int           = 26
    weights:             Optional[Dict] = None
    merge_tolerance_ang: float         = 2.0  # Angstrom
    # ``{resname: [chemotype_class, ...]}`` overriding
    # cosolvkit.analysis.core.chemotypes.DEFAULT_PROBE_CHEMOTYPES; null = built-in table only.
    probe_chemotypes:    Optional[Dict] = None


@dataclass
class PyMolConfig:
    enabled:     bool = True
    top_n_sites: int  = 0   # 0 = all sites


@dataclass
class CheckpointConfig:
    """Checkpoint save/load for the hotspot detection step."""
    # Write compressed NPZ files under out_path/hotspot_checkpoints/ after detection.
    save_hotspots: bool = True
    # Skip detection and reload the saved checkpoint instead.
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
    report:        ReportConfig     = field(default_factory=ReportConfig)
    density_maps:  DensityMapsConfig = field(default_factory=DensityMapsConfig)
    misc:          MiscConfig       = field(default_factory=MiscConfig)
    hotspots:      HotspotsConfig   = field(default_factory=HotspotsConfig)
    binding_sites: BindingSitesConfig = field(default_factory=BindingSitesConfig)
    pymol:         PyMolConfig      = field(default_factory=PyMolConfig)
    checkpoint:    CheckpointConfig = field(default_factory=CheckpointConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        """Load and validate an analysis config YAML file.

        Relative paths inside the YAML are resolved against the YAML's own directory.

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

        known_top = {"out_path", "simulations",
                     "report", "density_maps", "misc", "hotspots",
                     "binding_sites", "pymol", "checkpoint"}
        bad = set(raw) - known_top
        if bad:
            raise ValueError(
                f"Unknown keys in analysis config: {sorted(bad)}. "
                f"Valid keys are: {sorted(known_top)}"
            )

        if "out_path" not in raw:
            raise ValueError("'out_path' is required in the analysis config.")
        if "simulations" not in raw or not raw["simulations"]:
            raise ValueError("'simulations' must be a non-empty list.")

        def resolve(p):
            """Resolve a path relative to the config file's directory."""
            if p is None:
                return None
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        sims = []
        for i, s in enumerate(raw["simulations"]):
            missing = [k for k in ("trajectory", "topology", "cosolvents") if k not in s]
            if missing:
                raise ValueError(
                    f"simulations[{i}] is missing required keys: {missing}"
                )
            unknown_sim = set(s) - {"trajectory", "topology", "cosolvents", "label"}
            if unknown_sim:
                raise ValueError(
                    f"simulations[{i}] has unknown keys: {sorted(unknown_sim)}"
                )
            sims.append(SimulationEntry(
                trajectory=resolve(s["trajectory"]),
                topology=resolve(s["topology"]),
                cosolvents=list(s["cosolvents"]),
                label=s.get("label"),
            ))

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
        mg_raw = dict(raw.get("misc",         {}))
        hs_raw = dict(raw.get("hotspots",     {}))
        bs_raw = dict(raw.get("binding_sites", {}))
        pm_raw = dict(raw.get("pymol",        {}))
        ck_raw = dict(raw.get("checkpoint",   {}))

        # clustering is nested inside hotspots and must be injected as a dataclass instance
        cl_raw = dict(hs_raw.pop("clustering", {}))
        clustering = _parse(ClusteringConfig, cl_raw)
        hotspots = _parse(HotspotsConfig, {**hs_raw, "clustering": clustering})

        if dm_raw.get("atomtypes_file"):
            dm_raw["atomtypes_file"] = resolve(dm_raw["atomtypes_file"])

        if r_raw.get("reference_pdb"):
            r_raw["reference_pdb"] = resolve(r_raw["reference_pdb"])

        return cls(
            out_path=resolve(raw["out_path"]),
            simulations=sims,
            report=_parse(ReportConfig, r_raw),
            density_maps=_parse(DensityMapsConfig, dm_raw),
            misc=_parse(MiscConfig, mg_raw),
            hotspots=hotspots,
            binding_sites=_parse(BindingSitesConfig, bs_raw),
            pymol=_parse(PyMolConfig, pm_raw),
            checkpoint=_parse(CheckpointConfig, ck_raw),
        )

    @classmethod
    def generate_template(cls, path: str) -> None:
        """Write a fully-commented YAML template to *path*."""
        template_src = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "analysis_config.yaml"
        )
        shutil.copy(template_src, path)
