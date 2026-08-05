"""Tests for analysis_config.py — YAML loading, validation, path resolution."""

import os
import pytest
import yaml

from cosolvkit.analysis.analysis_config import (
    AnalysisConfig,
    ClusteringConfig,
    DensityMapsConfig,
    HotspotsConfig,
    MiscConfig,
    SimulationEntry,
)


def _write_yaml(tmp_path, content: dict, name="config.yaml") -> str:
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.dump(content, f)
    return str(path)


def _minimal_raw(tmp_path=None, traj="sim/traj.xtc", top="sim/top.prmtop"):
    return {
        "out_path": "results",
        "simulations": [
            {"trajectory": traj, "topology": top, "cosolvents": ["BEN"]}
        ],
    }


# ---------------------------------------------------------------------------
# Happy-path loading
# ---------------------------------------------------------------------------

class TestFromYamlValid:

    def test_minimal_config_loads(self, tmp_path):
        path = _write_yaml(tmp_path, _minimal_raw())
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.out_path.endswith("results")
        assert len(cfg.simulations) == 1
        assert cfg.simulations[0].cosolvents == ["BEN"]

    def test_defaults_applied(self, tmp_path):
        path = _write_yaml(tmp_path, _minimal_raw())
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.report.rmsf is True
        assert cfg.report.rmsf_avg_selec == "protein"
        assert cfg.density_maps.gridsize == 0.5
        assert cfg.density_maps.temperature == 300.0
        assert cfg.misc.merge_method == "mean"
        assert cfg.misc.merge_resampling_to == "first"
        assert cfg.hotspots.n_kt == 1.0

    def test_relative_paths_resolved(self, tmp_path):
        raw = _minimal_raw(traj="sim/traj.xtc", top="sim/top.prmtop")
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.simulations[0].trajectory == str(tmp_path / "sim/traj.xtc")
        assert cfg.simulations[0].topology == str(tmp_path / "sim/top.prmtop")
        assert os.path.isabs(cfg.out_path)

    def test_multiple_simulations_and_optional_label(self, tmp_path):
        raw = {
            "out_path": "results",
            "simulations": [
                {"trajectory": "s1/traj.xtc", "topology": "s1/top.prmtop",
                 "cosolvents": ["BEN"], "label": "my_sim"},
                {"trajectory": "s2/traj.xtc", "topology": "s2/top.prmtop", "cosolvents": ["ACE"]},
            ],
        }
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert len(cfg.simulations) == 2
        assert cfg.simulations[0].label == "my_sim"
        assert cfg.simulations[1].cosolvents == ["ACE"]

    def test_binding_sites_section_parsed(self, tmp_path):
        raw = _minimal_raw()
        raw["binding_sites"] = {"enabled": False, "connectivity": 6}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.binding_sites.enabled is False
        assert cfg.binding_sites.connectivity == 6

    def test_checkpoint_section_parsed(self, tmp_path):
        raw = _minimal_raw()
        raw["checkpoint"] = {"save_hotspots": False, "load_hotspots": True}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.checkpoint.save_hotspots is False
        assert cfg.checkpoint.load_hotspots is True

    def test_optional_file_fields_resolved_relative_to_the_yaml(self, tmp_path):
        raw = _minimal_raw()
        raw["density_maps"] = {"atomtypes_file": "my_types.json"}
        raw["report"] = {"reference_pdb": "protein.pdb"}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.density_maps.atomtypes_file == str(tmp_path / "my_types.json")
        assert cfg.report.reference_pdb == str(tmp_path / "protein.pdb")

    def test_nested_clustering_config(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"clustering": {"use_skimage_cleanup": True,
                                          "cleanup_min_size": 5,
                                          "strategy": "connected_components",
                                          "strategy_kwargs": {"connectivity": 6},
                                          "min_cluster_volume_ang3": 15.0}}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert isinstance(cfg.hotspots.clustering, ClusteringConfig)
        assert cfg.hotspots.clustering.use_skimage_cleanup is True
        assert cfg.hotspots.clustering.cleanup_min_size == 5
        assert cfg.hotspots.clustering.strategy == "connected_components"
        assert cfg.hotspots.clustering.strategy_kwargs == {"connectivity": 6}
        assert cfg.hotspots.clustering.min_cluster_volume_ang3 == 15.0

    def test_survival_kwargs_sp_top_n_parsed(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"survival_kwargs": {"sp_top_n": 3, "radius": 6.0}}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.hotspots.survival_kwargs["sp_top_n"] == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestFromYamlErrors:

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnalysisConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_missing_out_path_raises(self, tmp_path):
        raw = {"simulations": [{"trajectory": "t.xtc", "topology": "t.prmtop", "cosolvents": ["BEN"]}]}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="out_path"):
            AnalysisConfig.from_yaml(path)

    def test_missing_or_empty_simulations_raises(self, tmp_path):
        for i, raw in enumerate([{"out_path": "results"},
                                 {"out_path": "results", "simulations": []}]):
            path = _write_yaml(tmp_path, raw, name=f"sims_{i}.yaml")
            with pytest.raises(ValueError, match="simulations"):
                AnalysisConfig.from_yaml(path)

    def test_unknown_top_level_key_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["totally_unknown"] = "value"
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="totally_unknown"):
            AnalysisConfig.from_yaml(path)

    def test_unknown_simulation_key_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["simulations"][0]["bad_field"] = "oops"
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="bad_field"):
            AnalysisConfig.from_yaml(path)

    def test_missing_trajectory_in_simulation_raises(self, tmp_path):
        raw = {"out_path": "results", "simulations": [{"topology": "t.prmtop", "cosolvents": ["BEN"]}]}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="trajectory"):
            AnalysisConfig.from_yaml(path)

    def test_unknown_hotspots_key_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"nonexistent_param": 999}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="nonexistent_param"):
            AnalysisConfig.from_yaml(path)

    def test_unknown_clustering_key_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"clustering": {"unknown_cleanup": True}}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="unknown_cleanup"):
            AnalysisConfig.from_yaml(path)

    def test_removed_keys_raise(self, tmp_path):
        """Every key below was deliberately removed from the schema.

        Kept as one loop rather than one test per key: they all exercise the same
        unknown-key rejection, and the list is the point (it must not silently
        creep back into the config).  ``None`` = top level, ``"simulations"`` =
        inside a simulation entry, anything else = that section.
        """
        removed = [
            ("hotspots", "agfe_cutoff"),
            ("hotspots", "cutoff_mode"),
            ("hotspots", "score_weights"),
            ("hotspots", "export_label_map"),
            ("hotspots", "add_to_pymol"),
            ("hotspots", "gridsize"),
            ("hotspots", "top_n_plot"),
            ("hotspots", "compute_survival_probability"),
            ("hotspots", "min_cluster_voxels"),
            ("report", "equilibration"),
            ("report", "avg_selection"),
            ("pymol", "selection_string"),
            ("pymol", "reference_pdb"),
            (None, "reference_pdb"),
            (None, "merge"),
            ("simulations", "statistics"),
        ]
        for i, (section, key) in enumerate(removed):
            raw = _minimal_raw()
            if section is None:
                raw[key] = {"merge_method": "mean"} if key == "merge" else "value"
            elif section == "simulations":
                raw["simulations"][0][key] = "s.csv"
            else:
                raw[section] = {key: 1}
            path = _write_yaml(tmp_path, raw, name=f"removed_{i}.yaml")
            with pytest.raises(ValueError, match=key):
                AnalysisConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# generate_template
# ---------------------------------------------------------------------------

class TestGenerateTemplate:

    def test_generated_template_parses_under_new_schema(self, tmp_path):
        out = str(tmp_path / "template.yaml")
        AnalysisConfig.generate_template(out)
        assert os.path.exists(out)
        # add the required non-default fields the template leaves as placeholders
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data is not None
        data["out_path"] = "results"
        data["simulations"] = [
            {"trajectory": "t.xtc", "topology": "t.prmtop", "cosolvents": ["BEN"]}
        ]
        path2 = str(tmp_path / "filled.yaml")
        with open(path2, "w") as f:
            yaml.dump(data, f)
        cfg = AnalysisConfig.from_yaml(path2)   # must not raise
        assert cfg.hotspots.n_kt == 1.0
        assert cfg.misc.merge_method == "mean"
