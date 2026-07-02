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

    def test_out_path_resolved(self, tmp_path):
        path = _write_yaml(tmp_path, _minimal_raw())
        cfg = AnalysisConfig.from_yaml(path)
        assert os.path.isabs(cfg.out_path)

    def test_optional_sim_field_label(self, tmp_path):
        raw = _minimal_raw()
        raw["simulations"][0]["label"] = "my_sim"
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.simulations[0].label == "my_sim"

    def test_multiple_simulations(self, tmp_path):
        raw = {
            "out_path": "results",
            "simulations": [
                {"trajectory": "s1/traj.xtc", "topology": "s1/top.prmtop", "cosolvents": ["BEN"]},
                {"trajectory": "s2/traj.xtc", "topology": "s2/top.prmtop", "cosolvents": ["ACE"]},
            ],
        }
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert len(cfg.simulations) == 2
        assert cfg.simulations[1].cosolvents == ["ACE"]

    def test_nested_clustering_config(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"clustering": {"use_skimage_cleanup": True, "cleanup_min_size": 5}}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert isinstance(cfg.hotspots.clustering, ClusteringConfig)
        assert cfg.hotspots.clustering.use_skimage_cleanup is True
        assert cfg.hotspots.clustering.cleanup_min_size == 5

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

    def test_atomtypes_file_path_resolved(self, tmp_path):
        raw = _minimal_raw()
        raw["density_maps"] = {"atomtypes_file": "my_types.json"}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.density_maps.atomtypes_file == str(tmp_path / "my_types.json")

    def test_report_reference_pdb_resolved(self, tmp_path):
        raw = _minimal_raw()
        raw["report"] = {"reference_pdb": "protein.pdb"}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.report.reference_pdb == str(tmp_path / "protein.pdb")

    def test_clustering_strategy_parsed(self, tmp_path):
        raw = _minimal_raw()
        raw["hotspots"] = {"clustering": {"strategy": "dbscan",
                                          "strategy_kwargs": {"eps_angstrom": 2.0},
                                          "min_cluster_voxels": 15}}
        path = _write_yaml(tmp_path, raw)
        cfg = AnalysisConfig.from_yaml(path)
        assert cfg.hotspots.clustering.strategy == "dbscan"
        assert cfg.hotspots.clustering.strategy_kwargs == {"eps_angstrom": 2.0}
        assert cfg.hotspots.clustering.min_cluster_voxels == 15

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

    def test_missing_simulations_raises(self, tmp_path):
        path = _write_yaml(tmp_path, {"out_path": "results"})
        with pytest.raises(ValueError, match="simulations"):
            AnalysisConfig.from_yaml(path)

    def test_empty_simulations_raises(self, tmp_path):
        path = _write_yaml(tmp_path, {"out_path": "results", "simulations": []})
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

    @pytest.mark.parametrize("section,key", [
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
    ])
    def test_removed_section_keys_raise(self, tmp_path, section, key):
        raw = _minimal_raw()
        raw[section] = {key: 1}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match=key):
            AnalysisConfig.from_yaml(path)

    def test_removed_top_level_reference_pdb_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["reference_pdb"] = "protein.pdb"
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="reference_pdb"):
            AnalysisConfig.from_yaml(path)

    def test_removed_top_level_merge_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["merge"] = {"merge_method": "mean"}
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="merge"):
            AnalysisConfig.from_yaml(path)

    def test_removed_simulation_statistics_key_raises(self, tmp_path):
        raw = _minimal_raw()
        raw["simulations"][0]["statistics"] = "s.csv"
        path = _write_yaml(tmp_path, raw)
        with pytest.raises(ValueError, match="statistics"):
            AnalysisConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# generate_template
# ---------------------------------------------------------------------------

class TestGenerateTemplate:

    def test_generate_template_writes_file(self, tmp_path):
        out = str(tmp_path / "template.yaml")
        AnalysisConfig.generate_template(out)
        assert os.path.exists(out)
        # Must be parseable YAML
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data is not None
