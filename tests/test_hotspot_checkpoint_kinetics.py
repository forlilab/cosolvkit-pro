"""The hotspot checkpoint must carry the kinetics, not just the geometry.

``save_checkpoint`` ran immediately after detection, while the nine ``sp_*`` metrics are attached
later by ``fit_survival_probability``. So a checkpoint written by the pipeline held only the
``geom_*`` properties, and anything reloading it — ``combine_binding_sites.py``, the weight
sweeps — saw kinetics of None and silently scored a zero kinetics term.

The early save is still worth keeping: survival probability is the long step, and if it dies the
geometry checkpoint should survive. So the checkpoint is written twice and the post-kinetics
write overwrites the first.
"""

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

from cosolvkit.analysis.hotspots_detection import HotspotDetector
from cosolvkit.analysis.multi_report import MultiReport

SP_KEYS = [
    "sp_mrt", "sp_half_life", "sp_plateau", "sp_tau_single", "sp_r2_single",
    "sp_amplitude_fast", "sp_tau_fast", "sp_tau_slow", "sp_r2_biexp",
]


def _report(merged_dir, save_hotspots=True):
    r = MultiReport.__new__(MultiReport)
    r.logger = logging.getLogger(__name__)
    r._merged_dir = str(merged_dir)
    r.config = SimpleNamespace(checkpoint=SimpleNamespace(save_hotspots=save_hotspots))
    os.makedirs(r._merged_dir, exist_ok=True)
    return r


def _with_kinetics(site):
    for i, k in enumerate(SP_KEYS):
        site.add_property(k, float(i) + 0.5)
    return site


def test_checkpoint_is_written_when_enabled(make_hotspot, tmp_path):
    r = _report(tmp_path)
    r._save_hotspot_checkpoint({"BEN": [make_hotspot()]})
    assert os.path.isfile(
        tmp_path / "hotspot_checkpoints" / "hotspot_checkpoint_BEN.npz")


def test_checkpoint_is_skipped_when_disabled(make_hotspot, tmp_path):
    r = _report(tmp_path, save_hotspots=False)
    r._save_hotspot_checkpoint({"BEN": [make_hotspot()]})
    assert not os.path.exists(tmp_path / "hotspot_checkpoints")


def test_all_nine_sp_metrics_survive_the_round_trip(make_hotspot, tmp_path):
    """The substance: a checkpoint written after kinetics carries every sp_* metric."""
    r = _report(tmp_path)
    r._save_hotspot_checkpoint({"BEN": [_with_kinetics(make_hotspot())]})

    loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])
    props = loaded["BEN"][0].properties
    missing = [k for k in SP_KEYS if k not in props]
    assert not missing, f"kinetics lost in checkpoint: {missing}"
    for i, k in enumerate(SP_KEYS):
        assert props[k] == pytest.approx(float(i) + 0.5)


def test_geometry_is_still_preserved_alongside_kinetics(make_hotspot, tmp_path):
    r = _report(tmp_path)
    site = _with_kinetics(make_hotspot(agfe_min=-2.5))
    site.add_property("geom_volume", 123.0)
    r._save_hotspot_checkpoint({"BEN": [site]})

    loaded = HotspotDetector.load_checkpoint(str(tmp_path), ["BEN"])[  "BEN"][0]
    assert loaded.properties["geom_volume"] == pytest.approx(123.0)
    assert loaded.properties["sp_mrt"] == pytest.approx(0.5)
    assert loaded.agfe_min == pytest.approx(-2.5)
    assert loaded.voxel_mask.sum() == site.voxel_mask.sum()


def test_second_write_overwrites_the_pre_kinetics_one(make_hotspot, tmp_path):
    """Mirrors the pipeline: save after detection, then again after kinetics."""
    r = _report(tmp_path)
    bare = make_hotspot()
    r._save_hotspot_checkpoint({"BEN": [bare]})
    assert "sp_mrt" not in HotspotDetector.load_checkpoint(
        str(tmp_path), ["BEN"])["BEN"][0].properties

    r._save_hotspot_checkpoint({"BEN": [_with_kinetics(make_hotspot())]})
    assert "sp_mrt" in HotspotDetector.load_checkpoint(
        str(tmp_path), ["BEN"])["BEN"][0].properties


def test_empty_results_is_not_an_error(tmp_path):
    r = _report(tmp_path)
    r._save_hotspot_checkpoint({})
    r._save_hotspot_checkpoint({"BEN": []})
