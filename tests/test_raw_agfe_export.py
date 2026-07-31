"""The unclamped AGFE map must be written, not just the display-filtered one.

``_smooth_grid_free_energy(energy_cutoff=0)`` sets every voxel with AGFE >= 0 to exactly 0, so
the shipped ``map_agfe_*.dx`` always has MAX = 0.00. That is a display filter, and it is lossy in
two ways that matter:

* **Depletion is erased.** Positive AGFE means the species is excluded from that voxel. For a
  dilute probe those voxels are mostly shot noise, but for water the depleted region is the
  signal — an enclosed region that excludes water is where a hydrophobic ligand atom goes.
* **Two clamped maps cannot be differenced.** A displacement field
  (AGFE_with_probe - AGFE_reference) is meaningless wherever either input was clipped, which is
  most of the volume.

``GridAnalysis`` already computes the unclamped map as ``_agfe_raw`` and can export it, but the
call was commented out in ``report.py``, so no raw map had ever been written.
"""

import os

import numpy as np
import pytest
from gridData import Grid

from tests.test_grid_analysis import HAS_MDA, _make_universe

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

COSOLVENT = "BEN"


def _report(universe, out_path):
    """A Report wired to a synthetic universe, bypassing the file-loading constructor."""
    import logging

    from cosolvkit.analysis.report import Report

    r = Report.__new__(Report)
    r.logger = logging.getLogger(__name__)
    r.cosolvent_names = [COSOLVENT]
    r.universe = universe
    r.out_path = str(out_path)
    r._temperature = None
    os.makedirs(r.out_path, exist_ok=True)
    return r


@pytest.fixture
def universe():
    return _make_universe()


def test_config_exposes_export_raw_defaulting_on():
    from cosolvkit.analysis.config import DensityMapsConfig
    assert DensityMapsConfig().export_raw is True


def test_raw_map_is_written_alongside_the_clamped_one(universe, tmp_cwd, tmp_path):
    out = tmp_path / "out"
    r = _report(universe, out)
    r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                            gridsize=1.0, temperature=300.0, export_raw=True)
    assert os.path.isfile(out / f"map_agfe_{COSOLVENT}.dx")
    assert os.path.isfile(out / f"map_agfe_raw_{COSOLVENT}.dx")


def test_raw_export_can_be_switched_off(universe, tmp_cwd, tmp_path):
    out = tmp_path / "out"
    r = _report(universe, out)
    r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                            gridsize=1.0, temperature=300.0, export_raw=False)
    assert os.path.isfile(out / f"map_agfe_{COSOLVENT}.dx")
    assert not os.path.exists(out / f"map_agfe_raw_{COSOLVENT}.dx")


def test_raw_map_keeps_the_positive_values_the_clamp_destroys(universe, tmp_cwd, tmp_path):
    """The substance of the fix: raw retains depletion, the shipped map is capped at 0."""
    out = tmp_path / "out"
    r = _report(universe, out)
    r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                            gridsize=1.0, temperature=300.0, export_raw=True)
    clamped = Grid(str(out / f"map_agfe_{COSOLVENT}.dx")).grid
    raw = Grid(str(out / f"map_agfe_raw_{COSOLVENT}.dx")).grid

    assert clamped.max() == pytest.approx(0.0, abs=1e-9), "shipped map should be capped at 0"
    assert raw.max() > 0.0, "raw map must retain depleted (positive) voxels"
    assert np.any(raw > 0) and not np.any(clamped > 0)


def test_raw_and_clamped_share_one_grid_so_they_can_be_compared(universe, tmp_cwd, tmp_path):
    out = tmp_path / "out"
    r = _report(universe, out)
    r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                            gridsize=1.0, temperature=300.0, export_raw=True)
    a = Grid(str(out / f"map_agfe_{COSOLVENT}.dx"))
    b = Grid(str(out / f"map_agfe_raw_{COSOLVENT}.dx"))
    assert a.grid.shape == b.grid.shape
    np.testing.assert_allclose(a.origin, b.origin)
    np.testing.assert_allclose(a.delta, b.delta)


def test_existing_raw_map_does_not_block_regeneration_check(universe, tmp_cwd, tmp_path):
    """The skip-if-exists guard keys on the clamped map; a stray raw file must not trigger it."""
    out = tmp_path / "out"
    os.makedirs(out, exist_ok=True)
    Grid(np.zeros((3, 3, 3)), edges=[np.arange(4.0)] * 3).export(
        str(out / f"map_agfe_raw_{COSOLVENT}.dx"))
    r = _report(universe, out)
    r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                            gridsize=1.0, temperature=300.0, export_raw=True)
    assert Grid(str(out / f"map_agfe_raw_{COSOLVENT}.dx")).grid.shape != (3, 3, 3)
