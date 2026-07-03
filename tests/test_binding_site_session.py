import os
import numpy as np
import pytest
from gridData import Grid

from cosolvkit.analysis.viz import pymol as pmod


def test_write_mask_dx_roundtrips(tmp_path):
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2:4, 2:4, 2:4] = True
    origin = np.array([0.0, 0.0, 0.0])
    delta = np.array([0.5, 0.5, 0.5])
    out = str(tmp_path / "pocket.dx")
    ret = pmod._write_mask_dx(mask, origin, delta, out)
    assert ret == out
    assert os.path.isfile(out)
    g = Grid(out)
    # 8 voxels set to 1.0, everything else 0.0
    assert int(np.count_nonzero(g.grid > 0.5)) == 8
    assert g.grid.max() == pytest.approx(1.0)


def test_site_carve_radius_positive_for_nonempty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:4, 2:4] = True
    delta = np.array([0.5, 0.5, 0.5])
    r = pmod._site_carve_radius(mask, delta)
    # extent 6x2x2 voxels -> 3.0 x 1.0 x 1.0 Å; half-diagonal ~1.66 + 2 pad
    assert r == pytest.approx(0.5 * np.linalg.norm([3.0, 1.0, 1.0]) + 2.0, abs=1e-6)


def test_site_carve_radius_empty_mask_is_zero():
    mask = np.zeros((4, 4, 4), dtype=bool)
    assert pmod._site_carve_radius(mask, np.array([0.5, 0.5, 0.5])) == 0.0


def test_session_noops_without_pymol(tmp_path, monkeypatch):
    monkeypatch.setattr(pmod, "_PYMOL_AVAILABLE", False)
    ret = pmod.generate_binding_site_session(
        binding_sites=[], reference_pdb=None,
        density_dir=str(tmp_path), out_path=str(tmp_path),
    )
    assert ret is None
