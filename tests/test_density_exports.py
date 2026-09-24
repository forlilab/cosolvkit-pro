"""Density exports: raw counts and a fixed, mergeable export region.

* Counts are the only export a user can re-smooth correctly: smoothing an AGFE map is not the
  same as smoothing the density and inverting it. The sidecar must carry every constant needed.
* In atom-type mode the per-type counts used to be overwritten by AGFE values, so any export of
  them wrote free energies under a density name.
* A fixed export region must give identical grids across replicas WITHOUT changing any value:
  the bulk density and the smoothing are computed on the full grid, and only the output is cut.
"""

import json
import os

import numpy as np
import pytest
from gridData import Grid

from tests.test_grid_analysis import HAS_MDA, _make_universe

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

COS = "BEN"
T = 300.0


def _report(universe, out_path):
    import logging
    from cosolvkit.analysis.report import Report

    r = Report.__new__(Report)
    r.logger = logging.getLogger(__name__)
    r.cosolvent_names = [COS]
    r.universe = universe
    r.out_path = str(out_path)
    r._temperature = None
    os.makedirs(r.out_path, exist_ok=True)
    return r


def _maps(universe, out, **kw):
    _report(universe, out).generate_density_maps(
        cosolvent_names=[COS], use_atomtypes=False, gridsize=1.0, temperature=T, **kw)
    return out


def _read(out, name):
    return Grid(str(out / name))


# --- counts -----------------------------------------------------------------

def test_counts_are_always_exported_and_hold_every_position(tmp_cwd, tmp_path):
    out = _maps(_make_universe(), tmp_path / "out")
    counts = _read(out, f"map_counts_{COS}.dx").grid
    assert np.allclose(counts, np.round(counts)), "counts must be integers"
    assert counts.sum() == pytest.approx(6 * 5), "6 probe atoms x 5 frames"


def test_sidecar_carries_the_normalisation(tmp_cwd, tmp_path):
    out = _maps(_make_universe(), tmp_path / "out")
    meta = json.load(open(out / f"map_counts_{COS}.json"))
    assert meta["n_frames"] == 5 and meta["n_atoms"] == 6
    assert meta["n_accessible_voxels"] > 0
    assert meta["temperature_K"] == T and meta["gridsize"] == 1.0
    counts = _read(out, f"map_counts_{COS}.dx")
    np.testing.assert_allclose(meta["origin"], counts.origin)
    assert tuple(meta["shape"]) == counts.grid.shape


def test_smoothing_the_counts_reproduces_the_shipped_map(tmp_cwd, tmp_path):
    """The point of exporting counts: the user's own smoothing path is exact, not approximate."""
    from scipy.ndimage import gaussian_filter
    from cosolvkit.analysis.core.grid import _detection_floor_counts

    out = _maps(_make_universe(), tmp_path / "out")
    meta = json.load(open(out / f"map_counts_{COS}.json"))
    counts = _read(out, f"map_counts_{COS}.dx").grid
    sigma = (1.4 / 3.0) / meta["gridsize"]
    smooth = np.maximum(gaussian_filter(counts, sigma=sigma, mode="constant", cval=0.0),
                        _detection_floor_counts(sigma))
    n_o = meta["n_atoms"] / meta["n_accessible_voxels"]
    agfe = -meta["kB_kcal_per_mol_K"] * meta["temperature_K"] * np.log(
        smooth / meta["n_frames"] / n_o)
    np.testing.assert_allclose(agfe, _read(out, f"map_agfe_raw_{COS}.dx").grid, atol=1e-4)


def _analysis(tmp_path, **kw):
    from cosolvkit.analysis.core.grid import GridAnalysis
    u = _make_universe()
    a = GridAnalysis(u.select_atoms(f"resname {COS}"), gridsize=1.0, use_atomtypes=False,
                     out_dir=str(tmp_path), **kw)
    a.run()
    return a


def test_per_type_counts_are_not_overwritten_by_agfe(tmp_cwd, tmp_path):
    """Regression: atomic_grid_free_energy used to store AGFE back into _type_histograms."""
    a = _analysis(tmp_path)
    total = a._histogram
    half = Grid(total.grid / 2.0, edges=total.edges)
    a.use_atomtypes = True
    a._type_histograms = {"Car": half, "Cal": Grid(total.grid - half.grid, edges=total.edges)}
    a._n_atoms_by_type = {"Car": 3, "Cal": 3}
    before = {k: g.grid.copy() for k, g in a._type_histograms.items()}

    a.atomic_grid_free_energy(T)
    for k in before:
        np.testing.assert_array_equal(a._type_histograms[k].grid, before[k])

    a.export_histogram(str(tmp_path / f"map_counts_{COS}.dx"))
    a.export_atomic_grid_free_energy(str(tmp_path / f"map_agfe_{COS}.dx"))
    np.testing.assert_allclose(_read(tmp_path, f"map_counts_Car_{COS}.dx").grid, before["Car"])
    assert _read(tmp_path, f"map_agfe_Car_{COS}.dx").grid.max() <= 0.0, "AGFE, clipped at 0"
    assert os.path.isfile(tmp_path / f"map_counts_{COS}.dx"), "total counts written too"


def test_per_type_name_needs_its_token(tmp_cwd, tmp_path):
    """A bare str.replace used to write every type to the same file when the token was absent."""
    a = _analysis(tmp_path)
    a.use_atomtypes = True
    a._type_histograms = {"Car": a._histogram}
    with pytest.raises(ValueError, match="map_counts"):
        a.export_histogram(str(tmp_path / "counts.dx"))


def test_failed_atomtype_matching_still_writes_an_agfe_map(tmp_cwd, tmp_path, monkeypatch):
    """The fallback kept use_atomtypes=True, so the per-type loop ran empty and wrote nothing."""
    from cosolvkit.analysis.core.grid import GridAnalysis
    monkeypatch.setattr(GridAnalysis, "_map_atomtypes", lambda self, defs: None)
    u = _make_universe()
    a = GridAnalysis(u.select_atoms(f"resname {COS}"), gridsize=1.0, use_atomtypes=True,
                     atomtypes_definitions=[{"atype": "X", "smarts": "[#99]"}],
                     out_dir=str(tmp_path))
    a.run()
    a.atomic_grid_free_energy(T)
    a.export_atomic_grid_free_energy(str(tmp_path / f"map_agfe_{COS}.dx"))
    assert os.path.isfile(tmp_path / f"map_agfe_{COS}.dx")


# --- fixed export region ----------------------------------------------------

BOX = dict(grid_center=[3.3, 4.1, 5.0], grid_size=[7.0, 6.2, 8.0])
NAMES = [f"map_agfe_{COS}.dx", f"map_agfe_raw_{COS}.dx", f"map_counts_{COS}.dx",
         f"solvent_accessible_map_{COS}.dx"]


def test_fixed_region_gives_identical_grids_for_different_boxes(tmp_cwd, tmp_path):
    free_a = _maps(_make_universe(box_size=10.0), tmp_path / "fa")
    free_b = _maps(_make_universe(box_size=12.0), tmp_path / "fb")
    assert _read(free_a, NAMES[0]).grid.shape != _read(free_b, NAMES[0]).grid.shape, \
        "precondition: the two boxes produce different natural grids"

    a = _maps(_make_universe(box_size=10.0), tmp_path / "a", **BOX)
    b = _maps(_make_universe(box_size=12.0), tmp_path / "b", **BOX)
    for name in NAMES:
        ga, gb = _read(a, name), _read(b, name)
        assert ga.grid.shape == gb.grid.shape, name
        np.testing.assert_allclose(ga.origin, gb.origin, atol=1e-9)
        np.testing.assert_allclose(ga.delta, gb.delta, atol=1e-12)


def test_fixed_region_is_snapped_to_the_lattice(tmp_cwd, tmp_path):
    out = _maps(_make_universe(), tmp_path / "out", **BOX)
    g = _read(out, f"map_agfe_{COS}.dx")
    np.testing.assert_allclose(np.asarray(g.origin) / 1.0, np.round(g.origin), atol=1e-9)
    lo = np.asarray(BOX["grid_center"]) - np.asarray(BOX["grid_size"]) / 2
    hi = np.asarray(BOX["grid_center"]) + np.asarray(BOX["grid_size"]) / 2
    top = np.asarray(g.origin) + np.asarray(g.grid.shape) * np.asarray(g.delta)
    assert np.all(np.asarray(g.origin) <= lo + 1e-9) and np.all(top >= hi - 1e-9), \
        "snapped outward: the requested region is fully covered"


def test_cropping_changes_no_value(tmp_cwd, tmp_path):
    """The fixed-region map equals the same voxels of the unrestricted map, bit for bit."""
    free = _maps(_make_universe(), tmp_path / "free")
    fixed = _maps(_make_universe(), tmp_path / "fixed", **BOX)
    for name in (f"map_agfe_raw_{COS}.dx", f"map_counts_{COS}.dx"):
        gf, gx = _read(free, name), _read(fixed, name)
        start = np.round((np.asarray(gx.origin) - np.asarray(gf.origin)) / 1.0).astype(int)
        assert np.all(start >= 0), "region inside the natural grid for this check"
        sl = tuple(slice(s, s + n) for s, n in zip(start, gx.grid.shape))
        np.testing.assert_allclose(gx.grid, gf.grid[sl], atol=1e-5)


def test_region_larger_than_the_system_is_padded_not_truncated(tmp_cwd, tmp_path):
    big = dict(grid_center=[5.0, 5.0, 5.0], grid_size=[30.0, 30.0, 30.0])
    out = _maps(_make_universe(), tmp_path / "out", **big)
    counts = _read(out, f"map_counts_{COS}.dx").grid
    assert counts.shape == (30, 30, 30)
    assert counts.sum() == pytest.approx(30)


@pytest.mark.parametrize("kw, match", [
    (dict(grid_center=[0, 0, 0]), "together"),
    (dict(grid_size=[1, 1, 1]), "together"),
    (dict(grid_center=[0, 0], grid_size=[1, 1, 1]), "three values"),
    (dict(grid_center=[0, 0, 0], grid_size=[1, -1, 1]), "positive"),
])
def test_bad_region_is_rejected(kw, match):
    from cosolvkit.analysis.core.grid import _validate_fixed_box
    from cosolvkit.analysis.config import DensityMapsConfig
    with pytest.raises(ValueError, match=match):
        _validate_fixed_box(kw.get("grid_center"), kw.get("grid_size"))
    with pytest.raises(ValueError, match=match):
        DensityMapsConfig(**kw)
