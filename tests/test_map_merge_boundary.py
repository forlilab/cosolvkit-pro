"""Merging AGFE maps must not manufacture favourable density at the grid faces.

Regression test for a bug found while benchmarking against crystallographic sites:
replicas of one system have origins differing by a few hundredths of an Angstrom, which
failed the old absolute 1e-3 A alignment test and sent every map through
``gridData.Grid.resample``. gridData fills out-of-range voxels with ``grid.min()`` --
the most *favourable* value for a free-energy map -- so the whole outer voxel layer came
back at the map minimum. On a 148^3 grid that is ~87,000 spurious favourable voxels
forming a shell on all six faces, which then merged into one connected cluster holding
~99% of the favourable volume and ranked as the top binding site.
"""

import numpy as np
from gridData import Grid

from cosolvkit.analysis.core.grid import (
    _grids_spatially_aligned,
    _resample_grid,
    combine_dx_maps_with_resampling,
)

SHAPE = (20, 20, 20)
DELTA = 0.5


def _agfe_grid(origin_shift=0.0, well_value=-3.0):
    """An AGFE-like map: 0.0 everywhere (bulk) with one favourable well inside."""
    data = np.zeros(SHAPE, dtype=float)
    data[9:12, 9:12, 9:12] = well_value
    origin = np.array([-5.0, -5.0, -5.0]) + origin_shift
    edges = [origin[d] + np.arange(SHAPE[d] + 1) * DELTA for d in range(3)]
    return Grid(data, edges=edges)


def _face_mask(shape):
    i, j, k = np.indices(shape)
    n = np.array(shape)
    d = np.minimum.reduce([i, j, k, n[0] - 1 - i, n[1] - 1 - j, n[2] - 1 - k])
    return d == 0


# ---------------------------------------------------------------------------
# Alignment tolerance
# ---------------------------------------------------------------------------

def test_subvoxel_origin_offset_counts_as_aligned():
    a = _agfe_grid(0.0)
    b = _agfe_grid(0.023)          # the real offset seen between replicas
    assert _grids_spatially_aligned(a, b)


def test_offset_of_half_a_voxel_is_not_aligned():
    a = _agfe_grid(0.0)
    b = _agfe_grid(0.25)           # half a voxel -> genuinely different registration
    assert not _grids_spatially_aligned(a, b)


# ---------------------------------------------------------------------------
# Resampling fill value
# ---------------------------------------------------------------------------

def test_resample_fills_out_of_range_with_neutral_not_minimum():
    """The bug in isolation: gridData would fill the faces with grid.min()."""
    ref = _agfe_grid(0.0)
    shifted = _agfe_grid(0.25)
    out = _resample_grid(shifted, ref, fill_value=0.0)
    faces = _face_mask(out.shape)
    assert out[faces].min() >= -1e-6, (
        "resampling put favourable AGFE on the grid faces")
    # The genuine well must survive the resample.
    assert out.min() < -1.0


def test_resample_is_identity_for_the_same_grid():
    """Guards the index arithmetic in _resample_grid."""
    g = _agfe_grid(0.0)
    out = _resample_grid(g, g, fill_value=0.0)
    np.testing.assert_allclose(out, g.grid, atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end merge
# ---------------------------------------------------------------------------

def test_merge_never_creates_favourable_faces(tmp_path):
    """Half a voxel of offset is the case that takes the resample path."""
    shift = 0.25
    paths = []
    for idx, s in enumerate((0.0, shift)):
        p = tmp_path / f"map_{idx}.dx"
        _agfe_grid(s).export(str(p))
        paths.append(str(p))

    merged = combine_dx_maps_with_resampling(
        paths, method="mean", resample_to="first",
        out_fname=str(tmp_path / "combined.dx"))

    faces = _face_mask(merged.grid.shape)
    n_fav_faces = int((merged.grid[faces] < -0.596).sum())
    assert n_fav_faces == 0, (
        f"{n_fav_faces} face voxels came back favourable for shift={shift}")
    # The real well is still there.
    assert merged.grid.min() < -1.0


def test_merge_of_subvoxel_offset_replicas_matches_plain_mean(tmp_path):
    """Sub-voxel offsets take the no-resample path, so the result is the plain mean."""
    a, b = _agfe_grid(0.0, well_value=-3.0), _agfe_grid(0.02, well_value=-1.0)
    paths = []
    for idx, g in enumerate((a, b)):
        p = tmp_path / f"m{idx}.dx"
        g.export(str(p))
        paths.append(str(p))
    merged = combine_dx_maps_with_resampling(
        paths, method="mean", resample_to="first",
        out_fname=str(tmp_path / "c.dx"))
    np.testing.assert_allclose(merged.grid, (a.grid + b.grid) / 2.0, atol=1e-9)
