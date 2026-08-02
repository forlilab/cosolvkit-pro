import numpy as np
from cosolvkit.analysis.core.grid import resample_mask_to_grid, grids_aligned


def test_resample_translated_grid_preserves_occupancy():
    # source mask on grid A; reference grid B shifted by +0.5 Å (=1 voxel) in x.
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2, 2, 2] = True
    src_o = np.array([0.0, 0.0, 0.0]); d = np.array([0.5, 0.5, 0.5])
    ref_o = np.array([0.5, 0.0, 0.0])  # ref voxel (i,j,k) is at src (i+1,j,k)
    out = resample_mask_to_grid(mask, src_o, d, ref_o, d, (6, 6, 6))
    # src voxel (2,2,2) is at Å (1.0,1.0,1.0); on ref grid that is index (1,2,2)
    assert out.dtype == bool
    assert out[1, 2, 2] == True
    assert out.sum() == 1


def test_grids_aligned():
    o = np.array([0.0, 0.0, 0.0]); d = np.array([0.5, 0.5, 0.5])
    assert grids_aligned(o, d, (6, 6, 6), o, d, (6, 6, 6))
    assert not grids_aligned(o, d, (6, 6, 6), o + 0.5, d, (6, 6, 6))
