"""Tests for density_analysis.py — pure functions.

All tests in this file run without any MD trajectory or real .dx assets.
"""

import os
import numpy as np
import pytest
from gridData import Grid

from cosolvkit.density_analysis import (
    BOLTZMANN_CONSTANT_KB,
    _grid_free_energy,
    _smooth_grid_free_energy,
    combine_dx_maps,
    combine_dx_maps_with_resampling,
)


# ---------------------------------------------------------------------------
# _grid_free_energy
# ---------------------------------------------------------------------------

class TestGridFreeEnergy:
    """Boltzmann inversion: AGFE = -kT * ln(N_local / N_bulk)."""

    def _bulk_hist(self, n_atoms, n_frames, n_accessible):
        """Uniform occupancy: each voxel gets exactly the bulk density."""
        n_vox = n_accessible
        val = (n_atoms / n_frames) / (n_atoms / n_accessible)  # N / N_o = 1
        hist = np.full((10, 10, 10), n_atoms / n_accessible * n_frames)
        return hist

    def test_uniform_occupancy_gives_zero_agfe(self):
        """When N_local == N_bulk at every voxel, AGFE ~ 0 everywhere."""
        n_atoms = 100
        n_frames = 50
        n_accessible = 1000
        N_o = n_atoms / n_accessible
        hist = np.full((10, 10, 10), N_o * n_frames)
        gfe = _grid_free_energy(hist, n_atoms, n_frames, n_accessible, temperature=300)
        assert np.allclose(gfe, 0.0, atol=1e-6)

    def test_enrichment_gives_negative_agfe(self):
        """Voxels with more atoms than bulk have negative (favorable) AGFE."""
        n_atoms = 100
        n_frames = 50
        n_accessible = 1000
        N_o = n_atoms / n_accessible
        # All voxels have 10x bulk occupancy
        hist = np.full((5, 5, 5), N_o * n_frames * 10)
        gfe = _grid_free_energy(hist, n_atoms, n_frames, n_accessible, temperature=300)
        assert np.all(gfe < 0)

    def test_depletion_gives_positive_agfe(self):
        """Voxels with fewer atoms than bulk have positive (unfavorable) AGFE."""
        n_atoms = 100
        n_frames = 50
        n_accessible = 1000
        N_o = n_atoms / n_accessible
        # All voxels at 10% of bulk
        hist = np.full((5, 5, 5), N_o * n_frames * 0.1)
        gfe = _grid_free_energy(hist, n_atoms, n_frames, n_accessible, temperature=300)
        assert np.all(gfe > 0)

    def test_zero_occupancy_uses_floor_not_inf(self):
        """Voxels with zero occupancy hit the 1e-10 floor; output is finite."""
        hist = np.zeros((5, 5, 5))
        gfe = _grid_free_energy(hist, n_atoms=100, n_frames=50, n_accessible_voxels=1000)
        assert np.all(np.isfinite(gfe))
        assert np.all(gfe > 0)

    def test_formula_matches_analytical_value(self):
        """Verify one cell against the closed-form expression."""
        n_atoms = 100
        n_frames = 50
        n_accessible = 500
        temperature = 300
        N_o = n_atoms / n_accessible       # bulk density
        # Single-voxel array at 2× bulk occupancy
        occupancy = 2 * N_o
        hist = np.array([[[occupancy * n_frames]]])
        gfe = _grid_free_energy(hist, n_atoms, n_frames, n_accessible, temperature)
        expected = -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(2.0)
        assert abs(float(gfe[0, 0, 0]) - expected) < 1e-8

    def test_output_shape_matches_input(self):
        hist = np.ones((8, 6, 4))
        gfe = _grid_free_energy(hist, n_atoms=10, n_frames=10, n_accessible_voxels=100)
        assert gfe.shape == (8, 6, 4)


# ---------------------------------------------------------------------------
# _smooth_grid_free_energy
# ---------------------------------------------------------------------------

class TestSmoothGridFreeEnergy:
    """The PyMol contouring path relies on max(agfe) <= 0 after smoothing."""

    def test_all_values_nonpositive(self):
        """Load-bearing invariant: every smoothed voxel <= 0."""
        rng = np.random.default_rng(42)
        # Mix of negative and positive raw AGFE (before zeroing unfavorable)
        gfe = rng.normal(0, 1, (10, 10, 10))
        smoothed = _smooth_grid_free_energy(gfe, energy_cutoff=0, sigma=1)
        assert np.all(smoothed <= 0.0)

    def test_output_shape_preserved(self):
        gfe = np.full((6, 6, 6), -1.5)
        smoothed = _smooth_grid_free_energy(gfe, sigma=1)
        assert smoothed.shape == (6, 6, 6)

    def test_custom_energy_cutoff(self):
        """With cutoff=-1, only voxels below -1 survive."""
        gfe = np.array([[[-0.5, -1.5, -2.0]]])
        smoothed = _smooth_grid_free_energy(gfe, energy_cutoff=-1.0, sigma=0)
        # sigma=0 disables smoothing influence; voxels >= -1 zeroed
        assert smoothed[0, 0, 0] == 0.0   # -0.5 >= -1 → zeroed
        assert smoothed[0, 0, 1] < 0.0   # -1.5 < -1 → kept

    def test_deep_negative_values_survive(self):
        """Very negative voxels should remain negative after smoothing."""
        gfe = np.full((5, 5, 5), -10.0)
        smoothed = _smooth_grid_free_energy(gfe, energy_cutoff=0, sigma=0)
        assert np.all(smoothed < 0)


# ---------------------------------------------------------------------------
# combine_dx_maps
# ---------------------------------------------------------------------------

class TestCombineDxMaps:

    def _write_grid(self, tmp_path, name, array):
        edges = [np.linspace(0, 5, array.shape[i] + 1) for i in range(3)]
        g = Grid(array, edges=edges)
        fpath = str(tmp_path / name)
        g.export(fpath)
        return fpath

    def test_mean_of_identical_grids(self, tmp_path):
        arr = np.random.default_rng(0).uniform(-2, 0, (6, 6, 6))
        p1 = self._write_grid(tmp_path, "g1.dx", arr)
        p2 = self._write_grid(tmp_path, "g2.dx", arr)
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps([p1, p2], method="mean", out_fname=out)
        assert np.allclose(result.grid, arr, atol=1e-5)

    def test_min_picks_elementwise_minimum(self, tmp_path):
        a = np.array([[[0.0, -1.0]]])
        b = np.array([[[-0.5, -2.0]]])
        p1 = self._write_grid(tmp_path, "a.dx", a)
        p2 = self._write_grid(tmp_path, "b.dx", b)
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps([p1, p2], method="min", out_fname=out)
        assert np.allclose(result.grid, np.array([[[  -0.5, -2.0]]]), atol=1e-5)

    def test_max_picks_elementwise_maximum(self, tmp_path):
        a = np.array([[[0.0, -1.0]]])
        b = np.array([[[-0.5, -2.0]]])
        p1 = self._write_grid(tmp_path, "a.dx", a)
        p2 = self._write_grid(tmp_path, "b.dx", b)
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps([p1, p2], method="max", out_fname=out)
        assert np.allclose(result.grid, np.array([[[0.0, -1.0]]]), atol=1e-5)

    def test_shape_mismatch_raises_valueerror(self, tmp_path):
        a = np.zeros((4, 4, 4))
        b = np.zeros((6, 6, 6))
        p1 = self._write_grid(tmp_path, "a.dx", a)
        p2 = self._write_grid(tmp_path, "b.dx", b)
        with pytest.raises(ValueError):
            combine_dx_maps([p1, p2], out_fname=str(tmp_path / "out.dx"))

    def test_output_file_written(self, tmp_path):
        arr = np.zeros((4, 4, 4))
        p = self._write_grid(tmp_path, "g.dx", arr)
        out = str(tmp_path / "combined.dx")
        combine_dx_maps([p], out_fname=out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# combine_dx_maps_with_resampling
# ---------------------------------------------------------------------------

class TestCombineDxMapsWithResampling:

    def _write_grid(self, tmp_path, name, shape, fill=-1.0):
        arr = np.full(shape, fill)
        edges = [np.linspace(0, 5, s + 1) for s in shape]
        g = Grid(arr, edges=edges)
        fpath = str(tmp_path / name)
        g.export(fpath)
        return fpath

    def test_same_shape_fast_path_mean(self, tmp_path):
        p1 = self._write_grid(tmp_path, "a.dx", (6, 6, 6), fill=-2.0)
        p2 = self._write_grid(tmp_path, "b.dx", (6, 6, 6), fill=-4.0)
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps_with_resampling([p1, p2], method="mean", out_fname=out)
        assert np.allclose(result.grid, -3.0, atol=1e-4)

    def test_resample_to_first(self, tmp_path):
        p1 = self._write_grid(tmp_path, "small.dx", (6, 6, 6))
        p2 = self._write_grid(tmp_path, "large.dx", (10, 10, 10))
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps_with_resampling(
            [p1, p2], method="mean", resample_to="first", out_fname=out
        )
        assert result.grid.shape == (6, 6, 6)

    def test_resample_to_largest(self, tmp_path):
        p1 = self._write_grid(tmp_path, "small.dx", (6, 6, 6))
        p2 = self._write_grid(tmp_path, "large.dx", (10, 10, 10))
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps_with_resampling(
            [p1, p2], method="mean", resample_to="largest", out_fname=out
        )
        assert result.grid.shape == (10, 10, 10)

    def test_resample_to_smallest(self, tmp_path):
        p1 = self._write_grid(tmp_path, "small.dx", (6, 6, 6))
        p2 = self._write_grid(tmp_path, "large.dx", (10, 10, 10))
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps_with_resampling(
            [p1, p2], method="mean", resample_to="smallest", out_fname=out
        )
        assert result.grid.shape == (6, 6, 6)

    def test_unknown_resample_to_raises(self, tmp_path):
        p = self._write_grid(tmp_path, "a.dx", (4, 4, 4))
        with pytest.raises(ValueError, match="resample_to"):
            combine_dx_maps_with_resampling(
                [p], resample_to="bogus", out_fname=str(tmp_path / "out.dx")
            )

    def test_unknown_method_raises(self, tmp_path):
        p = self._write_grid(tmp_path, "a.dx", (4, 4, 4))
        with pytest.raises(ValueError, match="method"):
            combine_dx_maps_with_resampling(
                [p], method="harmonic", out_fname=str(tmp_path / "out.dx")
            )

    def test_output_file_written(self, tmp_path):
        p = self._write_grid(tmp_path, "a.dx", (4, 4, 4))
        out = str(tmp_path / "combined.dx")
        combine_dx_maps_with_resampling([p], out_fname=out)
        assert os.path.exists(out)

    def test_single_file_passthrough(self, tmp_path):
        """A single input should return a grid equal to that input."""
        arr = np.random.default_rng(7).uniform(-2, 0, (5, 5, 5))
        edges = [np.linspace(0, 2.5, 6)] * 3
        g = Grid(arr, edges=edges)
        p = str(tmp_path / "only.dx")
        g.export(p)
        out = str(tmp_path / "out.dx")
        result = combine_dx_maps_with_resampling([p], out_fname=out)
        assert np.allclose(result.grid, arr, atol=1e-5)
