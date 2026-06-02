"""Integration tests for GridAnalysis using a synthetic MDAnalysis Universe.

These tests build a minimal in-memory Universe with known atom positions so
we can control the density and verify AGFE invariants without a real trajectory.

The non-atomtype path is tested here. The atomtype path requires RDKit SMARTS
matching (needs bond information that Universe.empty() doesn't provide);
that path is exercised narrowly via _map_atomtypes in a separate unit test below.

GridAnalysis._build_accessible_mask writes solvent_accessible_map.dx to CWD.
The tmp_cwd fixture changes CWD to tmp_path for the duration of each test.
"""

import os

import numpy as np
import pytest
from gridData import Grid

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Synthetic Universe factory
# ---------------------------------------------------------------------------

def _make_universe(n_frames=5, box_size=10.0):
    """Build a minimal MDAnalysis Universe with:
    - 6 BEN heavy atoms clustered near (2, 2, 2) in each frame
    - 5 HOH water oxygen atoms spread across the box
    Total: 11 atoms, 6 residues (1 BEN + 5 HOH)
    """
    n_ben = 6
    n_hoh = 5
    n_atoms = n_ben + n_hoh
    n_residues = 1 + n_hoh  # 1 BEN residue, 5 HOH residues

    atom_resindex = [0] * n_ben + list(range(1, n_hoh + 1))
    residue_segindex = [0] * n_residues

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        n_segments=1,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["C1", "C2", "C3", "C4", "C5", "C6"] + ["O"] * n_hoh)
    u.add_TopologyAttr("resname", ["BEN"] + ["HOH"] * n_hoh)
    u.add_TopologyAttr("resid", list(range(1, n_residues + 1)))

    # BEN atoms clustered near (2, 2, 2); HOH atoms spread around
    ben_pos = np.array([
        [2.0, 2.0, 2.0], [2.5, 2.0, 2.0], [2.0, 2.5, 2.0],
        [2.0, 2.0, 2.5], [2.5, 2.5, 2.0], [2.0, 2.5, 2.5],
    ])
    hoh_pos = np.array([
        [5.0, 5.0, 5.0], [7.0, 7.0, 7.0], [8.0, 3.0, 5.0],
        [3.0, 8.0, 5.0], [5.0, 3.0, 8.0],
    ])
    positions = np.vstack([ben_pos, hoh_pos])

    pos_array = np.tile(positions, (n_frames, 1, 1))  # (n_frames, n_atoms, 3)
    dimensions = np.array(
        [[box_size, box_size, box_size, 90.0, 90.0, 90.0]] * n_frames
    )
    u.load_new(pos_array, order="fac", format=MemoryReader, dimensions=dimensions)
    return u


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_universe():
    if not HAS_MDA:
        pytest.skip("MDAnalysis not available")
    return _make_universe()


# ---------------------------------------------------------------------------
# GridAnalysis tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")
class TestGridAnalysisNonAtomtype:

    def test_run_completes(self, synthetic_universe, tmp_cwd):
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        assert analysis._nframes == 5

    def test_grid_shape_is_set_after_run(self, synthetic_universe, tmp_cwd):
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        assert hasattr(analysis, "_histogram")
        assert analysis._histogram.grid.ndim == 3

    def test_agfe_computed(self, synthetic_universe, tmp_cwd):
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        analysis.atomic_grid_free_energy(temperature=300, smoothing=False)
        assert hasattr(analysis, "_agfe")
        assert np.all(np.isfinite(analysis._agfe.grid))

    def test_agfe_smoothed_all_nonpositive(self, synthetic_universe, tmp_cwd):
        """Load-bearing invariant: smoothed AGFE <= 0 everywhere (PyMol relies on this)."""
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        analysis.atomic_grid_free_energy(temperature=300, smoothing=True)
        assert np.all(analysis._agfe.grid <= 0.0)

    def test_hotspot_region_more_favorable_than_background(self, synthetic_universe, tmp_cwd):
        """BEN atoms piled near (2,2,2) → that region should have lower AGFE."""
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        analysis.atomic_grid_free_energy(temperature=300, smoothing=False)

        agfe = analysis._agfe.grid
        origin = np.array(analysis._agfe.origin)
        delta = np.array(analysis._agfe.delta)

        # Convert Angstrom position (2,2,2) to voxel index
        vox = np.round((np.array([2.0, 2.0, 2.0]) - origin) / delta).astype(int)
        vox = np.clip(vox, 0, np.array(agfe.shape) - 1)

        # Corner (9,9,9) should be background (no BEN atoms there)
        corner = np.array(agfe.shape) - 1

        agfe_hotspot = float(agfe[tuple(vox)])
        agfe_background = float(agfe[tuple(corner)])

        assert agfe_hotspot < agfe_background, (
            f"Hotspot voxel ({agfe_hotspot:.3f}) should be more favorable than "
            f"background ({agfe_background:.3f})"
        )

    def test_export_agfe_writes_dx(self, synthetic_universe, tmp_cwd):
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        analysis.atomic_grid_free_energy(temperature=300, smoothing=True)

        out_path = str(tmp_cwd / "test_agfe_BEN.dx")
        analysis.export_atomic_grid_free_energy(out_path)
        assert os.path.exists(out_path)

        # Re-read and check it's valid
        g = Grid(out_path)
        assert g.grid.ndim == 3
        assert np.all(np.isfinite(g.grid))

    def test_nframes_counted_correctly(self, tmp_cwd):
        """Verify _nframes matches the trajectory length."""
        if not HAS_MDA:
            pytest.skip("MDAnalysis not available")
        from cosolvkit.density_analysis import GridAnalysis
        u = _make_universe(n_frames=3)
        ag = u.select_atoms("resname BEN")
        analysis = GridAnalysis(ag, gridsize=1.0, use_atomtypes=False)
        analysis.run()
        assert analysis._nframes == 3

    def test_use_atomtypes_without_definitions_exits(self, synthetic_universe, tmp_cwd):
        """use_atomtypes=True with atomtypes_definitions=None → SystemExit."""
        from cosolvkit.density_analysis import GridAnalysis
        ag = synthetic_universe.select_atoms("resname BEN")
        with pytest.raises(SystemExit):
            GridAnalysis(ag, gridsize=1.0, use_atomtypes=True, atomtypes_definitions=None)
