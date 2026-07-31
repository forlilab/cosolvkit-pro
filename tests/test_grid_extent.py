"""The density grid must contain every observed position, and must not inflate the reference.

Aligned trajectories rotate coordinates without rotating the box vectors, so wrapped solvent ends
up occupying a *rotated* box while the density grid is axis-aligned. Measured on FosAKP benzene
r0: probe positions span 81-84 A per frame against a 73.7 A box, and ``np.histogramdd`` silently
discarded **16.32%** of atom-frames (histogram total 371,538 of 444,000). That wastes ~16% of the
simulation and biases every AGFE value by +0.106 kcal/mol, because ``N_o = n_atoms /
n_accessible_voxels`` still assumes all atoms landed inside.

Enlarging the grid alone would swap one bias for a bigger one: ``_build_accessible_mask`` marks a
voxel accessible if solvent was *ever* observed there, so the corners the rotated box sweeps
through would count as accessible, inflating the reference volume by up to ~1.5x and shifting AGFE
~0.20 kcal/mol in the other direction. Hence two requirements, tested here:

1. The grid is the union of the box-derived grid and the observed extent — never smaller than
   before (so ordinary wrapped trajectories are untouched), larger only when atoms fall outside.
2. The accessible voxel count is capped at the box's own solvent volume, which is a physical
   bound: the solvent cannot occupy more volume than the periodic box minus the protein.
"""

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

BOX = 20.0
GRIDSIZE = 1.0


def _universe(probe_offset=0.0, n_frames=4):
    """6 BEN atoms + many HOH oxygens in a BOX cube; probe_offset pushes BEN outside the box.

    The water count is deliberately large so the mean centre-of-geometry (which sets the grid
    centre) is dominated by solvent and stays put when the probe is displaced — as in a real box,
    where 11,924 waters fix the centre and only the aligned probe spills out.
    """
    n_ben, n_hoh = 6, 400
    n_res = 1 + n_hoh
    u = mda.Universe.empty(n_ben + n_hoh, n_residues=n_res, n_segments=1,
                           atom_resindex=[0] * n_ben + list(range(1, n_res)),
                           residue_segindex=[0] * n_res, trajectory=True)
    u.add_TopologyAttr("name", [f"C{(i % 6) + 1}" for i in range(n_ben)] + ["O"] * n_hoh)
    u.add_TopologyAttr("resname", ["BEN"] + ["HOH"] * n_hoh)
    u.add_TopologyAttr("resid", list(range(1, n_res + 1)))

    rng = np.random.default_rng(0)
    ben = np.array([[6.0, 6.0, 6.0], [7.0, 6.0, 6.0], [6.0, 7.0, 6.0],
                    [6.0, 6.0, 7.0], [7.0, 7.0, 6.0], [6.0, 7.0, 7.0]]) + probe_offset
    hoh = rng.uniform(2.0, BOX - 2.0, size=(n_hoh, 3))
    pos = np.tile(np.vstack([ben, hoh]), (n_frames, 1, 1))
    u.load_new(pos, order="fac", format=MemoryReader,
               dimensions=np.array([[BOX] * 3 + [90.0] * 3] * n_frames))
    return u


def _run(u):
    from cosolvkit.analysis.core.grid import GridAnalysis
    an = GridAnalysis(u.select_atoms("resname BEN"), gridsize=GRIDSIZE,
                      use_atomtypes=False, verbose=False)
    an.run()
    return an


def test_grid_matches_the_box_when_every_atom_is_inside(tmp_cwd):
    """Backwards compatibility: ordinary wrapped trajectories must be unaffected."""
    an = _run(_universe(probe_offset=0.0))
    expected = int(round(BOX / GRIDSIZE))
    assert an._histogram.grid.shape == (expected,) * 3


def test_nothing_is_dropped_when_every_atom_is_inside(tmp_cwd):
    an = _run(_universe(probe_offset=0.0))
    assert an._histogram.grid.sum() == pytest.approx(an._n_atoms * an._nframes)


def test_grid_expands_to_contain_atoms_outside_the_box(tmp_cwd):
    """The rotated-box case: probe sits well outside the axis-aligned box."""
    an = _run(_universe(probe_offset=BOX))          # BEN pushed a full box length out
    assert an._histogram.grid.shape[0] > int(round(BOX / GRIDSIZE))


def test_no_positions_are_dropped_when_atoms_fall_outside_the_box(tmp_cwd):
    """The actual bug: histogram total must equal n_atoms x n_frames regardless."""
    an = _run(_universe(probe_offset=BOX))
    assert an._histogram.grid.sum() == pytest.approx(an._n_atoms * an._nframes), \
        "positions outside the box-derived grid were discarded"


def test_out_of_box_fraction_is_recorded(tmp_cwd):
    """The pipeline must be able to say how bad the spill was."""
    inside = _run(_universe(probe_offset=0.0))
    outside = _run(_universe(probe_offset=BOX))
    assert inside._frac_outside_box == pytest.approx(0.0)
    assert outside._frac_outside_box > 0.5


def test_accessible_voxels_never_exceed_the_box_solvent_volume(tmp_cwd):
    """Physical bound: solvent cannot occupy more than the box minus the protein."""
    an = _run(_universe(probe_offset=BOX))
    box_voxels = np.prod(np.asarray(an._box_size) / GRIDSIZE)
    assert an._n_accessible_voxels <= box_voxels


def test_reference_volume_is_not_inflated_by_the_expansion(tmp_cwd):
    """N_o must not collapse just because the grid got bigger."""
    a = _run(_universe(probe_offset=0.0))
    b = _run(_universe(probe_offset=BOX))
    # same box, same atom count -> the reference density should be comparable, not 1.5x apart
    ratio = b._n_accessible_voxels / a._n_accessible_voxels
    assert 0.5 < ratio < 2.0, f"reference volume moved by {ratio:.2f}x on expansion"


def test_expanded_grid_still_contains_the_original_box(tmp_cwd):
    """Expansion must be a superset, never a shift that loses the box interior."""
    an = _run(_universe(probe_offset=BOX))
    origin = np.asarray(an._histogram.origin, dtype=float)
    far = origin + np.asarray(an._histogram.grid.shape) * GRIDSIZE
    centre = np.asarray(an._center, dtype=float)
    box_lo = centre - np.asarray(an._box_size) / 2.0
    box_hi = centre + np.asarray(an._box_size) / 2.0
    assert np.all(origin <= box_lo + 1e-6)
    assert np.all(far >= box_hi - 1e-6)
