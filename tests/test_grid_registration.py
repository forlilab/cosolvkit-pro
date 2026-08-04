"""Maps of the same protein must be *registered* to each other, whatever the sampling.

Motivating measurement (FosAKP benzene, 10 replicas = 2 independent solvations x 5 seeds,
100 and 250 ns). Decomposing the grid frame produced by ``GridAnalysis._conclude`` gave:

    protein CA centre of geometry (the alignment target) : spread 0.006 - 0.010 A
    whole-system centre of geometry (sets the grid centre): spread 0.037 - 0.050 A
    mean box size                                        : spread 0.072 A
    voxel delta                                          : spread 0.0008 A
    ---------------------------------------------------------------------------
    grid ORIGIN                                          : spread 8.1 - 10.3 A
    grid SHAPE                                           : spread 19 - 27 voxels

The alignment is excellent and every smooth term is sub-voxel, yet the origin moves by
ten Angstrom. The cause is ``_conclude``'s padding, which is driven by ``flat.min(axis=0)``
and ``flat.max(axis=0)`` -- the single most extreme excursion of any cosolvent heavy atom
over the whole trajectory. An extreme-value statistic over one outlier atom was defining the
coordinate system, so it is irreproducible across replicas *and* grows with trajectory
length.

The consequence is not the differing shape -- it is that the origins differ by a
*non-integer* number of voxels and ``delta`` differs in the 4th decimal. Two such grids
cannot be combined by array arithmetic, so every merge went through
``combine_dx_maps_with_resampling`` and paid trilinear interpolation. Merging maps from two
independent solvations then scored *below either solvation alone*, which was misread as
"independent solvations must not be merged" when it was really a registration defect.

The fix is to put every grid on one global lattice: ``delta == gridsize`` exactly and every
edge an integer multiple of ``gridsize``. The extent stays demand-driven (the contract in
``test_grid_extent.py`` is preserved), but any two grids are then *commensurate* -- related by
an integer index shift -- so they can be combined exactly, with no interpolation.
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

GRIDSIZE = 1.0


def _universe(probe_offset=0.0, box=20.0, n_frames=4, seed=0, jitter=0.0):
    """6 BEN atoms + 400 HOH oxygens in a cubic box.

    *probe_offset* pushes the probe outside the box (the aligned-trajectory spill).
    *jitter* perturbs the box vectors, mimicking independent NPT runs whose mean box
    differs in the second decimal. *seed* changes the water configuration, mimicking an
    independent solvation.
    """
    n_ben, n_hoh = 6, 400
    n_res = 1 + n_hoh
    u = mda.Universe.empty(n_ben + n_hoh, n_residues=n_res, n_segments=1,
                           atom_resindex=[0] * n_ben + list(range(1, n_res)),
                           residue_segindex=[0] * n_res, trajectory=True)
    u.add_TopologyAttr("name", [f"C{(i % 6) + 1}" for i in range(n_ben)] + ["O"] * n_hoh)
    u.add_TopologyAttr("resname", ["BEN"] + ["HOH"] * n_hoh)
    u.add_TopologyAttr("resid", list(range(1, n_res + 1)))

    rng = np.random.default_rng(seed)
    ben = np.array([[6.0, 6.0, 6.0], [7.0, 6.0, 6.0], [6.0, 7.0, 6.0],
                    [6.0, 6.0, 7.0], [7.0, 7.0, 6.0], [6.0, 7.0, 7.0]]) + probe_offset
    hoh = rng.uniform(2.0, box - 2.0, size=(n_hoh, 3))
    pos = np.tile(np.vstack([ben, hoh]), (n_frames, 1, 1))
    dims = np.array([[box + jitter] * 3 + [90.0] * 3] * n_frames)
    u.load_new(pos, order="fac", format=MemoryReader, dimensions=dims)
    return u


def _run(u, gridsize=GRIDSIZE):
    from cosolvkit.analysis.core.grid import GridAnalysis
    an = GridAnalysis(u.select_atoms("resname BEN"), gridsize=gridsize,
                      use_atomtypes=False, verbose=False)
    an.run()
    return an


def _edges(an):
    return [np.asarray(e, dtype=float) for e in an._edges]


# --------------------------------------------------------------------------------------
# 1. delta must be exactly the requested voxel size
# --------------------------------------------------------------------------------------

def test_delta_equals_the_requested_gridsize_even_for_an_incommensurate_box(tmp_cwd):
    """``delta = box_size / round(box_size / gridsize)`` made the voxel box-dependent.

    A 20.3 A box at 1.0 A voxels gave delta = 1.015; on FosAKP the nominal 0.8 came out as
    0.80142303 from a mean box of 73.73092.

    The concrete harm was an inconsistency rather than a misalignment: ``self._edges`` were
    spaced by that value and ``np.histogramdd`` binned at it, but the Grid was then labelled
    ``delta=self._gridsize`` (0.8). Every stored map therefore claims a 0.8 voxel while its
    contents were binned at 0.80142 -- a 0.18% scale error accumulating to ~0.22 A by the far
    edge of a 153-voxel axis. Binning and labelling now use one value.

    Note this is NOT what forced the historical resamples; differing SHAPE was.
    """
    an = _run(_universe(box=20.3))
    assert np.allclose(an._delta, GRIDSIZE, atol=1e-12), (
        f"voxel size {an._delta} is box-dependent, not the requested {GRIDSIZE}")


# --------------------------------------------------------------------------------------
# 2. every edge on a global lattice
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("probe_offset,box,jitter,seed", [
    (0.0, 20.0, 0.0, 0),
    (20.0, 20.0, 0.0, 0),      # probe spills out -> padding path
    (20.0, 20.3, 0.07, 1),     # incommensurate box, jittered, different solvation
])
def test_grid_edges_lie_on_the_global_lattice(tmp_cwd, probe_offset, box, jitter, seed):
    """Edges must be integer multiples of gridsize, so all grids share one lattice."""
    an = _run(_universe(probe_offset=probe_offset, box=box, jitter=jitter, seed=seed))
    for d, e in enumerate(_edges(an)):
        k = e / GRIDSIZE
        assert np.allclose(k, np.round(k), atol=1e-9), (
            f"axis {d} edges are off-lattice; first three are {e[:3]}")


# --------------------------------------------------------------------------------------
# 3. the actual defect: two runs of the same system must be commensurate
# --------------------------------------------------------------------------------------

def test_two_replicas_of_the_same_system_are_commensurate(tmp_cwd):
    """Origins may differ, but only by a whole number of voxels.

    This is the property that makes merging exact. It failed before the fix: the padding
    was driven by the probe's extreme excursion, so replicas differed by a fractional
    voxel offset and had to be interpolated onto each other.
    """
    a = _run(_universe(probe_offset=20.0, box=20.0, jitter=0.00, seed=0))
    b = _run(_universe(probe_offset=13.0, box=20.3, jitter=0.07, seed=1))
    off = (np.asarray(a._histogram.origin, float)
           - np.asarray(b._histogram.origin, float)) / GRIDSIZE
    assert np.allclose(off, np.round(off), atol=1e-9), (
        f"replica origins differ by a fractional voxel offset {off}; "
        "the maps cannot be combined without interpolation")
    assert np.allclose(a._delta, b._delta, atol=1e-12)


# --------------------------------------------------------------------------------------
# 4. combining commensurate grids must be exact, and must not dilute partial coverage
# --------------------------------------------------------------------------------------

def _write(tmp_path, name, values, origin, delta=1.0):
    from gridData import Grid
    g = Grid(np.asarray(values, dtype=float), origin=np.asarray(origin, float),
             delta=np.full(3, float(delta)))
    p = tmp_path / name
    g.export(str(p))
    return str(p)


def test_commensurate_maps_combine_without_interpolation(tmp_path):
    """Values in the overlap must be carried over bit-for-bit.

    Trilinear interpolation onto a half-voxel-offset grid smears a sharp well across its
    neighbours, which is what shallowed out merged AGFE maps. On the lattice the offset is
    an integer, so the merge is a pure index shift.
    """
    from cosolvkit.analysis.core.grid import combine_dx_maps_with_resampling

    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6, 6))
    pa = _write(tmp_path, "a.dx", a, origin=(0.0, 0.0, 0.0))
    # Same values, same lattice, extent shifted by exactly 2 voxels.
    pb = _write(tmp_path, "b.dx", a, origin=(2.0, 2.0, 2.0))

    out = combine_dx_maps_with_resampling(
        [pa, pb], method="mean", resample_to="first",
        out_fname=str(tmp_path / "m.dx"))

    # Region covered by both: b's voxel [i-2] sits on a's voxel [i], so the mean of the two
    # is NOT a again -- but it must be an exact average of two stored values, never a
    # smeared one. Check against the exact integer-shift expectation.
    expected = a.copy()
    expected[2:, 2:, 2:] = 0.5 * (a[2:, 2:, 2:] + a[:-2, :-2, :-2])
    assert np.allclose(out.grid, expected, atol=1e-12), \
        "combined map does not equal the exact integer-shift average"


def test_partial_coverage_is_not_diluted_toward_the_fill_value(tmp_path):
    """A voxel only one map reaches must keep that map's value, not be halved toward 0.

    With ``fill_value=0.0`` and plain averaging, the rotated-box corners that only some
    replicas' grids reach were pulled toward bulk, weakening real density near the grid
    margins in proportion to how many replicas covered it.
    """
    from cosolvkit.analysis.core.grid import combine_dx_maps_with_resampling

    a = np.full((6, 6, 6), -4.0)
    b = np.full((6, 6, 6), -4.0)
    pa = _write(tmp_path, "a.dx", a, origin=(0.0, 0.0, 0.0))
    pb = _write(tmp_path, "b.dx", b, origin=(3.0, 3.0, 3.0))

    out = combine_dx_maps_with_resampling(
        [pa, pb], method="mean", resample_to="first",
        out_fname=str(tmp_path / "m.dx"))

    assert np.allclose(out.grid, -4.0, atol=1e-12), (
        "voxels covered by only one map were averaged against the fill value; "
        f"min={out.grid.min()} max={out.grid.max()}")


def test_incommensurate_legacy_maps_still_fall_back_to_resampling(tmp_path):
    """Maps written before the lattice fix have off-lattice origins; they must still combine."""
    from cosolvkit.analysis.core.grid import (
        combine_dx_maps_with_resampling, _lattice_commensurate, _read_dx)

    a = np.zeros((6, 6, 6))
    pa = _write(tmp_path, "a.dx", a, origin=(0.0, 0.0, 0.0))
    pb = _write(tmp_path, "b.dx", a, origin=(0.37, 0.0, 0.0))  # fractional voxel offset
    assert not _lattice_commensurate([_read_dx(pa), _read_dx(pb)])
    out = combine_dx_maps_with_resampling(
        [pa, pb], method="mean", resample_to="first",
        out_fname=str(tmp_path / "m.dx"))
    assert out.grid.shape == (6, 6, 6)


def test_padding_does_not_depend_on_a_single_outlier_atom(tmp_cwd):
    """One atom wandering further must not move the coordinate system of the interior.

    The grid may grow to contain the outlier, but the voxel *lattice* must be unchanged,
    so the interior voxels keep their physical positions.
    """
    near = _run(_universe(probe_offset=11.0))
    far = _run(_universe(probe_offset=40.0))
    off = (np.asarray(near._histogram.origin, float)
           - np.asarray(far._histogram.origin, float)) / GRIDSIZE
    assert np.allclose(off, np.round(off), atol=1e-9), (
        f"an outlier atom shifted the lattice by {off} voxels")
