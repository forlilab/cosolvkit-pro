#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Grid creation, analysis, and manipulation
#

import os
import sys
import logging
import warnings
from glob import glob
import numpy as np
from typing import List, Union

from scipy.ndimage import gaussian_filter, binary_dilation
from gridData import Grid

from MDAnalysis.analysis.base import AnalysisBase


BOLTZMANN_CONSTANT_KB = 0.0019872041  # kcal/(mol*K)

def _read_dx(filepath: str = None) -> Grid:
    """Reads a .dx map using gridData.Grid."""
    return Grid(str(filepath))

# Fraction of a voxel by which two grids' edges may differ and still count as aligned.
# Replica boxes are computed independently, so origins differ by sub-voxel amounts that
# an absolute tolerance would reject, forcing a resample that is not needed.
_ALIGNMENT_VOXEL_FRACTION = 0.1


def _grids_spatially_aligned(g1: Grid, g2: Grid, atol: float = None,
                             voxel_fraction: float = _ALIGNMENT_VOXEL_FRACTION) -> bool:
    """True if two grids share shape, origin, and spacing.

    Equal shape alone is not enough: identical dimensions with different origin or
    spacing put voxel [i, j, k] at a different physical location. Comparing the
    per-axis edges covers both in one check.

    The comparison is sub-voxel rather than exact: when *atol* is None it defaults to
    ``voxel_fraction`` x the smaller spacing, so slightly offset grids are averaged
    voxel-wise instead of resampled. The registration error is bounded by that fraction
    of a voxel, and resampling is the more damaging option (see
    :func:`combine_dx_maps_with_resampling`).
    """
    if g1.grid.shape != g2.grid.shape:
        return False
    if atol is None:
        spacing = min(float(np.min(g1.delta)), float(np.min(g2.delta)))
        atol = voxel_fraction * spacing
    return all(
        len(e1) == len(e2) and np.allclose(e1, e2, atol=atol)
        for e1, e2 in zip(g1.edges, g2.edges)
    )

def combine_dx_maps(filepaths: List[str] = None, method: str = 'mean', out_fname: str = 'combined.dx') -> Grid:
    """Combines multiple .dx map files into one using a specified method."""

    grids = [_read_dx(path) for path in filepaths]

    # Validate all grids are spatially aligned (same shape AND origin AND
    # spacing). Shape equality alone would silently average misaligned maps.
    ref = grids[0]
    for g in grids:
        if g.grid.shape != ref.grid.shape:
            raise ValueError("All input maps must have the same shape.")
        if not _grids_spatially_aligned(ref, g):
            raise ValueError(
                "Input maps share shape but differ in origin/spacing — they are "
                "not spatially aligned. Use combine_dx_maps_with_resampling() instead."
            )

    stacked = np.stack([g.grid for g in grids])

    agg_fn = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'sum': np.sum,
        'median': np.median
    }.get(method)

    if agg_fn is None:
        raise ValueError(f"Unsupported combination method: {method}")

    combined_data = agg_fn(stacked, axis=0)
    combined_grid = Grid(combined_data, grids[0].edges)

    combined_grid.export(out_fname)

    return combined_grid

def _axis_midpoints(edges) -> List[np.ndarray]:
    """Voxel-centre coordinate per axis from a grid's edge arrays."""
    return [0.5 * (np.asarray(e[:-1]) + np.asarray(e[1:])) for e in edges]


def _resample_grid(g: Grid, ref_grid: Grid, fill_value: float = 0.0,
                   order: int = 1) -> np.ndarray:
    """Resample *g* onto *ref_grid*'s edges with an explicit out-of-range fill.

    Deliberately does NOT use :meth:`gridData.Grid.resample`, whose two defaults are
    both wrong for a free-energy map:

    * ``Grid.interpolation_cval`` defaults to ``grid.min()``, which for an AGFE map is
      the *most favourable* value, so every out-of-range voxel comes back at maximal
      attraction and the box faces fill with spurious density. ``fill_value=0.0`` is
      the correct neutral (bulk) AGFE.
    * The interpolation is a cubic spline, which overshoots and invents favourable
      voxels at well edges. Linear cannot overshoot, and ``scipy.ndimage.spline_filter``
      refuses order < 2, so the linear path goes through ``map_coordinates`` directly.

    Assumes each axis is uniformly spaced, which is true for .dx grids.
    """
    from scipy.ndimage import map_coordinates

    src_mid = _axis_midpoints(g.edges)
    tgt_mid = _axis_midpoints(ref_grid.edges)

    # Fractional source index of every target voxel centre, per axis.
    frac = []
    for s, t in zip(src_mid, tgt_mid):
        spacing = s[1] - s[0] if len(s) > 1 else 1.0
        frac.append((t - s[0]) / spacing)
    coords = np.stack(np.meshgrid(*frac, indexing="ij"))

    return map_coordinates(g.grid, coords, order=order,
                           mode="constant", cval=fill_value)


def _lattice_commensurate(grids: List[Grid], voxel_fraction: float = 1e-3) -> bool:
    """True if all *grids* share a voxel size and differ only by integer index shifts.

    Commensurate grids can be combined by array indexing alone. This is the normal case
    since :meth:`GridAnalysis._conclude` snaps every grid onto the global ``k * gridsize``
    lattice; it is checked rather than assumed so that maps written by older versions
    (whose origin depended on the probe's extreme excursion) still fall back to resampling.
    """
    if not grids:
        return False
    ref = grids[0]
    delta = np.asarray(ref.delta, dtype=float)
    if not np.all(delta > 0):
        return False
    tol = voxel_fraction * float(np.min(delta))
    for g in grids:
        if not np.allclose(np.asarray(g.delta, dtype=float), delta, atol=tol):
            return False
        off = (np.asarray(g.origin, dtype=float)
               - np.asarray(ref.origin, dtype=float)) / delta
        if not np.allclose(off, np.round(off), atol=voxel_fraction):
            return False
    return True


def _place_on_reference(g: Grid, ref_grid: Grid) -> np.ndarray:
    """Copy *g* into *ref_grid*'s index window by integer offset; uncovered voxels are NaN.

    Exact: no interpolation, so voxel values are carried over bit-for-bit. NaN marks "this
    replica's grid did not reach here" so that aggregation can skip it instead of averaging
    a real value against a fill constant.
    """
    delta = np.asarray(ref_grid.delta, dtype=float)
    off = np.round((np.asarray(g.origin, dtype=float)
                    - np.asarray(ref_grid.origin, dtype=float)) / delta).astype(int)
    out = np.full(ref_grid.grid.shape, np.nan, dtype=float)
    src, dst = [], []
    for d in range(3):
        n_src, n_ref = g.grid.shape[d], ref_grid.grid.shape[d]
        lo, hi = max(0, off[d]), min(n_ref, off[d] + n_src)
        if hi <= lo:
            return out  # grids do not overlap on this axis
        dst.append(slice(lo, hi))
        src.append(slice(lo - off[d], hi - off[d]))
    out[tuple(dst)] = g.grid[tuple(src)]
    return out


def combine_dx_maps_with_resampling(
    filepaths: List[str],
    method: str = 'mean',
    resample_to: str = 'first',
    out_fname: str = 'combined.dx',
    fill_value: float = 0.0,
) -> Grid:
    """Combine .dx maps from simulations that may have different box sizes.

    When box sizes differ grids are first resampled onto a common set of
    edges before aggregation. When all shapes already match, the fast path
    is taken (no resampling overhead).

    :param filepaths: Paths to .dx files, one per simulation replica/probe.
    :type filepaths: list[str]
    :param method: Aggregation method: 'mean' | 'max' | 'min' | 'sum' | 'median'.
    :type method: str
    :param resample_to: Which grid's edges to use as the spatial reference:
        'first' uses filepaths[0] (fastest);
        'largest' uses the grid with the most voxels (widest coverage);
        'smallest' uses the grid with the fewest voxels (most conservative).
    :type resample_to: str
    :param out_fname: Output path for the combined .dx file.
    :type out_fname: str
    :param fill_value: Value used for target voxels that fall outside a source grid
        during resampling. Defaults to 0.0, the neutral/bulk AGFE. Do NOT leave this to
        gridData's default (the grid minimum) for energy-like maps — see
        :func:`_resample_grid`.
    :type fill_value: float
    :return: Combined grid exported to out_fname.
    :rtype: gridData.Grid
    """
    grids = [_read_dx(p) for p in filepaths]

    if resample_to == 'first':
        ref_grid = grids[0]
    elif resample_to == 'largest':
        ref_grid = max(grids, key=lambda g: g.grid.size)
    elif resample_to == 'smallest':
        ref_grid = min(grids, key=lambda g: g.grid.size)
    else:
        raise ValueError(
            f"Unknown resample_to value: {resample_to!r}. "
            "Valid values: 'first', 'largest', 'smallest'."
        )

    # Gate the fast path on full spatial alignment
    if all(_grids_spatially_aligned(g, ref_grid) for g in grids):
        resampled = [g.grid for g in grids]
        exact = True
    elif _lattice_commensurate(grids):
        # Same lattice, different extents: crop/pad by integer offset. Exact, and the only
        # path that lets maps from independently solvated systems be pooled without loss.
        resampled = [_place_on_reference(g, ref_grid) for g in grids]
        exact = True
    else:
        resampled = []
        for g in grids:
            if _grids_spatially_aligned(g, ref_grid):
                resampled.append(g.grid)
            else:
                resampled.append(_resample_grid(g, ref_grid, fill_value=fill_value))
        exact = False

    if exact:
        agg_fn = {
            'mean': np.nanmean, 'max': np.nanmax, 'min': np.nanmin,
            'sum': np.nansum, 'median': np.nanmedian,
        }.get(method)
    else:
        agg_fn = {
            'mean': np.mean, 'max': np.max, 'min': np.min,
            'sum': np.sum, 'median': np.median,
        }.get(method)

    if agg_fn is None:
        raise ValueError(f"Unsupported combination method: {method!r}")

    stacked = np.stack(resampled)
    if exact:
        # Aggregate only over grids that actually reach each voxel, so a voxel covered by
        # one replica is not pulled toward `fill_value` by the others. Voxels no grid
        # reaches fall back to fill_value (neutral bulk AGFE).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            combined_data = agg_fn(stacked, axis=0)
        combined_data = np.where(np.isfinite(combined_data), combined_data, fill_value)
    else:
        combined_data = agg_fn(stacked, axis=0)
    combined_grid = Grid(combined_data, ref_grid.edges)
    combined_grid.export(out_fname)
    return combined_grid


def combine_accessible_masks(filepaths: List[str], out_fname: str = None,
                             min_fraction: float = 0.5) -> np.ndarray:
    """Combine per-replica solvent-accessible masks by MAJORITY VOTE.

    A voxel counts as accessible when more than *min_fraction* of the replicas that cover it saw
    solvent there. Union was the obvious alternative and is wrong for this purpose: the accessible
    volume would grow with the number of replicas, so ``accessible_fraction`` would not be
    comparable between runs that merged different numbers of them. Majority is count-stable (1 of 3
    and 3 of 9 agree) and discards single-visit noise, which is consistent with the rest of the
    pipeline treating one visit as noise rather than signal.

    Voxels are compared on the shared ``k * gridsize`` lattice by integer index shift, so no
    interpolation is involved; a replica whose grid does not reach a voxel simply does not vote on
    it. Returns the boolean mask; also writes it as a float .dx when *out_fname* is given.
    """
    if not filepaths:
        raise ValueError("no accessible-mask files supplied")
    grids = [_read_dx(p) for p in filepaths]
    # Reference to the widest extent so no covered voxel is dropped.
    ref = max(grids, key=lambda g: g.grid.size)
    if _lattice_commensurate(grids):
        placed = [_place_on_reference(g, ref) for g in grids]
    else:
        warnings.warn(
            "accessible masks are not on a shared lattice (maps written before the lattice fix); "
            "falling back to nearest-neighbour resampling, which shifts mask edges by up to half "
            "a voxel.", RuntimeWarning)
        placed = [_resample_grid(g, ref, fill_value=np.nan, order=0) for g in grids]

    stack = np.stack(placed)
    seen = np.asarray(stack > 0.5, dtype=float)
    covered = np.isfinite(stack)
    n_cov = covered.sum(axis=0)
    votes = np.where(covered, seen, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_cov > 0, votes / np.maximum(n_cov, 1), 0.0)
    mask = frac >= float(min_fraction)

    if out_fname:
        Grid(mask.astype(float), edges=ref.edges).export(out_fname)
    return mask


def _grid_free_energy(hist, n_atoms, n_frames, n_accessible_voxels, temperature=300):
    """
    Compute the atomic grid free energy (GFE) from a given histogram.

    :param hist: Histogram of cosolvent occupancy in each voxel
    :param n_atoms: Total number of cosolvent atoms (not total system atoms). Also this is per atom-type
    :param n_frames: Number of frames in the trajectory
    :param n_accessible_voxels: Number of solvent accessible voxels in the grid
    :param temperature: Temperature in Kelvin (default 300K)
    :return: 3D numpy array of free energy values (same shape as `hist`)
    """
    # Apply occupancy filtering: remove low-occupancy grid points
    # occupancy = hist / n_frames
    # occupancy_threshold = 0.001
    # hist[occupancy < occupancy_threshold] = 0
    # hist[hist < 2] = 0

    N_o = n_atoms / n_accessible_voxels  # Bulk probability of cosolvent
    N = hist / n_frames  # Local probability in the grid

    #if hist contains very low values (or zeros), N = hist / n_frames can be much smaller than N_o
    # making log(N / N_o) too negative and gfe extremely large.
    N = np.maximum(N, 1E-10)

    gfe = -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)

    return gfe

def _detection_floor_counts(sigma_voxels, ndim: int = 3) -> float:
    """Smallest pooled count a Gaussian kernel of width *sigma_voxels* can represent.

    A single observation smoothed by the kernel peaks at ``1/((2*pi)^(d/2) * sigma^d)``.
    Without smoothing the limit is one count.
    """
    if sigma_voxels <= 0:
        return 1.0
    return float(1.0 / ((2.0 * np.pi) ** (ndim / 2.0) * float(sigma_voxels) ** ndim))


def _grid_free_energy_density_smoothed(hist, n_atoms, n_frames, n_accessible_voxels,
                                       sigma_voxels, temperature=300.0):
    """AGFE from a Gaussian-smoothed OCCUPANCY histogram.

    Smoothing the density and then inverting avoids the ``log(0)`` floor: the only bound is
    the kernel's resolution limit (:func:`_detection_floor_counts`), applied after smoothing.
    Note an isolated single visit reads as slightly UNFAVOURABLE, since one count spread over
    the kernel is below bulk density; -1 kT needs ~5 coincident visits.
    """
    hist = np.asarray(hist, dtype=float)
    counts = (gaussian_filter(hist, sigma=sigma_voxels, mode="constant", cval=0.0)
              if sigma_voxels and sigma_voxels > 0 else hist)
    N = np.maximum(counts, _detection_floor_counts(sigma_voxels)) / n_frames
    N_o = n_atoms / n_accessible_voxels
    return -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)


def _smooth_grid_free_energy(gfe,
                             energy_cutoff: float = 0,
                             sigma: float = 1,
                            ):
    """
    Smooths and filters the grid free energy (GFE) map.

    Applies Gaussian smoothing (preserving kcal/mol units) then zeros all voxels with
    energy >= energy_cutoff. The zeroing is a display filter, not a physical operation;
    no renormalization is applied, so values stay comparable across probes and replicas.

    :param gfe: 3D numpy array of grid free energy values (kcal/mol).
    :param energy_cutoff: Cutoff energy (default: 0 kcal/mol). Voxels with
        energy >= cutoff are set to 0 (display filter, not a physical operation).
    :param sigma: Standard deviation for Gaussian smoothing (default: 1).
    :return: Smoothed and filtered grid free energy map (new array, kcal/mol).
    """

    gfe_smoothed = gaussian_filter(gfe, sigma=sigma)

    # Zero non-favorable voxels (display filter only — does not affect kcal/mol scale).
    gfe_smoothed[gfe_smoothed >= energy_cutoff] = 0.0

    return gfe_smoothed

def _grid_density(hist):
    return (hist - np.mean(hist)) / np.std(hist)

def _export(fname, grid):
    """Write *grid* to an OpenDX .dx file, unchanged.

    The voxel size, origin and shape are properties of *grid*; there is nothing to configure
    here. This used to take ``gridsize``/``center``/``box_size`` and, when given a centre and a
    box, resample onto a sub-grid via ``_subset_grid``. That path was unreachable -- the only
    live caller (``report.py``) passes a filename alone -- so ``gridsize`` was a silent no-op
    that several callers passed in the belief that it set the voxel size. Removed rather than
    documented, since a resample-on-write is not something this function should offer.
    """
    grid.export(fname)


class GridAnalysis(AnalysisBase):
    """GridAnalysis class to generate density grids

    :param AnalysisBase: Base MDAnalysis class
    :type AnalysisBase: AnalysisBase
    """
    def __init__(self, atomgroup,
                        gridsize: float = 0.5,
                        use_atomtypes: bool = True,
                        atomtypes_definitions: dict = None,
                        out_dir: str = None,
                        **kwargs):
        """*out_dir* is where the solvent-accessible mask is written.

        It used to be read as ``getattr(self, "_out_dir", None) or os.getcwd()`` with nothing in
        the package ever setting it, so the mask landed in the current working directory under one
        shared filename and every probe of every replica overwrote the previous one -- while
        ``HotspotDetector`` looked for it in its own output directory and never found one. That
        silently disabled ``accessible_fraction``, which carries a non-zero default weight.
        Pass the directory the maps go to.
        """
        super(GridAnalysis, self).__init__(atomgroup.universe.trajectory, **kwargs)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        self._u = atomgroup.universe
        self._ag = atomgroup
        self._out_dir = out_dir
        self._gridsize = gridsize
        self._nframes = 0
        self._n_atoms = atomgroup.n_atoms
        self._center = None
        self._box_size = None
        self.use_atomtypes = use_atomtypes
        self.atomtypes_definitions = atomtypes_definitions

        if use_atomtypes and atomtypes_definitions is None:
            self.logger.error("Error: Atom types definitions are required for atom type density analysis.")
            sys.exit(1)

    def _prepare(self):
        self._positions = []
        self._centers = []
        self._dimensions = []

    def _single_frame(self):
        self._positions.append(self._ag.atoms.positions.astype(float))
        self._dimensions.append(self._u.dimensions[:3])
        self._centers.append(self._u.atoms.center_of_geometry())
        self._nframes += 1

    def _conclude(self):

        self._positions = np.array(self._positions, dtype=float)
        self._box_size = np.mean(self._dimensions, axis=0)
        self._center = np.mean(self._centers, axis=0)

        # Get grid edges and origin.
        #
        # The voxel is exactly the requested gridsize. It used to be box_size / round(box_size /
        # gridsize), which made the voxel depend on the replica's mean NPT box (on FosAKP
        # 0.80142303 for a nominal 0.8, from a mean box of 73.73092).
        #
        # That produced a genuine if small inconsistency: `self._edges` were spaced by that value
        # and `np.histogramdd` binned at it, but the resulting Grid was then labelled
        # `Grid(hist, origin=origin, delta=self._gridsize)` -- i.e. declared to be 0.8-spaced. The
        # exported map therefore claimed a 0.8 voxel while its contents were binned at 0.80142, a
        # 0.18% scale error that accumulates away from the origin: ~0.22 A by the far edge of a
        # 153-voxel axis. (This is why every stored map in analysis_v3/analysis_250ns reads back at
        # exactly delta 0.8 despite the box-dependent binning.)
        #
        # Making delta exactly gridsize removes the discrepancy at the source: the same value now
        # bins the positions and labels the grid, and it is a precondition for a shared lattice.
        sd = self._box_size / 2.
        self._delta = np.full(3, float(self._gridsize), dtype=float)

        # An aligned trajectory rotates coordinates but not the box vectors, so solvent occupies
        # a rotated box while this grid is axis-aligned and np.histogramdd silently discards the
        # positions falling outside. The grid must therefore span the union of the periodic box
        # and the observed extent.
        flat = self._positions.reshape(-1, 3)
        box_lo, box_hi = self._center - sd, self._center + sd
        self._frac_outside_box = float(
            np.mean(np.any((flat < box_lo) | (flat > box_hi), axis=1))
        ) if flat.size else 0.0

        need_lo, need_hi = box_lo.copy(), box_hi.copy()
        if flat.size:
            need_lo = np.minimum(need_lo, flat.min(axis=0) - self._delta)
            need_hi = np.maximum(need_hi, flat.max(axis=0) + self._delta)

        # Snap the edges outward onto the global lattice k * gridsize. The extent stays
        # demand-driven, but the *registration* no longer is: any two grids of the same protein
        # are related by an integer index shift, so their maps combine by array arithmetic with
        # no interpolation. Previously the origin was pinned to box_lo - pad_lo * delta, where
        # pad_lo came from flat.min(axis=0) -- the single most extreme excursion of any probe atom
        # over the trajectory. That extreme-value statistic moved the origin by 8-10 A across
        # replicas of one system (protein CA alignment spread, for comparison: 0.01 A), varied the
        # SHAPE by 19-27 voxels, and grew with trajectory length. Because shapes differed, every
        # merge was forced down the resampling path (18/18 probes, both generations).
        #
        # Scope of the damage, measured on the stored maps rather than assumed: pad_lo is an
        # INTEGER voxel count, so the origins differ by whole voxels plus only the drift in box_lo
        # (0.04-0.09 A). The *fractional* misregistration was mean 0.023 voxel = 0.019 A, 95th
        # 0.054, max 0.236, with 99-100% of pairs inside the 0.1-voxel tolerance. Trilinear
        # interpolation at those offsets is nearly a no-op, so this bug did NOT materially corrupt
        # existing results. What it did cost is real but narrower: grids of differing extent were
        # combined with fill_value=0.0 under a plain mean, so voxels only some replicas reached
        # were diluted toward bulk (see the coverage-aware branch in
        # `combine_dx_maps_with_resampling`), plus the fragility of never being able to stack maps.
        lo = np.floor(need_lo / self._delta) * self._delta
        hi = np.ceil(need_hi / self._delta) * self._delta
        nbins = np.maximum(1, np.round((hi - lo) / self._delta).astype(int))
        self._edges = tuple(lo[d] + np.arange(nbins[d] + 1) * self._delta[d]
                            for d in range(3))

        if self._frac_outside_box > 0:
            box_bins = tuple(np.round(self._box_size / self._delta).astype(int))
            self.logger.warning(
                f"{100 * self._frac_outside_box:.2f}% of positions fell outside the "
                f"box-sized grid (aligned trajectory in an axis-aligned grid); grid spans "
                f"{box_bins} -> {tuple(nbins)} voxels so none are discarded."
            )
        origin = (self._edges[0][0], self._edges[1][0], self._edges[2][0])

        # get the mask of accesible voxels that will be used for the free energy calculation
        self._build_accessible_mask()

        # Get positions and atom types
        positions = self._get_positions()

        if self.use_atomtypes: # turn on for atomtype density
            self._type_histograms = {}  # Create per-type histograms

            # Map atom types to atoms in the system
            mapped_atomtypes = self._map_atomtypes(self.atomtypes_definitions)

            # Fall back to standard density if SMARTS matching failed completely
            if mapped_atomtypes is None:
                hist, _ = np.histogramdd(positions, bins=self._edges)
                self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
                self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)
                return

            # Get atom types for all frames as a single array
            atom_types_array = np.tile(mapped_atomtypes, self._nframes)

            for atom_type in self.atomtypes_dict.keys():

                self.logger.info(f"Processing atom type: {atom_type}")

                # Select positions for this atom type
                mask = np.char.startswith(atom_types_array.astype(str), atom_type)

                type_positions = positions[mask]

                # Skip empty positions for a type
                if len(type_positions) == 0:
                    self.logger.warning(f"Skipping atom type {atom_type} as it has no positions.")
                    continue

                # Generate histogram for this type
                hist, _ = np.histogramdd(type_positions, bins=self._edges)
                self._type_histograms[atom_type] = Grid(hist, origin=origin, delta=self._gridsize)

            # Create a combined density grid by summing all atom types
            if not self._type_histograms:
                self.logger.warning(
                    "No atom type histograms were produced (all types had no matching positions). "
                    "Falling back to standard density estimation."
                )
                hist, _ = np.histogramdd(positions, bins=self._edges)
                self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
                self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)
                return
            total_hist = sum(grid.grid for grid in self._type_histograms.values())
            self._histogram = Grid(total_hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(total_hist), origin=origin, delta=self._gridsize)
        else:
            hist, _ = np.histogramdd(positions, bins=self._edges)
            self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop, :, :]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)

        return positions

    def _build_accessible_mask(self, traj_step=5, probe_radius=1.4, export=True):
        """
        Build a boolean grid where True = voxel is solvent-accessible.
        Uses the union of water-oxygen and cosolvent heavy-atom positions so that
        hydrophobic/cryptic regions sampled by the probe but not by water are included
        in the reference volume used to compute N_o.
        The grid is dilated by `probe_radius` to account for the size of the probe.

        Parameters
        ----------
        traj_step   : int   use every `traj_step`-th frame to save time
        probe_radius: float Å, radius you want to allow beyond sampled O positions
        export      : bool  if True, export the grid to a .dx file
        """
        if hasattr(self, "_n_accessible_voxels"):
            return  # already built

        # collect water-oxygen + cosolvent heavy atoms to capture hydrophobic/cryptic
        # regions that water undersamples, giving an accurate reference volume for N_o
        O_sel       = self._u.select_atoms("resname HOH WAT and name O")
        probe_heavy = self._ag.select_atoms("not name H*")
        # protein heavy atoms define the buried/excluded volume
        protein_sel = self._u.select_atoms(
            "protein and not name H* and not (resname HOH WAT)"
        )

        coords = []
        protein_coords = []
        for ts in self._u.trajectory[::traj_step]: # this stride saves time
            coords.append(O_sel.positions.copy())
            coords.append(probe_heavy.positions.copy())
            if protein_sel.n_atoms > 0:
                protein_coords.append(protein_sel.positions.copy())
        coords = np.vstack(coords)

        # histogram into current grid
        hist, _ = np.histogramdd(coords, bins=self._edges)
        mask = hist > 0

        # dilate by ≈ probe_radius
        n_iter = int(round(probe_radius / self._gridsize))
        mask = binary_dilation(mask, iterations=max(1, n_iter))

        # exclude voxels occupied by protein atoms (buried volume)
        if protein_coords:
            protein_coords_arr = np.vstack(protein_coords)
            protein_hist, _ = np.histogramdd(protein_coords_arr, bins=self._edges)
            protein_mask = protein_hist > 0
            mask = mask & ~protein_mask

        # Count the reference volume only inside the box-sized central region, so N_o does not
        # depend on the padding.
        delta = getattr(self, "_delta", np.full(3, self._gridsize, dtype=float))
        box_lo = np.asarray(self._center) - np.asarray(self._box_size) / 2.0
        box_hi = np.asarray(self._center) + np.asarray(self._box_size) / 2.0
        in_box = np.ones(mask.shape, dtype=bool)
        for d in range(3):
            mids = (self._edges[d][:-1] + self._edges[d][1:]) / 2.0
            keep = (mids >= box_lo[d] - 1e-9) & (mids <= box_hi[d] + 1e-9)
            shape = [1, 1, 1]
            shape[d] = -1
            in_box &= keep.reshape(shape)
        n_acc = int((mask & in_box).sum())
        box_voxels = int(round(float(np.prod(np.asarray(self._box_size) / np.asarray(delta)))))
        n_protein = int(protein_mask.sum()) if protein_coords else 0
        cap = max(1, box_voxels - n_protein)
        if n_acc > cap:
            self.logger.warning(
                f"Accessible voxels ({n_acc:,}) exceed the box solvent volume "
                f"({cap:,} = box {box_voxels:,} - protein {n_protein:,}); capping. "
                "The reference volume cannot exceed the periodic box."
            )
            n_acc = cap
        self._n_accessible_voxels = n_acc
        grid_vol = self._gridsize ** 3
        self.logger.info(f"Number of accessible voxels: {self._n_accessible_voxels:.2f}")
        self.logger.info(f"Volume of accessible voxels: {self._n_accessible_voxels/1000 * grid_vol:.2f} nm³")

        if export:
            mask_grid = mask.astype(float)
            grid = Grid(mask_grid, edges=self._edges)
            # Write beside the other maps, not into whatever the current working directory happens
            # to be, and tag the filename with the probe. The mask is the union of water oxygens
            # and THIS probe's heavy atoms, so it is probe-specific; a single shared filename in
            # the cwd meant every probe of every replica overwrote the previous one and the
            # detector found nothing in its own output directory.
            out_dir = self._out_dir or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            grid.export(os.path.join(out_dir, f"solvent_accessible_map{self._probe_tag()}.dx"))

        return

    def _probe_tag(self):
        """``_BEN`` for a single-residue probe selection, ``""`` if it cannot be determined."""
        try:
            names = sorted({str(r) for r in self._ag.residues.resnames})
        except AttributeError:
            return ""
        return "_" + "_".join(names) if names else ""

    def _map_atomtypes(self, atomtypes_definitions: list = None) -> np.ndarray:
        """Maps atom types to their respective categories based on SMARTS patterns.
        Some useful definitions here:  https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        :param atomtypes_definitions: A list of atom types definitions based on SMARTS patterns.
        :type atomtypes_definitions: list
        :return: Array of mapped atom types.
        :rtype: np.ndarray
        """

        # select atoms based on SMARTS patterns
        self.atomtypes_dict = {atomtype['atype']: self._ag.select_atoms(f"smarts {atomtype['smarts']}") for atomtype in atomtypes_definitions}
        # Count the number of atoms by type, this is required for the free energy calculation
        self._n_atoms_by_type = {key: ag.n_atoms for key, ag in self.atomtypes_dict.items()}
        self.logger.debug(f"Atom types count: {self._n_atoms_by_type}")

        # Warn if any SMARTS pattern matched no atoms in the cosolvent molecule
        for key, n in self._n_atoms_by_type.items():
            if n == 0:
                self.logger.warning(
                    f"SMARTS pattern for atom type '{key}' matched no atoms in the cosolvent. "
                    f"Check that the SMARTS is appropriate for this molecule."
                )

        # Map each atom to its category using atom indices from SMARTS matches directly.
        # Using FF types as an intermediary (e.g. np.unique(ag.atoms.types)) fails when
        # multiple categories share the same FF type (e.g. C3 appears in both HBA and Car).
        ag_indices = self._ag.atoms.indices
        mapped_atomtypes = np.zeros_like(self._ag.atoms.types, dtype=object)

        for key, matched_ag in self.atomtypes_dict.items():
            if matched_ag.n_atoms == 0:
                continue
            match_mask = np.isin(ag_indices, matched_ag.atoms.indices)
            # First-match wins: don't overwrite atoms already claimed by a prior category
            unassigned_mask = mapped_atomtypes == 0
            mapped_atomtypes[match_mask & unassigned_mask] = key

        # Rebuild atomtypes_dict so callers can still iterate over its keys
        self.atomtypes_dict = {key: ag for key, ag in self.atomtypes_dict.items()}

        # Warn about atoms that could not be assigned to any SMARTS-defined type
        unmatched_mask = mapped_atomtypes == 0
        n_unmatched = int(unmatched_mask.sum())
        if n_unmatched > 0:
            unmatched_ff_types = np.unique(self._ag.atoms.types[unmatched_mask])
            self.logger.warning(
                f"{n_unmatched} atom(s) with force-field types {list(unmatched_ff_types)} "
                f"did not match any SMARTS pattern and will be excluded from the density maps."
            )

        # If no atom was assigned at all, signal the caller to fall back to standard density
        if n_unmatched == len(mapped_atomtypes):
            self.logger.warning(
                "No atoms matched any SMARTS pattern. Falling back to standard (non-atomtype) density estimation."
            )
            return None

        return mapped_atomtypes

    def _agfe_from_hist(self, hist, n_atoms, sigma_vox, smoothing, smoothing_space,
                        temperature):
        """(raw, display) AGFE arrays for one occupancy histogram.

        ``raw`` is the full unclipped field; ``display`` has unfavourable voxels zeroed.
        """
        if smoothing and smoothing_space == "density":
            raw = _grid_free_energy_density_smoothed(
                hist, n_atoms, self._nframes, self._n_accessible_voxels,
                sigma_vox, temperature)
            display = raw.copy()
            display[display >= 0] = 0.0
            return raw, display

        raw = _grid_free_energy(hist, n_atoms, self._nframes,
                                self._n_accessible_voxels, temperature)
        if not smoothing:
            return raw, raw
        return raw, _smooth_grid_free_energy(raw, sigma=sigma_vox, energy_cutoff=0)

    def atomic_grid_free_energy(self, temperature=300., atom_radius=1.4, smoothing=True,
                                smoothing_space="density"):
        """Compute grid free energy by boltzmann inversion of the occupancy histogram at a given temperature.
        Optionally, the free energy map can be smoothed using a Gaussian filter and some tricks.

        :param temperature: Temperature in Kelvin (default 300K)
        :param atom_radius: Atomic radius for smoothing (default 1.4A)
        :param smoothing: Apply smoothing to the free energy map (default True)
        :param smoothing_space: ``"density"`` (default) smooths the occupancy histogram then
            inverts it. ``"energy"`` is the legacy path: invert with ``N`` floored at 1e-10,
            then smooth the energy field. Kept only to reproduce earlier results.

        """
        if smoothing_space not in ("density", "energy"):
            raise ValueError("smoothing_space must be 'density' or 'energy', "
                             f"got {smoothing_space!r}")

        # gaussian_filter interprets sigma in VOXELS
        sigma_vox = (atom_radius / 3.0) / self._gridsize

        if self.use_atomtypes:
            self._type_agfe_raw = {}
            for atom_type, grid in self._type_histograms.items():
                n_atoms_type = self._n_atoms_by_type[atom_type]
                raw, agfe = self._agfe_from_hist(grid.grid, n_atoms_type, sigma_vox,
                                                 smoothing, smoothing_space, temperature)
                self._type_agfe_raw[atom_type] = Grid(raw, edges=grid.edges)

                self.logger.info(f"Free energy for {atom_type}: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")
                self._type_histograms[atom_type] = Grid(agfe, edges=grid.edges)
        else:
            raw, agfe = self._agfe_from_hist(self._histogram.grid, self._n_atoms, sigma_vox,
                                             smoothing, smoothing_space, temperature)
            self._agfe_raw = Grid(raw, edges=self._histogram.edges)
            if smoothing:
                self.logger.info(f"Free energy: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")

            self._agfe = Grid(agfe, edges=self._histogram.edges)

        return

    def export_histogram(self, fname):
        """ Export histogram maps
        """
        _export(fname, self._histogram)

    def export_density(self, fname):
        """ Export density maps, either for the total density or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                density_fname = fname.replace('map_rawdensity', f'map_density_{atom_type}')
                _export(density_fname, grid)
        else:
            _export(fname, self._density)

    def export_atomic_grid_free_energy(self, fname):
        """ Export atomic grid free energy, either for the total free energy or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                gfe_fname = fname.replace('map_agfe', f'map_agfe_{atom_type}')
                _export(gfe_fname, grid)
        else:
            _export(fname, self._agfe)

    def export_raw_atomic_grid_free_energy(self, fname):
        """Export the raw (unsmoothed) AGFE map in kcal/mol.

        Direct Boltzmann inversion of the occupancy histogram, with no zeroing or
        rescaling, so values stay comparable across probes, systems, and replicas.
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_agfe_raw.items():
                gfe_fname = fname.replace('map_agfe_raw', f'map_agfe_raw_{atom_type}')
                _export(gfe_fname, grid)
        else:
            _export(fname, self._agfe_raw)


from scipy.ndimage import map_coordinates as _map_coordinates


def grids_aligned(o1, d1, s1, o2, d2, s2, atol=1e-3):
    """True if two grids share origin, spacing, and shape (within atol)."""
    return (tuple(s1) == tuple(s2)
            and np.allclose(np.asarray(o1), np.asarray(o2), atol=atol)
            and np.allclose(np.asarray(d1), np.asarray(d2), atol=atol))


def resample_mask_to_grid(mask, src_origin, src_delta, ref_origin, ref_delta, ref_shape):
    """Nearest-neighbour resample a boolean voxel mask onto a reference grid.

    Voxels of the reference grid that fall outside the source volume are False.
    Returns a boolean ndarray of shape ``ref_shape``. Fast path (identity) when
    the source and reference grids coincide.
    """
    src_origin = np.asarray(src_origin, dtype=float)
    src_delta = np.asarray(src_delta, dtype=float)
    ref_origin = np.asarray(ref_origin, dtype=float)
    ref_delta = np.asarray(ref_delta, dtype=float)
    ref_shape = tuple(int(s) for s in ref_shape)

    if grids_aligned(src_origin, src_delta, mask.shape, ref_origin, ref_delta, ref_shape):
        return mask.astype(bool)

    # Reference voxel indices -> Angstrom -> source fractional indices.
    gi, gj, gk = np.meshgrid(
        np.arange(ref_shape[0]), np.arange(ref_shape[1]), np.arange(ref_shape[2]),
        indexing="ij",
    )
    pos = (np.stack([gi, gj, gk], axis=0).astype(float)
           * ref_delta[:, None, None, None] + ref_origin[:, None, None, None])
    frac = (pos - src_origin[:, None, None, None]) / src_delta[:, None, None, None]
    coords = frac.reshape(3, -1)
    sampled = _map_coordinates(mask.astype(np.float32), coords,
                               order=0, mode="constant", cval=0.0)
    return (sampled.reshape(ref_shape) >= 0.5)
