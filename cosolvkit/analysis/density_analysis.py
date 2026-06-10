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
from glob import glob
import numpy as np
from typing import List, Union

from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.interpolate import RegularGridInterpolator
from gridData import Grid

from MDAnalysis.analysis.base import AnalysisBase


BOLTZMANN_CONSTANT_KB = 0.0019872041  # kcal/(mol*K)

def _read_dx(filepath: str = None) -> Grid:
    """Reads a .dx map using gridData.Grid."""
    return Grid(str(filepath))

def combine_dx_maps(filepaths: List[str] = None, method: str = 'mean', out_fname: str = 'combined.dx') -> Grid:
    """Combines multiple .dx map files into one using a specified method."""

    grids = [_read_dx(path) for path in filepaths]

    # Validate all grids match in shape. This is kinda clunky, but it works.
    shape = grids[0].grid.shape
    for g in grids:
        if g.grid.shape != shape:
            raise ValueError("All input maps must have the same shape.")

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

def combine_dx_maps_with_resampling(
    filepaths: List[str],
    method: str = 'mean',
    resample_to: str = 'first',
    out_fname: str = 'combined.dx',
) -> Grid:
    """Combine .dx maps from simulations that may have different box sizes.

    When box sizes differ (common when different cosolvent probes are run in
    independent simulations), grids are first resampled onto a common set of
    edges before aggregation.  When all shapes already match, the fast path
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

    shapes_match = all(g.grid.shape == ref_grid.grid.shape for g in grids)

    if shapes_match:
        resampled = [g.grid for g in grids]
    else:
        resampled = []
        for g in grids:
            if g.grid.shape == ref_grid.grid.shape:
                resampled.append(g.grid)
            else:
                resampled.append(g.resample(ref_grid.edges).grid)

    agg_fn = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'sum': np.sum,
        'median': np.median,
    }.get(method)

    if agg_fn is None:
        raise ValueError(f"Unsupported combination method: {method!r}")

    combined_data = agg_fn(np.stack(resampled), axis=0)
    combined_grid = Grid(combined_data, ref_grid.edges)
    combined_grid.export(out_fname)
    return combined_grid


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

def _smooth_grid_free_energy(gfe,
                             energy_cutoff: float = 0,
                             sigma: float = 1,
                            ):
    """
    Smooths and filters the grid free energy (GFE) map.

    Applies Gaussian smoothing (preserving kcal/mol units) then zeros all
    voxels with energy >= energy_cutoff.  The zeroing is a display/filtering
    choice: unfavorable regions are suppressed so that only hot-spots appear
    in the output map.  No renormalization is applied so that values remain
    physically comparable across probes, systems, and replicas.

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

def _subset_grid(grid, center, box_size, gridsize=0.5):

    #FIXME I think this part of the code is never triggered, not sure if we need this

    # Create grid interpolator
    # Number of midpoints is equal to the number of grid points
    grid_interpn = RegularGridInterpolator(grid.midpoints, grid.grid)

    # Create sub grid coordinates
    # We get first the edges of the grid box, and after the midpoints
    # So this we are sure (I guess) that the sub grid is well centered on center
    # There might be a better way of doing this... Actually I tried, but didn't worked very well.
    x, y, z = center
    sd = box_size / 2.
    hbins = np.round(box_size / gridsize).astype(int)
    edges = (np.linspace(0, box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
             np.linspace(0, box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
             np.linspace(0, box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
    midpoints = (edges[0][:-1] + np.diff(edges[0]) / 2.,
                 edges[1][:-1] + np.diff(edges[1]) / 2.,
                 edges[2][:-1] + np.diff(edges[2]) / 2.)
    X, Y, Z = np.meshgrid(midpoints[0], midpoints[1], midpoints[2])
    xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    # Configuration of the sub grid
    origin_subgrid = (midpoints[0][0], midpoints[1][0], midpoints[2][0])
    shape_subgrid = (midpoints[0].shape[0], midpoints[1].shape[0], midpoints[2].shape[0])

    # Do interpolation
    sub_grid_values = grid_interpn(xyzs)
    sub_grid_values = sub_grid_values.reshape(shape_subgrid)
    sub_grid_values = np.swapaxes(sub_grid_values, 0, 1)
    sub_grid = Grid(sub_grid_values, origin=origin_subgrid, delta=gridsize)

    return sub_grid

def _export(fname, grid, gridsize=0.5, center=None, box_size=None):
    assert (center is None and box_size is None) or (center is not None and box_size is not None), \
           "Both center and box size have to be defined, or none of them."

    if center is None and box_size is None:
        grid.export(fname)
    elif center is not None and box_size is not None:
        center = np.array(center)
        box_size = np.array(box_size)

        assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
        assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
        assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

        sub_grid = _subset_grid(grid, center, box_size, gridsize)
        sub_grid.export(fname)
    return

class GridAnalysis(AnalysisBase):
    """GridAnalysis class to generate density grids

    :param AnalysisBase: Base MDAnalysis class
    :type AnalysisBase: AnalysisBase
    """
    def __init__(self, atomgroup,
                        gridsize: float = 0.5,
                        use_atomtypes: bool = True,
                        atomtypes_definitions: dict = None,
                        **kwargs):
        super(GridAnalysis, self).__init__(atomgroup.universe.trajectory, **kwargs)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        self._u = atomgroup.universe
        self._ag = atomgroup
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

        # Get grid edges and origin
        x, y, z = self._center
        sd = self._box_size / 2.
        hbins = np.round(self._box_size / self._gridsize).astype(int)
        self._edges = (np.linspace(0, self._box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
                    np.linspace(0, self._box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
                    np.linspace(0, self._box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
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
                self._build_accessible_mask()
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
                self._build_accessible_mask()
                return
            total_hist = sum(grid.grid for grid in self._type_histograms.values())
            self._histogram = Grid(total_hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(total_hist), origin=origin, delta=self._gridsize)
        else:
            hist, _ = np.histogramdd(positions, bins=self._edges)
            self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

        # Calculate the number of accessible voxels, once per trajectory
        self._build_accessible_mask()

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

        # count and save the mask
        self._n_accessible_voxels = int(mask.sum())
        grid_vol = self._gridsize ** 3
        self.logger.info(f"Number of accessible voxels: {self._n_accessible_voxels:.2f}")
        self.logger.info(f"Volume of accessible voxels: {self._n_accessible_voxels/1000 * grid_vol:.2f} nm³")

        if export:
            mask_grid = mask.astype(float)
            grid = Grid(mask_grid, edges=self._edges)
            grid.export(f"solvent_accessible_map.dx")

        return

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

    def atomic_grid_free_energy(self, temperature=300., atom_radius=1.4, smoothing=True):
        """Compute grid free energy by boltzmann inversion of the occupancy histogram at a given temperature.
        Optionally, the free energy map can be smoothed using a Gaussian filter and some tricks.

        :param temperature: Temperature in Kelvin (default 300K)
        :param atom_radius: Atomic radius for smoothing (default 1.4A)
        :param smoothing: Apply smoothing to the free energy map (default True)

        """

        if self.use_atomtypes:
            self._type_agfe_raw = {}
            for atom_type, grid in self._type_histograms.items():
                n_atoms_type = self._n_atoms_by_type[atom_type]
                agfe = _grid_free_energy(grid.grid, n_atoms_type, self._nframes, self._n_accessible_voxels, temperature)
                self._type_agfe_raw[atom_type] = Grid(agfe, edges=grid.edges)
                if smoothing:
                    agfe = _smooth_grid_free_energy(agfe, sigma=atom_radius / 3.0, energy_cutoff=0)

                self.logger.info(f"Free energy for {atom_type}: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")
                self._type_histograms[atom_type] = Grid(agfe, edges=grid.edges)
        else:
            agfe = _grid_free_energy(self._histogram.grid, self._n_atoms, self._nframes, self._n_accessible_voxels, temperature)
            self._agfe_raw = Grid(agfe, edges=self._histogram.edges)

            if smoothing:
                # We divide by 3 in order to have radius == 3 sigma
                agfe = _smooth_grid_free_energy(agfe, sigma=atom_radius / 3.0, energy_cutoff=0)
                self.logger.info(f"Free energy: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")

            self._agfe = Grid(agfe, edges=self._histogram.edges)

        return

    def export_histogram(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export histogram maps
        """
        _export(fname, self._histogram, gridsize, center, box_size)

    def export_density(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps, either for the total density or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                density_fname = fname.replace('map_rawdensity', f'map_density_{atom_type}')
                _export(density_fname, grid, gridsize, center, box_size)
        else:
            _export(fname, self._density, gridsize, center, box_size)

    def export_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export atomic grid free energy, either for the total free energy or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                gfe_fname = fname.replace('map_agfe', f'map_agfe_{atom_type}')
                _export(gfe_fname, grid, gridsize, center, box_size)
        else:
            _export(fname, self._agfe, gridsize, center, box_size)

    def export_raw_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """Export the raw (unsmoothed, physical) AGFE map in kcal/mol.

        Values are the direct Boltzmann inversion of the occupancy histogram
        with no zeroing or rescaling applied, making them suitable for
        quantitative comparisons across probes, systems, and replicas.
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_agfe_raw.items():
                gfe_fname = fname.replace('map_agfe_raw', f'map_agfe_raw_{atom_type}')
                _export(gfe_fname, grid, gridsize, center, box_size)
        else:
            _export(fname, self._agfe_raw, gridsize, center, box_size)


def generate_pymol_session(out_path: str,
                            cosolvent_names: list,
                            avg_pdb_path: str,
                            density_files: Union[str, list] = None,
                            selection_string: str = None,
                            reference_pdb: str = None,
                            compute_avg_structure: callable = None):
    """Generate a PyMol session from the density maps.

    :param out_path: directory where outputs are written.
    :type out_path: str
    :param cosolvent_names: list of cosolvent residue names.
    :type cosolvent_names: list
    :param avg_pdb_path: path to the averaged-trajectory PDB used as the reference structure.
    :type avg_pdb_path: str
    :param density_files: .dx file(s) to load.  If None the final agfe maps from
        ``out_path`` are used.  Accepts a single path, a directory, or a list.
    :type density_files: Union[str, list]
    :param selection_string: PyMol selection string for residues of interest.
    :type selection_string: str
    :param reference_pdb: additional reference PDB to load alongside the average structure.
    :type reference_pdb: str
    :param compute_avg_structure: optional callable invoked when ``avg_pdb_path`` does
        not exist and no ``reference_pdb`` is provided.  Should generate the file at
        ``avg_pdb_path`` and return it.
    :type compute_avg_structure: callable
    """
    from pymol import cmd

    logger = logging.getLogger(__name__)

    if density_files is None:
        density_files = []
        for cosolvent in cosolvent_names:
            agfe_file = os.path.join(out_path, f"map_agfe_{cosolvent}.dx")
            if os.path.isfile(agfe_file):
                density_files.append(agfe_file)
            else:
                # atomtypes mode: collect per-atom-type agfe maps, exclude raw
                per_type = sorted(
                    f for f in glob(os.path.join(out_path, f"map_agfe_*_{cosolvent}.dx"))
                    if 'raw' not in os.path.basename(f)
                )
                density_files.extend(per_type)
    elif os.path.isfile(density_files):
        density_files = [density_files]
    elif os.path.isdir(density_files):
        density_files = [os.path.join(density_files, f) for f in os.listdir(density_files) if f.endswith('.dx')]
    elif isinstance(density_files, list):
        pass
    else:
        logger.error("Please provide a list of density files to include in the PyMol session.")
        return

    colors = ['marine', 'orange', 'magenta', 'salmon', 'purple']
    assert len(density_files) <= len(colors), "Error! Too many density files, not enough colors available!"

    if avg_pdb_path is None or not os.path.exists(avg_pdb_path):
        if reference_pdb is not None and reference_pdb.endswith('.pdb'):
            structures = {os.path.basename(reference_pdb).split('.')[0]: reference_pdb}
        elif compute_avg_structure is not None:
            compute_avg_structure()
            structures = {'average_structure': avg_pdb_path}
        else:
            logger.error("avg_pdb_path does not exist and no reference_pdb or compute_avg_structure provided.")
            return
    else:
        structures = {'average_structure': avg_pdb_path}
        if reference_pdb is not None and reference_pdb.endswith('.pdb'):
            reference_pdb_name = os.path.basename(reference_pdb).split('.')[0]
            structures[reference_pdb_name] = reference_pdb

    cmd_string = ""

    for structure_name, pdb_path in structures.items():
        cmd.load(pdb_path, structure_name)
        cmd_string += f"cmd.load('{pdb_path}', '{structure_name}')\n"
        cmd.color("grey50", f"{structure_name} and name C*")
        cmd_string += f"cmd.color('grey50', '{structure_name} and name C*')\n"

    for color, density in zip(colors, density_files):
        dens_name = os.path.basename(density).split('.')[0]

        dx_data = _read_dx(density)
        # AGFE maps are capped at 0 from above (unfavorable regions zeroed out),
        # so we contour at the bottom 1% (most favorable/negative values).
        # Density maps (z-score) have positive peaks, so we use the top 1%.
        is_agfe = np.max(dx_data.grid) <= 0.0
        dx_01 = np.quantile(dx_data.grid, 0.001 if is_agfe else 0.999)

        cmd.load(density, f'{dens_name}_map')
        cmd_string += f"cmd.load('{density}', '{dens_name}_map')\n"
        cmd.isomesh(f'{dens_name}_mesh', f'{dens_name}_map', dx_01)
        cmd_string += f"cmd.isomesh('{dens_name}_mesh', '{dens_name}_map', {dx_01})\n"
        cmd.color(color, f'{dens_name}_mesh')
        cmd_string += f"cmd.color('{color}', '{dens_name}_mesh')\n"

    if selection_string:
        cmd.show("sticks", selection_string)
        cmd_string += f"cmd.show('sticks', '{selection_string}')\n"

    cmd.hide("spheres")
    cmd.set('specular', 1)
    cmd.set("cartoon_side_chain_helper", 1)
    cmd_string += "cmd.hide('spheres')\n"
    cmd_string += "cmd.set('specular', 1)\n"
    cmd_string += "cmd.set('cartoon_side_chain_helper', 1)\n"

    if selection_string:
        cmd.spectrum("b", "blue_white_red", selection_string)
        cmd_string += f"cmd.spectrum('b', 'blue_white_red', '{selection_string}')\n"

    cmd.bg_color("white")
    cmd_string += "cmd.bg_color('white')"

    with open(os.path.join(out_path, "pymol_session_cmd.pml"), "w") as fo:
        fo.write(cmd_string)

    cmd.save(os.path.join(out_path, "pymol_results_session.pse"))


def generate_vmd_session(out_path: str,
                          topology: str,
                          trajectory: str,
                          density_files: Union[str, list] = None):
    """Generate a VMD session script to visualize the trajectory and density.

    :param out_path: directory where the script is written.
    :type out_path: str
    :param topology: path to the topology file.
    :type topology: str
    :param trajectory: path to the trajectory file.
    :type trajectory: str
    :param density_files: list of .dx density files to load.
    :type density_files: Union[str, list]
    """
    logger = logging.getLogger(__name__)

    # FIXME at some point like for pymol
    isovalue = 1.0
    output_vmd_file = os.path.join(out_path, "vmd_session.vmd")

    topology_abs_path = os.path.abspath(topology)
    trajectory_abs_path = os.path.abspath(trajectory)

    vmd_script = f"""
# VMD visualization script

# Load topology and trajectory
mol new {topology_abs_path} type parm7
mol addfile {trajectory_abs_path} type netcdf waitfor all

# Set up protein visualization
mol delrep 0 top
mol representation NewCartoon
mol color Structure
mol selection "protein"
mol material Opaque
mol addrep top"""

    for i, density in enumerate(density_files or []):
        density_dx_abs_path = os.path.abspath(density)
        vmd_script += f"""

# Load density map
mol new {density_dx_abs_path} type dx waitfor all
mol representation Isosurface {isovalue} 0 0 0 1
mol color ColorID {i}
mol material Transparent
mol addrep top"""

    vmd_script += f"""

color Display Background white

save_state {output_vmd_file}
"""

    with open(output_vmd_file, "w") as f:
        f.write(vmd_script)

    logger.info(f"VMD session script saved as {output_vmd_file}")
