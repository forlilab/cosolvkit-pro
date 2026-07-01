#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# PyMol session generation
#

import os
import logging
from glob import glob
from typing import Union

import numpy as np
from gridData import Grid
from scipy.ndimage import center_of_mass

from cosolvkit.analysis.core.grid import _read_dx

try:
    from pymol import cmd as _pymol_cmd
    _PYMOL_AVAILABLE = True
except ImportError:
    _PYMOL_AVAILABLE = False

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Hotspot PyMOL builders — moved verbatim from hotspot_visualization.py
# ---------------------------------------------------------------------------


def _contour_level_from_dx(dx_path):
    """Return an isomesh contour level appropriate for a DX file.

    AGFE maps (all values ≤ 0) are contoured at the 0.1th percentile
    (most negative / most favourable).  Positive maps (z-score density)
    use the 99.9th percentile.
    """
    data = Grid(dx_path).grid
    is_agfe = np.max(data) <= 0.0
    return float(np.quantile(data, 0.001 if is_agfe else 0.999))


# RGB colours (0–1 range) paired with PyMol named colours for the .pml script
_PYMOL_CLUSTER_COLORS = [
    ((0.12, 0.47, 0.71), 'marine'),
    ((1.00, 0.50, 0.05), 'orange'),
    ((0.84, 0.15, 0.16), 'red'),
    ((0.17, 0.63, 0.17), 'forest'),
    ((0.58, 0.40, 0.74), 'purple'),
    ((0.55, 0.34, 0.29), 'chocolate'),
    ((0.89, 0.47, 0.76), 'pink'),
    ((0.74, 0.74, 0.13), 'olive'),
    ((0.09, 0.75, 0.81), 'cyan'),
    ((1.00, 0.85, 0.18), 'yellow'),
    ((0.68, 0.78, 0.91), 'lightblue'),
    ((1.00, 0.60, 0.60), 'salmon'),
    ((0.60, 0.87, 0.54), 'palegreen'),
    ((0.77, 0.69, 0.84), 'violet'),
    ((0.77, 0.61, 0.49), 'wheat'),
    ((0.97, 0.51, 0.47), 'firebrick'),
    ((0.62, 0.85, 0.90), 'teal'),
    ((1.00, 0.73, 0.47), 'gold'),
    ((0.60, 0.76, 0.98), 'slate'),
    ((0.60, 0.98, 0.80), 'aquamarine'),
]


def visualise_clustering(
    cosolvent,
    labeled_array,
    combined_grid,
    results,
    out_path,
    voxel_to_angstrom_fn,
    reference_pdb=None,
):
    """Generate a PyMol session to visually inspect clustering results.

    All clusters are encoded in a **single** label DX file (voxel value =
    cluster ID, 0 = background).  A volume object with a per-cluster colour
    ramp is used instead of one isomesh per cluster, which avoids writing N
    files and dramatically reduces I/O for large maps.

    Pseudoatom labels are placed at each site's centroid showing its rank
    and composite score.

    Parameters
    ----------
    cosolvent : str
    labeled_array : np.ndarray of int
        Cluster label grid (0 = background).
    combined_grid : gridData.Grid
        AGFE grid used for coordinate conversion.
    results : list[Hotspot]
    out_path : str
        Directory for output files.
    voxel_to_angstrom_fn : callable
        ``f(grid, vox_idx) -> np.ndarray`` — converts voxel indices to Ångströms.
    reference_pdb : str, optional
        Path to a PDB file to load as structural context.

    Returns
    -------
    str
        Path to the saved ``.pse`` session file.
    """
    if not _PYMOL_AVAILABLE:
        raise ImportError(
            "PyMol is required for visualise_clustering. "
            "Install it with: conda install -c schrodinger pymol"
        )

    site_labels = sorted(int(lbl) for lbl in np.unique(labeled_array) if lbl != 0)
    site_by_id = {s.site_id: s for s in results}

    cmd_string = ""

    if reference_pdb is not None and os.path.isfile(reference_pdb):
        struct_name = os.path.splitext(os.path.basename(reference_pdb))[0]
        _pymol_cmd.load(reference_pdb, struct_name)
        _pymol_cmd.color('grey50', f'{struct_name} and name C*')
        cmd_string += f"cmd.load('{reference_pdb}', '{struct_name}')\n"
        cmd_string += f"cmd.color('grey50', '{struct_name} and name C*')\n"

    # --- Single DX for all clusters ---
    # Prefer the rank-label map written by export_results() (voxel = rank).
    # If not present, write a site-ID label map now.
    rank_dx = os.path.join(out_path, f"hotspot_labels_{cosolvent}.dx")
    if os.path.isfile(rank_dx):
        dx_path = rank_dx
        # Ramp values are site ranks (1, 2, ...)
        label_values = [site.rank for site in sorted(results, key=lambda s: s.rank)]
    else:
        dx_path = os.path.join(out_path, f"_cluster_labels_{cosolvent}.dx")
        Grid(labeled_array.astype(float), combined_grid.edges).export(dx_path)
        # Ramp values are raw site IDs from labeled_array
        label_values = site_labels

    map_name = f'cluster_labels_{cosolvent}'
    vol_name = f'cluster_vol_{cosolvent}'
    ramp_name = f'ramp_clusters_{cosolvent}'

    _pymol_cmd.load(dx_path, map_name)
    cmd_string += f"cmd.load('{dx_path}', '{map_name}')\n"

    # Build a volume colour ramp: background (0) transparent; each integer
    # label gets a distinct opaque colour in a ±0.4 window around its value.
    # Format: [value, r, g, b, alpha, ...]
    ramp = [0.0, 1.0, 1.0, 1.0, 0.0]  # background transparent
    for i, v in enumerate(label_values):
        (r, g, b), _ = _PYMOL_CLUSTER_COLORS[i % len(_PYMOL_CLUSTER_COLORS)]
        v = float(v)
        ramp += [v - 0.4, r, g, b, 0.0,
                 v - 0.05, r, g, b, 0.7,
                 v + 0.05, r, g, b, 0.7,
                 v + 0.4, r, g, b, 0.0]

    _pymol_cmd.volume(vol_name, map_name)
    _pymol_cmd.volume_ramp_new(ramp_name, ramp)
    _pymol_cmd.volume_color(vol_name, ramp_name)
    cmd_string += f"cmd.volume('{vol_name}', '{map_name}')\n"
    cmd_string += f"cmd.volume_ramp_new('{ramp_name}', {ramp})\n"
    cmd_string += f"cmd.volume_color('{vol_name}', '{ramp_name}')\n"

    # --- Centroid pseudoatoms ---
    for lbl in site_labels:
        com_vox = center_of_mass(np.abs(combined_grid.grid), labeled_array, lbl)
        centroid = voxel_to_angstrom_fn(combined_grid, com_vox)
        x, y, z = float(centroid[0]), float(centroid[1]), float(centroid[2])

        site = site_by_id.get(lbl)
        label_text = f"rank{site.rank} agfe={site.agfe_min:.2f}" if site else f"lbl{lbl}"

        pa_name = f'site_{cosolvent}_lbl{lbl}'
        _pymol_cmd.pseudoatom(pa_name, pos=[x, y, z], label=label_text)
        _pymol_cmd.show('label', pa_name)
        cmd_string += (
            f"cmd.pseudoatom('{pa_name}', pos=[{x:.3f}, {y:.3f}, {z:.3f}], "
            f"label='{label_text}')\n"
        )
        cmd_string += f"cmd.show('label', '{pa_name}')\n"

    _pymol_cmd.set('label_size', 14)
    _pymol_cmd.set('specular', 1)
    _pymol_cmd.bg_color('white')
    cmd_string += "cmd.set('label_size', 14)\n"
    cmd_string += "cmd.set('specular', 1)\n"
    cmd_string += "cmd.bg_color('white')\n"

    pml_path = os.path.join(out_path, f"clustering_session_{cosolvent}.pml")
    pse_path = os.path.join(out_path, f"clustering_session_{cosolvent}.pse")

    with open(pml_path, 'w') as fh:
        fh.write(cmd_string)

    _pymol_cmd.save(pse_path)
    logger.info(f"Clustering PyMol session saved to {pse_path}")
    return pse_path


def add_hotspots_to_pymol_session(results, pse_path, out_path, top_n=10):
    """Add hotspot pseudoatom spheres to an existing PyMol session file.

    The ``.pse`` file is overwritten in-place.  Pseudoatom commands are
    also appended to the ``.pml`` script (if it exists).

    Parameters
    ----------
    results : dict[str, list[Hotspot]]
    pse_path : str
        Path to existing ``.pse`` file.
    out_path : str
        Directory containing the ``.pml`` script (if any).
    top_n : int
        Maximum sites per cosolvent to add (default 10).
    """
    if not _PYMOL_AVAILABLE:
        logger.warning("PyMol is not available — skipping hotspot session update.")
        return

    _RANK_COLORS = {1: "tv_green", 2: "yellow", 3: "orange", 4: "salmon", 5: "tv_red"}
    _DEFAULT_COLOR = "grey"

    _pymol_cmd.load(pse_path)
    pml_lines = ["\n# Hotspot sites added by HotspotDetector\n"]

    for cosolvent, sites in results.items():
        group_members = []
        for site in sites[:top_n]:
            name = f"hotspot_{cosolvent}_rank{site.rank}"
            color = _RANK_COLORS.get(site.rank, _DEFAULT_COLOR)
            vdw = min(site.n_voxels / 50.0, 4.0)
            cx, cy, cz = float(site.centroid[0]), float(site.centroid[1]), float(site.centroid[2])

            _pymol_cmd.pseudoatom(name, pos=[cx, cy, cz], vdw=vdw)
            _pymol_cmd.color(color, name)
            _pymol_cmd.show("spheres", name)
            group_members.append(name)

            pml_lines.append(
                f"pseudoatom {name}, pos=[{cx:.3f},{cy:.3f},{cz:.3f}], vdw={vdw:.2f}\n"
                f"color {color}, {name}\n"
                f"show spheres, {name}\n"
            )

        if group_members:
            group_name = f"hotspots_{cosolvent}"
            _pymol_cmd.group(group_name, " ".join(group_members))
            pml_lines.append(f"group {group_name}, {' '.join(group_members)}\n")

    _pymol_cmd.save(pse_path)
    logger.info(f"Updated PyMol session: {pse_path}")

    pml_path = pse_path.replace(".pse", ".pml")
    if os.path.exists(pml_path):
        with open(pml_path, "a") as fh:
            fh.writelines(pml_lines)
        logger.info(f"Appended hotspot commands to: {pml_path}")

