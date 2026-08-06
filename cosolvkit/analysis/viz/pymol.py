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
    """Build a PyMol session with one isomesh per density map.

    Writes ``pymol_results_session.pse`` and the replayable ``pymol_session_cmd.pml``.

    :param out_path: directory where outputs are written.
    :type out_path: str
    :param cosolvent_names: list of cosolvent residue names.
    :type cosolvent_names: list
    :param avg_pdb_path: averaged-trajectory PDB used as the reference structure.
    :type avg_pdb_path: str
    :param density_files: .dx file(s) to load: a path, a directory, or a list.  If
        None, the final agfe maps in ``out_path`` are used.
    :type density_files: Union[str, list]
    :param selection_string: PyMol selection shown as sticks and spectrum-coloured
        by B-factor.
    :type selection_string: str
    :param reference_pdb: additional reference PDB to load alongside the average.
    :type reference_pdb: str
    :param compute_avg_structure: callable used when ``avg_pdb_path`` is missing and
        no ``reference_pdb`` is given; must create the file at ``avg_pdb_path``.
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
                # atomtypes mode: per-atom-type agfe maps, excluding 'raw' ones
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
        # see _contour_level_from_dx for the sign convention
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
# Hotspot PyMOL builders
# ---------------------------------------------------------------------------


def _contour_level_from_dx(dx_path):
    """Return an isomesh contour level for a DX file.

    AGFE maps (all values <= 0, favourable = negative) use the 0.1th percentile;
    positive maps (z-score density) use the 99.9th.
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

    All clusters go into a single label DX rendered as one volume with a
    per-cluster colour ramp, rather than one isomesh per cluster.  A labelled
    pseudoatom marks each site's centroid.  Writes
    ``clustering_session_{cosolvent}.pse`` / ``.pml`` and returns the ``.pse`` path.

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
        ``f(grid, vox_idx) -> np.ndarray``, voxel indices to Ångströms.
    reference_pdb : str, optional
        PDB loaded as structural context.
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

    # --- single DX for all clusters ---
    # Prefer the rank-label map written by export_results (voxel = rank); otherwise
    # write a site-ID map here.  The ramp values must match whichever is loaded.
    rank_dx = os.path.join(out_path, f"hotspot_labels_{cosolvent}.dx")
    if os.path.isfile(rank_dx):
        dx_path = rank_dx
        label_values = [site.rank for site in sorted(results, key=lambda s: s.rank)]
    else:
        dx_path = os.path.join(out_path, f"_cluster_labels_{cosolvent}.dx")
        Grid(labeled_array.astype(float), combined_grid.edges).export(dx_path)
        label_values = site_labels

    map_name = f'cluster_labels_{cosolvent}'
    vol_name = f'cluster_vol_{cosolvent}'
    ramp_name = f'ramp_clusters_{cosolvent}'

    _pymol_cmd.load(dx_path, map_name)
    cmd_string += f"cmd.load('{dx_path}', '{map_name}')\n"

    # Volume ramp, flat [value, r, g, b, alpha, ...]: background transparent, each
    # integer label opaque within a +/-0.4 window so neighbouring labels stay distinct.
    ramp = [0.0, 1.0, 1.0, 1.0, 0.0]
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

    # --- centroid pseudoatoms ---
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
    """Add hotspot pseudoatom spheres to an existing PyMol session.

    Sphere radius scales with cluster size (capped at 4 Å) and colour encodes
    rank.  The ``.pse`` is overwritten in place and the matching ``.pml``, if
    present, is appended to.  No-op when PyMol is unavailable.

    Parameters
    ----------
    results : dict[str, list[Hotspot]]
    pse_path : str
        Path to existing ``.pse`` file.
    out_path : str
        Directory containing the ``.pml`` script (if any).
    top_n : int
        Maximum sites per cosolvent to add.
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


# ---------------------------------------------------------------------------
# Binding-site PyMOL session
# ---------------------------------------------------------------------------

# Per-cosolvent isomesh colours (PyMol named colours).
_BS_COSOLVENT_COLORS = [
    "marine", "orange", "magenta", "salmon", "purple",
    "forest", "yellow", "cyan", "wheat", "slate",
]


def _write_mask_dx(voxel_mask, grid_origin, grid_delta, path):
    """Write a boolean voxel mask to *path* as a ``.dx`` (1 inside, 0 outside).

    Origin and per-axis spacing are in Å.  Returns *path*.
    """
    grid = np.asarray(voxel_mask, dtype=float)
    Grid(
        grid,
        origin=np.asarray(grid_origin, dtype=float),
        delta=np.asarray(grid_delta, dtype=float),
    ).export(path)
    return path


def _site_carve_radius(voxel_mask, grid_delta):
    """Isomesh carve radius (Å): half the mask bounding-box diagonal plus a 2 Å pad.

    Returns ``0.0`` for an empty mask.
    """
    idx = np.argwhere(voxel_mask)
    if idx.size == 0:
        return 0.0
    extent_vox = idx.max(axis=0) - idx.min(axis=0) + 1
    extent_ang = extent_vox * np.asarray(grid_delta, dtype=float)
    return float(0.5 * np.linalg.norm(extent_ang) + 2.0)


def generate_binding_site_session(binding_sites, reference_pdb, density_dir,
                                  out_path, top_n_sites=0):
    """Build a PyMol session showing each binding site's pocket + probe densities.

    Each site (all of them, or the top ``top_n_sites`` by rank if > 0) becomes a
    ``binding_site_{rank}`` group holding an isomesh of its union voxel mask plus
    one carved AGFE isomesh per member cosolvent, read from
    ``density_dir/map_agfe_{cosolvent}.dx``; missing maps are skipped with a warning.
    Note this reinitialises PyMol, discarding any loaded state.

    Writes ``binding_sites_session.pse`` and the replayable
    ``binding_sites_session.pml`` to *out_path*.  Returns the ``.pse`` path, or
    ``None`` if PyMol is unavailable.
    """
    if not _PYMOL_AVAILABLE:
        logger.warning("PyMol is not available — skipping binding-site session.")
        return None

    cmd = _pymol_cmd
    sites = sorted(
        binding_sites,
        key=lambda s: (s.rank if s.rank is not None else 1_000_000),
    )
    if top_n_sites and top_n_sites > 0:
        sites = sites[:top_n_sites]

    cmd.reinitialize()
    lines = ["# Binding-site session generated by generate_binding_site_session\n"]

    if reference_pdb and os.path.isfile(reference_pdb):
        struct = os.path.splitext(os.path.basename(reference_pdb))[0]
        cmd.load(reference_pdb, struct)
        cmd.color("grey70", f"{struct} and name C*")
        lines.append(f"cmd.load('{reference_pdb}', '{struct}')\n")
        lines.append(f"cmd.color('grey70', '{struct} and name C*')\n")
    else:
        logger.warning(
            "No reference PDB for binding-site session; continuing without protein."
        )

    for site in sites:
        rank = site.rank
        members = []

        # --- pocket surface (union mask) ---
        pocket_dx = os.path.join(out_path, f"bs{rank}_pocket.dx")
        _write_mask_dx(site.voxel_mask, site.grid_origin, site.grid_delta, pocket_dx)
        pocket_map = f"bs{rank}_pocket_map"
        pocket_mesh = f"bs{rank}_pocket"
        cmd.load(pocket_dx, pocket_map)
        cmd.isomesh(pocket_mesh, pocket_map, 0.5)
        cmd.color("grey50", pocket_mesh)
        members += [pocket_mesh, pocket_map]
        lines.append(f"cmd.load('{pocket_dx}', '{pocket_map}')\n")
        lines.append(f"cmd.isomesh('{pocket_mesh}', '{pocket_map}', 0.5)\n")
        lines.append(f"cmd.color('grey50', '{pocket_mesh}')\n")

        # --- carve anchor: isomesh(..., carve=r) needs a named selection ---
        cx, cy, cz = (float(site.centroid[0]),
                      float(site.centroid[1]),
                      float(site.centroid[2]))
        center = f"bs{rank}_center"
        cmd.pseudoatom(center, pos=[cx, cy, cz])
        radius = _site_carve_radius(site.voxel_mask, site.grid_delta)
        members += [center]
        lines.append(
            f"cmd.pseudoatom('{center}', pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}])\n"
        )

        # --- per-cosolvent densities, carved to the site region ---
        for i, cosolvent in enumerate(site.cosolvents):
            dx_path = os.path.join(density_dir, f"map_agfe_{cosolvent}.dx")
            if not os.path.isfile(dx_path):
                logger.warning(f"Density map not found for '{cosolvent}': {dx_path}")
                continue
            level = _contour_level_from_dx(dx_path)
            map_name = f"bs{rank}_{cosolvent}_map"
            mesh_name = f"bs{rank}_{cosolvent}_density"
            color = _BS_COSOLVENT_COLORS[i % len(_BS_COSOLVENT_COLORS)]
            cmd.load(dx_path, map_name)
            cmd.isomesh(mesh_name, map_name, level, center, carve=radius)
            cmd.color(color, mesh_name)
            members += [mesh_name, map_name]
            lines.append(f"cmd.load('{dx_path}', '{map_name}')\n")
            lines.append(
                f"cmd.isomesh('{mesh_name}', '{map_name}', {level}, "
                f"'{center}', carve={radius})\n"
            )
            lines.append(f"cmd.color('{color}', '{mesh_name}')\n")

        group = f"binding_site_{rank}"
        cmd.group(group, " ".join(members))
        lines.append(f"cmd.group('{group}', '{' '.join(members)}')\n")

    cmd.hide("spheres")
    cmd.set("specular", 1)
    cmd.bg_color("white")
    lines.append("cmd.hide('spheres')\n")
    lines.append("cmd.set('specular', 1)\n")
    lines.append("cmd.bg_color('white')\n")

    pml_path = os.path.join(out_path, "binding_sites_session.pml")
    pse_path = os.path.join(out_path, "binding_sites_session.pse")
    with open(pml_path, "w") as fh:
        fh.writelines(lines)
    cmd.save(pse_path)
    logger.info(f"Binding-site PyMol session saved to {pse_path}")
    return pse_path



# ---------------------------------------------------------------------------
# Combined hotspot + binding-site session
# ---------------------------------------------------------------------------

def _fmt_props(pairs):
    """``k=v`` label text, skipping missing values and rounding floats."""
    out = []
    for k, v in pairs:
        if v is None:
            continue
        if isinstance(v, float):
            if not np.isfinite(v):
                continue
            out.append(f"{k}={v:.3g}")
        else:
            out.append(f"{k}={v}")
    return " ".join(out)


def _hotspot_label(hs):
    p = hs.properties or {}
    return _fmt_props([
        ("rank", hs.rank),
        ("agfe", getattr(hs, "agfe_min", None)),
        ("vol", (hs.n_voxels * float(np.prod(hs.grid_delta)))
         if getattr(hs, "n_voxels", None) is not None and hs.grid_delta is not None else None),
        ("solidity", p.get("geom_solidity")),
        ("sharpness", p.get("field_sharpness")),
        ("contrast", p.get("field_contrast")),
    ])


def _site_label(site):
    return _fmt_props([
        ("rank", site.rank),
        ("score", getattr(site, "combined", None)),
        ("probes", ",".join(getattr(site, "cosolvents", []) or []) or None),
        ("nprobe", getattr(site, "n_cosolvents", None)),
        ("vol", getattr(site, "volume", None)),
        ("coverage", getattr(site, "probe_coverage", None)),
        ("residence", getattr(site, "residence", None)),
        ("accessible", (getattr(site, "properties", None) or {}).get("accessible_fraction")),
    ])


def write_full_session_script(probe_results, binding_sites, pml_path,
                              density_dir, reference_pdb=None, labels_on=True,
                              top_n_sites=0, keep_maps=True):
    """Emit a .pml building one session with a `hotspots` and a `bindingSites` group.

    hotspots/       one subgroup per probe, holding every hotspot's carved density
                    UNFILTERED -- the shape filter is a scoring decision, and a session for
                    inspecting what it would remove must not have removed it already.
    bindingSites/   one subgroup per ranked site, holding the site's union-mask pocket plus
                    the per-probe hotspot densities that were merged into it.

    Every label object is named ``*_lab`` so the set toggles with ``disable *_lab`` /
    ``enable *_lab``. PyMol puts an object in only one group, so a dedicated "labels" group
    would have to take them out of their probe/site groups.

    keep_maps : bool
        Keep the full AGFE volumes in the session. They are what makes it re-contourable, and
        also what makes it enormous: 18 probe maps put the FosAKP .pse at 2.4 GB, which most
        machines will not open. With ``keep_maps=False`` the maps are deleted once every
        isomesh has been built -- the meshes are standalone geometry and survive -- which
        trades interactive re-contouring for a session that loads.

    Returns *pml_path*. Writing the script does not require PyMol.
    """
    # Absolute paths: a .pml is meant to be replayable from anywhere, and relative ones
    # silently load nothing when PyMol's cwd is not the directory the script was generated in.
    density_dir = os.path.abspath(density_dir)
    if reference_pdb:
        reference_pdb = os.path.abspath(reference_pdb)

    dens = {}
    L = ["# Combined hotspot + binding-site session\n",
         "# Replay in ANY PyMol with:  @<this file>\n",
         "# Prefer this over the .pse: a .pse written by PyMol 3.x may not restore mesh\n",
         "# objects in an older PyMol, whereas this script rebuilds them from the .dx files.\n",
         "# Groups: hotspots/<probe>/<hotspot>, bindingSites/<rank>/<pocket + member probes>\n",
         "# Toggle every label with:  disable *_lab   /   enable *_lab\n\n"]

    if reference_pdb:
        struct = os.path.splitext(os.path.basename(reference_pdb))[0]
        L.append(f"load {reference_pdb}, {struct}\n")
        L.append(f"hide everything, {struct}\nshow cartoon, {struct}\n")
        L.append(f"color grey70, {struct}\n\n")

    # ---------------- hotspots ----------------
    probe_groups = []
    for res, hotspots in probe_results.items():
        dx = os.path.join(density_dir, f"map_agfe_{res}.dx")
        mapobj = f"map_{res}"
        dens[res] = mapobj
        L.append(f"# --- probe {res} ---\n")
        L.append(f"load {dx}, {mapobj}\n")
        members = [mapobj]
        for hs in hotspots:
            base = f"hs_{res}_r{hs.rank}"
            cx, cy, cz = (float(hs.centroid[0]), float(hs.centroid[1]), float(hs.centroid[2]))
            carve = _site_carve_radius(hs.voxel_mask, hs.grid_delta) if hs.voxel_mask is not None else 5.0
            L.append(f"pseudoatom {base}_anchor, pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}]\n")
            L.append(f"isomesh {base}_dens, {mapobj}, -1.0, {base}_anchor, carve={carve:.2f}\n")
            L.append(f"pseudoatom {base}_lab, pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}], "
                     f"label=\"{_hotspot_label(hs)}\"\n")
            members += [f"{base}_anchor", f"{base}_dens", f"{base}_lab"]
        L.append(f"group hs_{res}, {' '.join(members)}\n")
        L.append(f"disable {mapobj}\n\n")
        probe_groups.append(f"hs_{res}")
    if probe_groups:
        L.append(f"group hotspots, {' '.join(probe_groups)}\n\n")

    # ---------------- binding sites ----------------
    sites = sorted(binding_sites, key=lambda s: (s.rank if s.rank is not None else 10**6))
    if top_n_sites and top_n_sites > 0:
        sites = sites[:top_n_sites]
    site_groups = []
    for site in sites:
        rank = site.rank
        base = f"bs_{rank}"
        members = []
        pocket_dx = os.path.join(density_dir, f"{base}_pocket.dx")
        if site.voxel_mask is not None and site.grid_origin is not None:
            _write_mask_dx(site.voxel_mask, site.grid_origin, site.grid_delta, pocket_dx)
            L.append(f"load {pocket_dx}, {base}_pocket_map\n")
            L.append(f"isomesh {base}_pocket, {base}_pocket_map, 0.5\n")
            L.append(f"color grey50, {base}_pocket\n")
            members += [f"{base}_pocket_map", f"{base}_pocket"]
        cx, cy, cz = (float(site.centroid[0]), float(site.centroid[1]), float(site.centroid[2]))
        L.append(f"pseudoatom {base}_anchor, pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}]\n")
        members.append(f"{base}_anchor")
        # the per-probe densities that were merged into this site
        for hs in (site.member_hotspots or []):
            res = getattr(hs, "cosolvent", None) or _probe_of(hs, probe_results)
            if res is None or res not in dens:
                continue
            name = f"{base}_{res}"
            if name in members:
                continue
            carve = _site_carve_radius(site.voxel_mask, site.grid_delta) \
                if site.voxel_mask is not None else 6.0
            L.append(f"isomesh {name}, {dens[res]}, -1.0, {base}_anchor, carve={carve:.2f}\n")
            members.append(name)
        L.append(f"pseudoatom {base}_lab, pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}], "
                 f"label=\"{_site_label(site)}\"\n")
        members.append(f"{base}_lab")
        L.append(f"group {base}, {' '.join(members)}\n\n")
        site_groups.append(base)
    if site_groups:
        L.append(f"group bindingSites, {' '.join(site_groups)}\n\n")

    if not keep_maps:
        L.append("# Maps deleted: the isomeshes above are standalone geometry and remain.\n")
        L.append("# Re-run with keep_maps=True if you need to change contour levels here.\n")
        for res in dens:
            L.append(f"delete {dens[res]}\n")
        L.append("\n")

    L.append("# --- final state ---\n")
    L.append("hide everything, *_anchor\n")
    L.append("set label_size, 14\nset label_color, yellow\n")
    L.append(("enable *_lab\n" if labels_on else "disable *_lab\n"))
    if not labels_on:
        L.append("# labels are off; turn them on with:  enable *_lab\n")
    L.append("# disable *_lab   # hide every label\n")
    with open(pml_path, "w") as fh:
        fh.writelines(L)
    return pml_path


def _probe_of(hs, probe_results):
    """Which probe a hotspot belongs to (hotspots do not always carry the resname)."""
    for res, hotspots in probe_results.items():
        if any(h is hs for h in hotspots):
            return res
    return None


def generate_full_session(probe_results, binding_sites, out_path, density_dir,
                          reference_pdb=None, labels_on=True, top_n_sites=0,
                          keep_maps=True):
    """Write the combined session .pml and, when PyMol is importable, the .pse beside it."""
    pml = os.path.join(out_path, "full_session.pml")
    os.makedirs(out_path, exist_ok=True)
    write_full_session_script(probe_results, binding_sites, pml, density_dir,
                              reference_pdb=reference_pdb, labels_on=labels_on,
                              top_n_sites=top_n_sites, keep_maps=keep_maps)
    if not _PYMOL_AVAILABLE:
        logger.warning("PyMol unavailable - wrote %s but no .pse", pml)
        return pml
    cmd = _pymol_cmd
    cmd.reinitialize()
    cmd.do(f"@{pml}")
    pse = os.path.join(out_path, "full_session.pse")
    cmd.save(pse)
    return pse
