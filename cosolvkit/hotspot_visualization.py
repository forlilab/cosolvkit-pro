#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Visualization utilities for hotspot detection results
#

import os
import logging
from glob import glob as _glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gridData import Grid
from scipy.ndimage import center_of_mass

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    from pymol import cmd as _pymol_cmd
    _PYMOL_AVAILABLE = True
except ImportError:
    _PYMOL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotly — 3D clustering viewer
# ---------------------------------------------------------------------------

# Distinct colours for up to 20 clusters (CSS named colours)
_CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def plot_hotspot_clustering_3d(
    labeled_array,
    agfe_array,
    sites,
    combined_grid,
    cosolvent,
    agfe_cutoff,
    output_path=None,
    max_voxels_per_cluster=3000,
    top_n=10,
):
    """Interactive 3-D Plotly figure of hotspot clusters from :class:`HotspotDetector`.

    Each cluster is rendered as a translucent point cloud in Angstrom space.
    Centroids are overlaid as larger markers with hover labels showing rank,
    composite score, and AGFE min.  The clustering and scoring are expected
    to have been performed already (e.g. via :meth:`HotspotDetector.detect`).

    Parameters
    ----------
    labeled_array : np.ndarray of int
        3-D cluster label grid (0 = background, positive ints = cluster IDs).
        Produced by the clustering strategy inside :meth:`HotspotDetector.detect`.
    agfe_array : np.ndarray of float
        3-D AGFE grid values (same shape as *labeled_array*).
    sites : list[BindingSite]
        Ranked binding sites returned by :meth:`HotspotDetector.detect`.
    combined_grid : gridData.Grid
        Grid object used for voxel-to-Angstrom coordinate conversion.
    cosolvent : str
        Cosolvent residue name — used in the figure title.
    agfe_cutoff : float
        AGFE threshold (kcal/mol) used to define favorable voxels — shown in title.
    output_path : str, optional
        If given, save an interactive HTML file to this path.
    max_voxels_per_cluster : int
        Maximum number of voxels rendered per cluster (random subsampling is
        applied when a cluster is larger).  Default 3000.
    top_n : int
        Maximum number of sites to plot, taken in rank order.  Default 10.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for plot_hotspot_clustering_3d. "
            "Install it with: pip install plotly"
        )

    origin = np.array(combined_grid.origin)
    delta = np.array(combined_grid.delta)
    if delta.ndim == 2:
        # General grid — extract diagonal (assumes orthogonal axes)
        delta = np.diag(delta)

    traces = []
    sites_to_plot = sorted(sites, key=lambda s: s.rank)[:top_n]

    for i, site in enumerate(sites_to_plot):
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        vox_coords = np.argwhere(labeled_array == site.site_id)

        # Subsample if needed
        if len(vox_coords) > max_voxels_per_cluster:
            idx = np.random.choice(len(vox_coords), max_voxels_per_cluster, replace=False)
            vox_coords = vox_coords[idx]

        # Convert voxel indices → Angstroms
        ang_coords = origin + vox_coords * delta  # (N, 3)
        agfe_vals = agfe_array[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]

        hover = (
            f"Rank {site.rank}<br>"
            f"Cluster ID: {site.site_id}<br>"
            f"AGFE: %{{customdata:.3f}} kcal/mol<br>"
            f"Composite score: {site.composite_score:.3f}<br>"
            f"Voxels: {site.n_voxels}"
        )

        traces.append(go.Scatter3d(
            x=ang_coords[:, 0],
            y=ang_coords[:, 1],
            z=ang_coords[:, 2],
            mode="markers",
            name=f"Rank {site.rank} (ID {site.site_id})",
            customdata=agfe_vals,
            hovertemplate=hover,
            marker=dict(
                size=3,
                color=color,
                opacity=0.35,
            ),
            legendgroup=f"cluster_{site.site_id}",
            showlegend=True,
        ))

        # Centroid marker
        cx, cy, cz = float(site.centroid[0]), float(site.centroid[1]), float(site.centroid[2])
        traces.append(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text",
            name=f"Rank {site.rank} centroid",
            text=[f"R{site.rank}"],
            textposition="top center",
            hovertemplate=(
                f"<b>Rank {site.rank}</b><br>"
                f"Centroid: ({cx:.2f}, {cy:.2f}, {cz:.2f}) Å<br>"
                f"AGFE min: {site.agfe_min:.3f} kcal/mol<br>"
                f"Composite score: {site.composite_score:.3f}<br>"
                f"Voxels: {site.n_voxels}"
                "<extra></extra>"
            ),
            marker=dict(
                size=5,
                color=color,
                symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            legendgroup=f"cluster_{site.site_id}",
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"Hotspot clustering — {cosolvent} "
                f"(AGFE cutoff {agfe_cutoff} kcal/mol, {len(sites_to_plot)} site(s))"
            ),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"3D clustering plot saved to {output_path}")

    return fig


def plot_sp_raw(cosolvent_name, df_sp, out_path):
    """Plot raw survival-probability curves with hotspot rank as legend labels.

    Parameters
    ----------
    cosolvent_name : str
    df_sp : pd.DataFrame
        Columns: Group, Zone, Time, SP, Cosolvent.
    out_path : str
        Directory where the PNG is saved.
    """
    n_groups = df_sp["Group"].nunique()
    palette = sns.color_palette("flare", n_colors=max(n_groups, 1))
    fig, ax = plt.subplots()
    for zone_idx, group_df in df_sp.groupby("Group"):
        rank = int(zone_idx) + 1
        ax.plot(group_df["Time"], group_df["SP"],
                label=f"Rank {rank}", color=palette[int(zone_idx)])
    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{cosolvent_name} — Survival Probability")
    ax.legend(title="Hotspot")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, f"survival_probability_{cosolvent_name}.png"))
    plt.close(fig)


def plot_sp_fits(cosolvent, sites, df, out_path):
    """Overlay fitted decay curves on SP data — one figure per model.

    Writes ``survival_probability_fit_{model}_{cosolvent}.png`` for each of
    the two models: single-exp and bi-exponential.

    Parameters
    ----------
    cosolvent : str
    sites : list[BindingSite]
    df : pd.DataFrame
        SP data as written by ``survival_probability()``.
    out_path : str
        Directory where PNGs are saved.
    """
    def _single_exp(t, tau):
        return np.exp(-t / tau)

    def _bi_exp(t, A, tau1, tau2):
        return A * np.exp(-t / tau1) + (1.0 - A) * np.exp(-t / tau2)

    site_by_rank = {site.rank: site for site in sites}
    n_groups = df["Group"].nunique()
    palette = sns.color_palette("flare", n_colors=max(n_groups, 1))

    models = [
        (
            "single", "Single-exponential",
            _single_exp,
            lambda p: (p.get("sp_tau_single"),),
            lambda p: f"τ={p['sp_tau_single']:.1f}, R²={p.get('sp_r2_single', 0):.3f}",
        ),
        (
            "biexp", "Bi-exponential",
            _bi_exp,
            lambda p: (p.get("sp_amplitude_fast"), p.get("sp_tau_fast"), p.get("sp_tau_slow")),
            lambda p: (
                f"A={p['sp_amplitude_fast']:.2f}, "
                f"τ_fast={p['sp_tau_fast']:.1f}, "
                f"τ_slow={p['sp_tau_slow']:.1f}, "
                f"R²={p.get('sp_r2_biexp', 0):.3f}"
            ),
        ),
    ]

    for model_key, model_title, model_fn, param_getter, label_fn in models:
        fig, ax = plt.subplots()
        for zone_idx, group_df in df.groupby("Group"):
            rank = int(zone_idx) + 1
            site = site_by_rank.get(rank)
            color = palette[int(zone_idx)]
            tau_arr = group_df["Time"].values.astype(float)
            sp_arr = group_df["SP"].values.astype(float)

            ax.scatter(tau_arr, sp_arr, color=color, s=10, alpha=0.5, zorder=2)

            if site is not None:
                params = param_getter(site.properties)
                if all(v is not None for v in params):
                    t_fine = np.linspace(tau_arr[0], tau_arr[-1], 300)
                    ax.plot(
                        t_fine, model_fn(t_fine, *params),
                        color=color,
                        label=f"Rank {rank} — {label_fn(site.properties)}",
                    )
                else:
                    ax.plot([], [], color=color, label=f"Rank {rank} (fit failed)")

        ax.set_xlabel("Lag time (frames)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"{cosolvent} — {model_title} fit")
        ax.legend(title="Hotspot", fontsize="small")
        fig.tight_layout()
        out = os.path.join(out_path, f"survival_probability_fit_{model_key}_{cosolvent}.png")
        fig.savefig(out)
        plt.close(fig)
        logger.info(f"Saved {model_title} fit plot: {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# PyMol
# ---------------------------------------------------------------------------


# Pharmacophore atom-type → PyMOL colour name.
# Keys are element symbols or common GAFF/AMBER type prefixes.
_PHARMACOPHORE_COLORS = {
    'C':  'yellow',
    'c':  'yellow',
    'N':  'marine',
    'n':  'marine',
    'O':  'red',
    'o':  'red',
    'S':  'tv_green',
    's':  'tv_green',
    'Cl': 'cyan',
    'Br': 'orange',
    'F':  'palegreen',
    'I':  'purple',
    'P':  'salmon',
}


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
    results : list[BindingSite]
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
        label_text = f"rank{site.rank} s={site.composite_score:.2f}" if site else f"lbl{lbl}"

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
    results : dict[str, list[BindingSite]]
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


# ---------------------------------------------------------------------------
# New canonical sessions (replace the three legacy session creation points)
# ---------------------------------------------------------------------------


def generate_consensus_pockets_session(
    consensus_sites,
    out_path,
    reference_pdb=None,
    top_n=None,
):
    """Generate a PyMOL session visualising all consensus pockets.

    Objects are organised into one PyMOL **group per pocket rank**.  Inside
    each group the contributing AGFE density for every member probe is shown
    as a separate isomesh, plus a labelled centroid pseudoatom.

    Parameters
    ----------
    consensus_sites : list[ConsensusSite]
        Ranked consensus sites from :class:`CrossProbeConsensusDetector`.
    out_path : str
        Directory that contains the AGFE ``.dx`` maps and receives the session.
    reference_pdb : str, optional
        PDB file to load as structural context.
    top_n : int, optional
        Limit to the first *top_n* pockets (default: all).

    Returns
    -------
    str or None
        Path to the saved ``.pse`` file, or *None* if PyMOL is unavailable.
    """
    if not _PYMOL_AVAILABLE:
        logger.warning("PyMOL not available — skipping consensus pockets session.")
        return None

    _pymol_cmd.reinitialize()

    sites = sorted(consensus_sites, key=lambda s: s.consensus_rank)
    if top_n is not None:
        sites = sites[:top_n]

    # Reference structure
    if reference_pdb and os.path.isfile(reference_pdb):
        struct_name = os.path.splitext(os.path.basename(reference_pdb))[0]
        _pymol_cmd.load(reference_pdb, struct_name)
        _pymol_cmd.color('grey50', f'{struct_name} and name C*')

    n_colors = len(_PYMOL_CLUSTER_COLORS)

    for site in sites:
        rank = site.consensus_rank
        group_members = []

        # --- centroid sphere --------------------------------------------------
        cx, cy, cz = (float(v) for v in site.consensus_centroid)
        centroid_name = f'pocket_rank{rank}_centroid'
        label_text = (
            f'R{rank} score={site.consensus_score:.2f} '
            f'({len(site.member_cosolvents)} probe(s))'
        )
        rank_color = _PYMOL_CLUSTER_COLORS[(rank - 1) % n_colors][1]

        _pymol_cmd.pseudoatom(centroid_name, pos=[cx, cy, cz], label=label_text)
        _pymol_cmd.show('label', centroid_name)
        _pymol_cmd.show('spheres', centroid_name)
        _pymol_cmd.set('sphere_scale', 2.0, centroid_name)
        _pymol_cmd.color(rank_color, centroid_name)
        group_members.append(centroid_name)

        # --- per-probe density isomeshes --------------------------------------
        seen_cosolvents = set()
        probe_color_idx = rank % n_colors  # start offset away from rank color

        for member_site in site.member_sites:
            cosolvent = member_site.cosolvent
            if cosolvent in seen_cosolvents:
                continue
            seen_cosolvents.add(cosolvent)

            # Prefer the combined AGFE map; fall back to the first per-type map.
            dx_path = os.path.join(out_path, f"map_agfe_{cosolvent}.dx")
            if not os.path.isfile(dx_path):
                candidates = sorted(
                    f for f in _glob(os.path.join(out_path, f"map_agfe_*_{cosolvent}.dx"))
                    if 'raw' not in os.path.basename(f)
                )
                if not candidates:
                    logger.warning(f"No AGFE map for {cosolvent} in {out_path} — skipping probe.")
                    continue
                dx_path = candidates[0]

            try:
                contour = _contour_level_from_dx(dx_path)
            except Exception as exc:
                logger.warning(f"Could not read {dx_path}: {exc} — skipping.")
                continue

            probe_color = _PYMOL_CLUSTER_COLORS[probe_color_idx % n_colors][1]
            probe_color_idx += 1

            map_name  = f'pocket_rank{rank}_{cosolvent}_map'
            mesh_name = f'pocket_rank{rank}_{cosolvent}_density'
            _pymol_cmd.load(dx_path, map_name)
            _pymol_cmd.isomesh(mesh_name, map_name, contour)
            _pymol_cmd.color(probe_color, mesh_name)
            group_members.extend([map_name, mesh_name])

        if group_members:
            _pymol_cmd.group(f'pocket_rank{rank}', ' '.join(group_members))

    _pymol_cmd.set('label_size', 14)
    _pymol_cmd.set('specular', 1)
    _pymol_cmd.bg_color('white')

    pse_path = os.path.join(out_path, "consensus_pockets_session.pse")
    _pymol_cmd.save(pse_path)
    logger.info(f"Consensus pockets PyMOL session saved: {pse_path}")
    return pse_path


def generate_pharmacophore_session(
    consensus_sites,
    out_path,
    reference_pdb=None,
    top_n=3,
):
    """Generate a PyMOL session painting per-atom-type densities for the top pockets.

    For each of the *top_n* consensus pockets a PyMOL **group** is created.
    Inside, per-probe **sub-groups** contain one isomesh per atom type, coloured
    by pharmacophore feature (hydrophobic → yellow, H-bond donor → blue,
    acceptor → red, etc.).  When only combined AGFE maps are available (no
    per-atom-type breakdown) a single isomesh per probe is shown instead.

    Parameters
    ----------
    consensus_sites : list[ConsensusSite]
        Ranked consensus sites from :class:`CrossProbeConsensusDetector`.
    out_path : str
        Directory that contains the AGFE ``.dx`` maps and receives the session.
    reference_pdb : str, optional
        PDB file to load as structural context.
    top_n : int
        Number of top-ranked pockets to include (default 3).

    Returns
    -------
    str or None
        Path to the saved ``.pse`` file, or *None* if PyMOL is unavailable.
    """
    if not _PYMOL_AVAILABLE:
        logger.warning("PyMOL not available — skipping pharmacophore session.")
        return None

    _pymol_cmd.reinitialize()

    sites = sorted(consensus_sites, key=lambda s: s.consensus_rank)[:top_n]

    # Reference structure
    if reference_pdb and os.path.isfile(reference_pdb):
        struct_name = os.path.splitext(os.path.basename(reference_pdb))[0]
        _pymol_cmd.load(reference_pdb, struct_name)
        _pymol_cmd.color('grey50', f'{struct_name} and name C*')

    n_colors = len(_PYMOL_CLUSTER_COLORS)

    for site in sites:
        rank = site.consensus_rank
        pocket_members = []

        # --- centroid label ---------------------------------------------------
        cx, cy, cz = (float(v) for v in site.consensus_centroid)
        centroid_name = f'pocket_rank{rank}_centroid'
        label_text = f'R{rank} score={site.consensus_score:.2f}'
        rank_color = _PYMOL_CLUSTER_COLORS[(rank - 1) % n_colors][1]

        _pymol_cmd.pseudoatom(centroid_name, pos=[cx, cy, cz], label=label_text)
        _pymol_cmd.show('label', centroid_name)
        _pymol_cmd.show('spheres', centroid_name)
        _pymol_cmd.set('sphere_scale', 1.5, centroid_name)
        _pymol_cmd.color(rank_color, centroid_name)
        pocket_members.append(centroid_name)

        # --- per-probe pharmacophore isomeshes --------------------------------
        seen_cosolvents = set()
        fallback_color_idx = 0

        for member_site in site.member_sites:
            cosolvent = member_site.cosolvent
            if cosolvent in seen_cosolvents:
                continue
            seen_cosolvents.add(cosolvent)

            # Collect per-atom-type maps for this probe.
            per_type_files = sorted(
                f for f in _glob(os.path.join(out_path, f"map_agfe_*_{cosolvent}.dx"))
                if 'raw' not in os.path.basename(f)
            )

            if not per_type_files:
                # Fall back to combined AGFE map
                combined = os.path.join(out_path, f"map_agfe_{cosolvent}.dx")
                if os.path.isfile(combined):
                    per_type_files = [combined]
                else:
                    logger.warning(f"No AGFE maps for {cosolvent} in {out_path} — skipping probe.")
                    continue

            probe_members = []
            for dx_path in per_type_files:
                fname = os.path.basename(dx_path)
                # Derive atom-type label from filename
                # pattern: map_agfe_{atomtype}_{cosolvent}.dx  or  map_agfe_{cosolvent}.dx
                stem = fname[len('map_agfe_'):-len('.dx')]
                cosolvent_suffix = f'_{cosolvent}'
                if stem.endswith(cosolvent_suffix):
                    atomtype = stem[:-len(cosolvent_suffix)]
                else:
                    atomtype = cosolvent  # combined map — use probe name as label

                pymol_color = _PHARMACOPHORE_COLORS.get(atomtype)
                if pymol_color is None:
                    # Unknown type — cycle through distinct colours
                    pymol_color = _PYMOL_CLUSTER_COLORS[fallback_color_idx % n_colors][1]
                    fallback_color_idx += 1

                try:
                    contour = _contour_level_from_dx(dx_path)
                except Exception as exc:
                    logger.warning(f"Could not read {dx_path}: {exc} — skipping.")
                    continue

                # Sanitise atom-type for PyMOL object names
                safe_atype = atomtype.replace('+', 'p').replace('-', 'm').replace(' ', '_')
                map_name  = f'r{rank}_{cosolvent}_{safe_atype}_map'
                mesh_name = f'r{rank}_{cosolvent}_{safe_atype}_mesh'
                _pymol_cmd.load(dx_path, map_name)
                _pymol_cmd.isomesh(mesh_name, map_name, contour)
                _pymol_cmd.color(pymol_color, mesh_name)
                probe_members.extend([map_name, mesh_name])

            if probe_members:
                probe_group = f'pocket_rank{rank}_{cosolvent}'
                _pymol_cmd.group(probe_group, ' '.join(probe_members))
                pocket_members.append(probe_group)

        if pocket_members:
            _pymol_cmd.group(f'pocket_rank{rank}', ' '.join(pocket_members))

    _pymol_cmd.set('label_size', 14)
    _pymol_cmd.set('specular', 1)
    _pymol_cmd.bg_color('white')

    pse_path = os.path.join(out_path, "pharmacophore_session.pse")
    _pymol_cmd.save(pse_path)
    logger.info(f"Pharmacophore PyMOL session saved: {pse_path}")
    return pse_path
