#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Plotly figure builders for hotspot detection results
# (moved verbatim from hotspot_visualization.py)
#

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

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
