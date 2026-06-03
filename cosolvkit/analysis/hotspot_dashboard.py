#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Interactive Plotly/Dash dashboard for visualizing hotspot detection results
#

import os
import socket
import logging
from glob import glob
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import html, dcc, Input, Output
    import dash_bio
    from dash import dash_table
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COSOLVENT_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]


def _make_cosolvent_color_map(cosolvents: List[str]) -> Dict[str, str]:
    return {c: _COSOLVENT_PALETTE[i % len(_COSOLVENT_PALETTE)]
            for i, c in enumerate(sorted(cosolvents))}


# ---------------------------------------------------------------------------
# Lightweight BindingSite stand-in (for data loaded from CSV)
# ---------------------------------------------------------------------------

class _SiteLike:
    """Minimal BindingSite replacement constructed from CSV rows."""

    def __init__(self, rank, site_id, centroid, composite_score, agfe_min, n_voxels):
        self.rank = rank
        self.site_id = site_id
        self.centroid = np.asarray(centroid, dtype=float)
        self.composite_score = composite_score
        self.agfe_min = agfe_min
        self.n_voxels = n_voxels


# ---------------------------------------------------------------------------
# PDB parsing for Molecule3dViewer
# ---------------------------------------------------------------------------

def _parse_pdb_for_viewer(pdb_path: str) -> dict:
    """Parse a PDB file into the dict format required by dash_bio.Molecule3dViewer.

    Includes all protein atoms (no trimming) and builds backbone + CA→CB bonds
    by walking MDAnalysis residues — never triggers slow bond guessing.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file (any format MDAnalysis can read).

    Returns
    -------
    dict
        ``{"atoms": [...], "bonds": [...]}`` as expected by Molecule3dViewer.
    """
    try:
        from MDAnalysis import Universe
    except ImportError:
        raise ImportError("MDAnalysis is required to parse PDB files.")

    u = Universe(pdb_path)
    sel = u.select_atoms("protein")
    if len(sel) == 0:
        sel = u.atoms  # fallback: everything

    # local index: global MDAnalysis index → position in atoms list
    local_idx = {a.index: i for i, a in enumerate(sel)}

    atoms = []
    for a in sel:
        elem = ""
        if hasattr(a, "element"):
            elem = a.element.strip()
        if not elem:
            elem = a.name[0]
        segid = a.segid.strip() if a.segid else "A"
        atoms.append({
            "serial": int(a.index),
            "name": a.name,
            "elem": elem,
            "positions": [
                float(a.position[0]),
                float(a.position[1]),
                float(a.position[2]),
            ],
            "residue_index": int(a.resid),
            "residue_name": a.resname,
            "chain": segid or "A",
            "bfactor": float(getattr(a, "tempfactor", 0.0)),
        })

    # Build backbone + CA→CB bonds by walking residues.
    # This is O(n_residues) and never triggers MDAnalysis bond guessing.
    _BB_PAIRS = [("N", "CA"), ("CA", "C"), ("C", "O")]
    bonds = []
    prev_C: dict = {}  # chain → local index of previous residue's C atom

    for res in u.select_atoms("protein").residues:
        chain = res.segid.strip() or "A"
        # name → local atom index for atoms that made it into sel
        name_map = {
            a.name: local_idx[a.index]
            for a in res.atoms
            if a.index in local_idx
        }

        # Intra-residue backbone bonds: N-CA, CA-C, C-O
        for n1, n2 in _BB_PAIRS:
            if n1 in name_map and n2 in name_map:
                bonds.append({"atom1_index": name_map[n1], "atom2_index": name_map[n2]})

        # First sidechain bond: CA → CB
        if "CA" in name_map and "CB" in name_map:
            bonds.append({"atom1_index": name_map["CA"], "atom2_index": name_map["CB"]})

        # Inter-residue peptide bond: previous C → this N
        if chain in prev_C and "N" in name_map:
            bonds.append({"atom1_index": prev_C[chain], "atom2_index": name_map["N"]})

        # Track this residue's C for the next iteration
        if "C" in name_map:
            prev_C[chain] = name_map["C"]
        else:
            prev_C.pop(chain, None)

    return {"atoms": atoms, "bonds": bonds}


# ---------------------------------------------------------------------------
# RMSF → per-atom colour styles
# ---------------------------------------------------------------------------

def _build_rmsf_styles(model_data: dict) -> list:
    """Return a list of per-atom style dicts for Molecule3dViewer.

    Colours each atom by its B-factor (RMSF) using the 'coolwarm' colormap:
    blue = low RMSF (rigid), red = high RMSF (flexible).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    bfactors = np.array([a["bfactor"] for a in model_data["atoms"]], dtype=float)
    vmin, vmax = bfactors.min(), bfactors.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap("coolwarm")

    styles = []
    for bf in bfactors:
        rgba = cmap(norm(bf))
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        styles.append({"visualization_type": "cartoon", "color": hex_color})
    return styles


# ---------------------------------------------------------------------------
# Hotspot data loading helpers
# ---------------------------------------------------------------------------

def _load_hotspot_csvs(search_dir: str) -> pd.DataFrame:
    """Load all ``hotspot_sites_*.csv`` files from *search_dir* into one DataFrame."""
    csv_files = sorted(glob(os.path.join(search_dir, "hotspot_sites_*.csv")))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as exc:
            logger.warning(f"Could not read {f}: {exc}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _build_sites_by_cosolvent(df: pd.DataFrame) -> Dict[str, List[_SiteLike]]:
    """Build ``{cosolvent: [_SiteLike, ...]}`` sorted by rank from DataFrame."""
    if df.empty:
        return {}
    result = {}
    for cosolvent, group in df.groupby("cosolvent"):
        sites = []
        for _, row in group.iterrows():
            centroid = np.array(
                [row["centroid_x"], row["centroid_y"], row["centroid_z"]], dtype=float
            )
            sites.append(_SiteLike(
                rank=int(row["rank"]),
                site_id=int(row["site_id"]),
                centroid=centroid,
                composite_score=float(row.get("composite_score", 0.0)),
                agfe_min=float(row.get("agfe_min", 0.0)),
                n_voxels=int(row.get("n_voxels", 0)),
            ))
        result[cosolvent] = sorted(sites, key=lambda s: s.rank)
    return result


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------

class HotspotDashboard:
    """Interactive Plotly/Dash dashboard for CoSolvKit hotspot visualization.

    Displays the reference protein structure (via ``dash_bio.Molecule3dViewer``)
    coloured by RMSF alongside interactive cosolvent pocket overlays.  A
    separate tab renders the full voxel point-cloud view using the existing
    :func:`~cosolvkit.hotspot_visualization.plot_hotspot_clustering_3d`
    function.  A sortable/filterable data table shows all hotspot metrics.

    Parameters
    ----------
    out_path : str
        Analysis output directory.  If a ``merged/`` subdirectory exists,
        hotspot CSVs and DX maps are loaded from there; otherwise *out_path*
        itself is searched.
    pdb_path : str, optional
        Path to the reference PDB file.  Auto-detected as
        ``averaged_trajectory.pdb`` in *out_path* or its parent when not given.
    port : int
        Port for the Dash development server (default 8050).
    agfe_cutoff : float
        AGFE cutoff label shown in voxel plot titles (default −1.0 kcal/mol).
    """

    def __init__(
        self,
        out_path: str,
        pdb_path: Optional[str] = None,
        port: int = 8050,
        agfe_cutoff: float = -1.0,
    ):
        if not _DASH_AVAILABLE:
            raise ImportError(
                "dash and dash_bio are required for the dashboard.\n"
                "Install with: pip install dash dash-bio"
            )
        if not _PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")

        self.out_path = os.path.abspath(out_path)
        self.port = port
        self._agfe_cutoff = agfe_cutoff

        # Prefer merged/ subdirectory for maps and hotspot CSVs
        merged = os.path.join(self.out_path, "merged")
        self._map_dir = merged if os.path.isdir(merged) else self.out_path

        # Resolve PDB path
        self._pdb_path = pdb_path or self._find_pdb()

        # Load hotspot data
        self._df = _load_hotspot_csvs(self._map_dir)
        if self._df.empty:
            self._df = _load_hotspot_csvs(self.out_path)

        self._cosolvents: List[str] = (
            sorted(self._df["cosolvent"].unique().tolist())
            if not self._df.empty else []
        )
        self._color_map = _make_cosolvent_color_map(self._cosolvents)
        self._sites_by_cosolvent = _build_sites_by_cosolvent(self._df)

        # Parse PDB once at startup
        if self._pdb_path and os.path.exists(self._pdb_path):
            logger.info(f"Parsing reference PDB: {self._pdb_path}")
            self._model_data = _parse_pdb_for_viewer(self._pdb_path)
            self._styles = _build_rmsf_styles(self._model_data)
        else:
            if self._pdb_path:
                logger.warning(f"Reference PDB not found: {self._pdb_path}")
            else:
                logger.warning("No reference PDB detected — protein viewer will be empty.")
            self._model_data = {"atoms": [], "bonds": []}
            self._styles = []

        self._app = self._create_app()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pdb(self) -> Optional[str]:
        """Locate ``averaged_trajectory.pdb`` in common output locations."""
        candidates = [
            os.path.join(self.out_path, "averaged_trajectory.pdb"),
            os.path.join(os.path.dirname(self.out_path), "averaged_trajectory.pdb"),
        ]
        candidates += sorted(
            glob(os.path.join(self.out_path, "*/averaged_trajectory.pdb"))
        )
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _build_shapes(
        self, cosolvents: List[str], top_n: int, min_score: float
    ) -> list:
        """Build sphere shapes for Molecule3dViewer from hotspot centroids.

        Sphere radius is scaled by pocket volume (∝ n_voxels^(1/3)).
        """
        if self._df.empty:
            return []

        df = self._df.copy()
        if cosolvents:
            df = df[df["cosolvent"].isin(cosolvents)]
        if "composite_score" in df.columns:
            df = df[df["composite_score"] >= min_score]

        shapes = []
        for cosolvent, group in df.groupby("cosolvent"):
            color = self._color_map.get(cosolvent, "#888888")
            for _, row in group.nsmallest(top_n, "rank").iterrows():
                n_vox = max(1, float(row.get("n_voxels", 64)))
                radius = max(2.0, (n_vox ** (1.0 / 3.0)) * 0.4)
                shapes.append({
                    "type": "Sphere",
                    "x": float(row["centroid_x"]),
                    "y": float(row["centroid_y"]),
                    "z": float(row["centroid_z"]),
                    "color": color,
                    "radius": radius,
                    "opacity": 0.65,
                })
        return shapes

    def _load_voxel_figure(self, cosolvent: str, top_n: int) -> "go.Figure":
        """Load DX label/AGFE grids and return a Plotly voxel point-cloud figure.

        Reuses :func:`~cosolvkit.hotspot_visualization.plot_hotspot_clustering_3d`.
        """
        from cosolvkit.analysis.hotspot_visualization import plot_hotspot_clustering_3d

        label_dx = os.path.join(self._map_dir, f"hotspot_labels_{cosolvent}.dx")
        agfe_dx = os.path.join(self._map_dir, f"map_agfe_{cosolvent}.dx")

        if not os.path.exists(label_dx) or not os.path.exists(agfe_dx):
            fig = go.Figure()
            fig.update_layout(
                title=f"No voxel data found for {cosolvent} in {self._map_dir}",
                scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
            )
            return fig

        try:
            from gridData import Grid
        except ImportError:
            logger.error("gridData is required for voxel visualization.")
            return go.Figure()

        label_grid = Grid(label_dx)
        agfe_grid = Grid(agfe_dx)
        sites = self._sites_by_cosolvent.get(cosolvent, [])[:top_n]

        if not sites:
            fig = go.Figure()
            fig.update_layout(title=f"No hotspot sites found for {cosolvent}")
            return fig

        return plot_hotspot_clustering_3d(
            labeled_array=label_grid.grid.astype(int),
            agfe_array=agfe_grid.grid,
            sites=sites,
            combined_grid=agfe_grid,
            cosolvent=cosolvent,
            agfe_cutoff=self._agfe_cutoff,
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # Protein + Pockets Plotly figure
    # ------------------------------------------------------------------

    def _build_ca_traces(self) -> list:
        """Build one Scatter3d line trace per chain, Cα atoms colored by RMSF."""
        from collections import defaultdict

        ca_atoms = [a for a in self._model_data.get("atoms", []) if a["name"] == "CA"]
        if not ca_atoms:
            return []

        chains: dict = defaultdict(list)
        for a in ca_atoms:
            chains[a["chain"]].append(a)

        all_bfs = [a["bfactor"] for a in ca_atoms]
        bf_min, bf_max = min(all_bfs), max(all_bfs)
        if bf_max - bf_min < 1e-6:
            bf_max = bf_min + 1.0

        traces = []
        for chain, ca_list in sorted(chains.items()):
            sorted_ca = sorted(ca_list, key=lambda a: a["residue_index"])
            traces.append(go.Scatter3d(
                x=[a["positions"][0] for a in sorted_ca],
                y=[a["positions"][1] for a in sorted_ca],
                z=[a["positions"][2] for a in sorted_ca],
                mode="lines",
                name=f"Chain {chain}",
                line=dict(
                    width=5,
                    color=[a["bfactor"] for a in sorted_ca],
                    colorscale="RdBu_r",
                    cmin=bf_min,
                    cmax=bf_max,
                ),
                hovertext=[f"{a['residue_name']} {a['residue_index']}" for a in sorted_ca],
                hoverinfo="text+name",
                legendgroup="protein",
                showlegend=True,
            ))
        return traces

    def _build_protein_pockets_figure(
        self,
        df_filtered: pd.DataFrame,
        visible_site_ids: set,
    ) -> "go.Figure":
        """Plotly figure: protein Cα backbone + hotspot centroid spheres.

        Each hotspot is its own trace so it can be hidden independently via
        the ``visible`` flag driven by the checklist.
        """
        traces = self._build_ca_traces()

        if not df_filtered.empty:
            first = True
            for _, row in df_filtered.iterrows():
                site_id = f"{row['cosolvent']}_rank{int(row['rank'])}"
                color = self._color_map.get(row["cosolvent"], "#888888")
                n_vox = max(1.0, float(row.get("n_voxels", 64)))
                # Pixel size scaled by volume; clamped to sensible range
                msize = int(np.clip(n_vox ** (1.0 / 3.0) * 1.8, 12, 40))
                cx = float(row["centroid_x"])
                cy = float(row["centroid_y"])
                cz = float(row["centroid_z"])
                traces.append(go.Scatter3d(
                    x=[cx], y=[cy], z=[cz],
                    mode="markers+text",
                    name=f"[{row['cosolvent']}] Rank {int(row['rank'])}",
                    text=[f"R{int(row['rank'])}"],
                    textposition="top center",
                    textfont=dict(color="white", size=11),
                    marker=dict(
                        size=msize,
                        color=color,
                        opacity=0.75,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    ),
                    visible=True if site_id in visible_site_ids else False,
                    hovertemplate=(
                        f"<b>[{row['cosolvent']}] Rank {int(row['rank'])}</b><br>"
                        f"AGFE min: {float(row.get('agfe_min', 0)):.3f} kcal/mol<br>"
                        f"Score: {float(row.get('composite_score', 0)):.3f}<br>"
                        f"Voxels: {int(row.get('n_voxels', 0))}<br>"
                        f"({cx:.1f}, {cy:.1f}, {cz:.1f}) Å"
                        "<extra></extra>"
                    ),
                    legendgroup="hotspots",
                    legendgrouptitle_text="Hotspots" if first else None,
                ))
                first = False

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                bgcolor="#111827",
                xaxis=dict(backgroundcolor="#111827", gridcolor="#2d3748", zerolinecolor="#2d3748"),
                yaxis=dict(backgroundcolor="#111827", gridcolor="#2d3748", zerolinecolor="#2d3748"),
                zaxis=dict(backgroundcolor="#111827", gridcolor="#2d3748", zerolinecolor="#2d3748"),
                aspectmode="data",
            ),
            paper_bgcolor="#111827",
            font_color="#e2e8f0",
            legend=dict(
                bgcolor="rgba(17,24,39,0.85)",
                bordercolor="#4a5568",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=11),
                itemsizing="constant",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    def _get_table_columns(self) -> list:
        desired = [
            "rank", "cosolvent", "composite_score", "agfe_min",
            "favorability_score", "diversity_score", "volume_score",
            "n_voxels", "favorable_atomtypes",
            "centroid_x", "centroid_y", "centroid_z",
        ]
        if self._df.empty:
            return [{"name": c, "id": c} for c in desired]
        available = [c for c in desired if c in self._df.columns]
        return [{"name": c, "id": c} for c in available]

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _create_app(self) -> "dash.Dash":
        app = dash.Dash(__name__, title="CoSolvKit Hotspot Dashboard")

        cosolvent_options = [{"label": c, "value": c} for c in self._cosolvents]
        default_cosolvents = self._cosolvents[:1] if self._cosolvents else []

        label_style = {"fontWeight": "bold", "fontSize": "0.82em", "marginBottom": "4px"}
        btn_style = {
            "fontSize": "0.72em", "padding": "2px 10px", "marginRight": "6px",
            "border": "1px solid #cbd5e0", "borderRadius": "4px",
            "backgroundColor": "#edf2f7", "cursor": "pointer",
        }

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f3f7", "minHeight": "100vh"},
            children=[

                # ---- Header ----
                html.Div(
                    style={"backgroundColor": "#1a3a5c", "color": "white", "padding": "12px 24px"},
                    children=[
                        html.H1("CoSolvKit Hotspot Dashboard", style={"margin": "0", "fontSize": "1.5em"}),
                        html.P(f"Results: {self.out_path}",
                               style={"margin": "4px 0 0", "fontSize": "0.8em", "opacity": "0.75"}),
                    ],
                ),

                # ---- Controls bar ----
                html.Div(
                    style={
                        "backgroundColor": "white", "padding": "10px 24px",
                        "display": "flex", "alignItems": "center", "gap": "32px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "flexWrap": "wrap",
                    },
                    children=[
                        html.Div([
                            html.Div("Cosolvents", style=label_style),
                            dcc.Dropdown(
                                id="cosolvent-dd", options=cosolvent_options,
                                value=default_cosolvents, multi=True, clearable=False,
                                style={"minWidth": "200px"},
                            ),
                        ]),
                        html.Div([
                            html.Div(id="topn-label", style=label_style, children="Top N sites: 5"),
                            html.Div(style={"width": "180px"}, children=[
                                dcc.Slider(id="topn-slider", min=1, max=20, step=1, value=5,
                                           marks={i: str(i) for i in range(1, 21, 4)},
                                           tooltip={"placement": "bottom", "always_visible": False}),
                            ]),
                        ]),
                        html.Div([
                            html.Div(id="score-label", style=label_style, children="Min score: 0.00"),
                            html.Div(style={"width": "180px"}, children=[
                                dcc.Slider(id="score-slider", min=0.0, max=1.0, step=0.05, value=0.0,
                                           marks={v: f"{v:.1f}" for v in [0.0, 0.25, 0.5, 0.75, 1.0]},
                                           tooltip={"placement": "bottom", "always_visible": False}),
                            ]),
                        ]),
                        html.Div([
                            html.Div("RMSF coloring (protein):", style=label_style),
                            html.Div(style={
                                "background": "linear-gradient(to right, #3b4cc0, #dddcdc, #b40426)",
                                "width": "110px", "height": "10px", "borderRadius": "4px",
                            }),
                            html.Div(style={"display": "flex", "justifyContent": "space-between",
                                            "width": "110px", "fontSize": "0.72em", "color": "#555"},
                                     children=[html.Span("Low"), html.Span("High")]),
                        ]),
                    ],
                ),

                # ---- Main content ----
                html.Div(
                    style={"display": "flex", "height": "calc(100vh - 148px)"},
                    children=[

                        # Left: tabbed viewer (65 %)
                        html.Div(
                            style={"width": "65%", "padding": "10px", "overflow": "hidden"},
                            children=[
                                dcc.Tabs(
                                    value="pockets",
                                    style={"fontSize": "0.88em"},
                                    children=[

                                        # ── Tab 1: Protein + Pockets (Plotly) ──────────────
                                        dcc.Tab(
                                            label="Protein + Pockets",
                                            value="pockets",
                                            children=[
                                                dcc.Graph(
                                                    id="protein-pockets-graph",
                                                    style={"height": "calc(100vh - 220px)"},
                                                    config={"displayModeBar": True},
                                                ),
                                            ],
                                        ),

                                        # ── Tab 2: Pocket voxel cloud ──────────────────────
                                        dcc.Tab(
                                            label="Pocket Voxels",
                                            value="voxels",
                                            children=[
                                                html.Div(style={"padding": "8px 12px"}, children=[
                                                    html.Div([
                                                        html.Label("Cosolvent:",
                                                                   style={**label_style, "marginRight": "8px"}),
                                                        dcc.Dropdown(
                                                            id="voxel-cosolvent-dd",
                                                            options=cosolvent_options,
                                                            value=self._cosolvents[0] if self._cosolvents else None,
                                                            clearable=False,
                                                            style={"width": "180px", "display": "inline-block"},
                                                        ),
                                                    ], style={"display": "flex", "alignItems": "center",
                                                               "marginBottom": "6px"}),
                                                    dcc.Graph(
                                                        id="voxel-graph",
                                                        style={"height": "calc(100vh - 280px)"},
                                                        config={"displayModeBar": True},
                                                    ),
                                                ]),
                                            ],
                                        ),

                                        # ── Tab 3: Protein Structure (Molecule3dViewer) ────
                                        dcc.Tab(
                                            label="Protein Structure",
                                            value="structure",
                                            children=[
                                                dash_bio.Molecule3dViewer(
                                                    id="mol-viewer",
                                                    modelData=self._model_data,
                                                    styles=self._styles,
                                                    shapes=[],
                                                    backgroundColor="#111827",
                                                    backgroundOpacity=1.0,
                                                    style={"height": "calc(100vh - 220px)", "width": "100%"},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # Right: checklist + table (35 %)
                        html.Div(
                            style={
                                "width": "35%", "padding": "10px 12px",
                                "backgroundColor": "white", "overflowY": "auto",
                                "boxShadow": "-2px 0 6px rgba(0,0,0,0.07)",
                            },
                            children=[

                                # Per-hotspot visibility checkboxes
                                html.Div([
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center",
                                               "justifyContent": "space-between", "marginBottom": "6px"},
                                        children=[
                                            html.H3("Hotspot Visibility",
                                                    style={"margin": "0", "fontSize": "0.92em", "color": "#1a3a5c"}),
                                            html.Div([
                                                html.Button("All", id="check-all-btn", n_clicks=0, style=btn_style),
                                                html.Button("None", id="check-none-btn", n_clicks=0, style=btn_style),
                                            ]),
                                        ],
                                    ),
                                    dcc.Checklist(
                                        id="site-checklist",
                                        options=[],
                                        value=[],
                                        labelStyle={"display": "flex", "alignItems": "center",
                                                    "fontSize": "0.8em", "marginBottom": "3px",
                                                    "cursor": "pointer"},
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                ]),

                                html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

                                # Metrics table
                                html.H3("Hotspot Sites",
                                        style={"margin": "0 0 4px", "fontSize": "0.92em", "color": "#1a3a5c"}),
                                html.P(id="table-summary",
                                       style={"fontSize": "0.78em", "color": "#666", "margin": "0 0 8px"}),
                                dash_table.DataTable(
                                    id="hotspot-table",
                                    columns=self._get_table_columns(),
                                    data=[],
                                    sort_action="native",
                                    filter_action="native",
                                    page_action="native",
                                    page_size=15,
                                    style_table={"overflowX": "auto"},
                                    style_header={
                                        "backgroundColor": "#1a3a5c", "color": "white",
                                        "fontWeight": "bold", "fontSize": "0.75em", "whiteSpace": "normal",
                                    },
                                    style_cell={
                                        "fontSize": "0.75em", "padding": "4px 8px", "textAlign": "left",
                                        "maxWidth": "110px", "overflow": "hidden", "textOverflow": "ellipsis",
                                    },
                                    style_data_conditional=[
                                        {"if": {"row_index": "odd"}, "backgroundColor": "#f0f4f8"}
                                    ],
                                    tooltip_duration=None,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        self._register_callbacks(app)
        return app

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self, app: "dash.Dash"):
        from dash import ctx, State

        # ── Slider labels ─────────────────────────────────────────────────────

        @app.callback(Output("topn-label", "children"), Input("topn-slider", "value"))
        def update_topn_label(value):
            return f"Top N sites: {value}"

        @app.callback(Output("score-label", "children"), Input("score-slider", "value"))
        def update_score_label(value):
            return f"Min score: {value:.2f}"

        # ── Helper: filter DataFrame ──────────────────────────────────────────

        def _filtered_df(cosolvents, top_n, min_score):
            if self._df.empty:
                return pd.DataFrame()
            df = self._df.copy()
            if cosolvents:
                df = df[df["cosolvent"].isin(cosolvents)]
            if "composite_score" in df.columns:
                df = df[df["composite_score"] >= min_score]
            return (
                df.groupby("cosolvent", group_keys=False)
                .apply(lambda g: g.nsmallest(top_n, "rank"))
                .reset_index(drop=True)
            )

        # ── Checklist options + value ─────────────────────────────────────────
        # Rebuilds on filter changes; All/None buttons also drive the value.

        @app.callback(
            [Output("site-checklist", "options"),
             Output("site-checklist", "value")],
            [Input("cosolvent-dd", "value"),
             Input("topn-slider", "value"),
             Input("score-slider", "value"),
             Input("check-all-btn", "n_clicks"),
             Input("check-none-btn", "n_clicks")],
        )
        def update_checklist(cosolvents, top_n, min_score, _all, _none):
            cosolvents = cosolvents or self._cosolvents
            df_top = _filtered_df(cosolvents, top_n, min_score)

            options = []
            for _, row in df_top.iterrows():
                cosolvent = row["cosolvent"]
                rank = int(row["rank"])
                score = float(row.get("composite_score", 0.0))
                color = self._color_map.get(cosolvent, "#888")
                site_id = f"{cosolvent}_rank{rank}"
                options.append({
                    "label": html.Span([
                        html.Span(style={
                            "backgroundColor": color,
                            "width": "10px", "height": "10px",
                            "borderRadius": "50%", "display": "inline-block",
                            "marginRight": "6px", "flexShrink": "0",
                        }),
                        f"[{cosolvent}] Rank {rank}  score={score:.2f}",
                    ], style={"display": "flex", "alignItems": "center"}),
                    "value": site_id,
                })

            all_values = [o["value"] for o in options]
            triggered = ctx.triggered_id
            if triggered == "check-none-btn":
                return options, []
            # Default (filter change or All button): all checked
            return options, all_values

        # ── Protein + Pockets figure ──────────────────────────────────────────

        @app.callback(
            Output("protein-pockets-graph", "figure"),
            [Input("cosolvent-dd", "value"),
             Input("topn-slider", "value"),
             Input("score-slider", "value"),
             Input("site-checklist", "value")],
        )
        def update_protein_pockets(cosolvents, top_n, min_score, visible_ids):
            cosolvents = cosolvents or self._cosolvents
            df_top = _filtered_df(cosolvents, top_n, min_score)
            visible = set(visible_ids or [])
            return self._build_protein_pockets_figure(df_top, visible)

        # ── Table ─────────────────────────────────────────────────────────────

        @app.callback(
            [Output("hotspot-table", "data"),
             Output("table-summary", "children")],
            [Input("cosolvent-dd", "value"),
             Input("topn-slider", "value"),
             Input("score-slider", "value")],
        )
        def update_table(cosolvents, top_n, min_score):
            cosolvents = cosolvents or self._cosolvents
            df_top = _filtered_df(cosolvents, top_n, min_score)
            float_cols = [
                "composite_score", "agfe_min", "agfe_mean_top_pct",
                "favorability_score", "diversity_score", "volume_score",
                "centroid_x", "centroid_y", "centroid_z",
            ]
            for col in float_cols:
                if col in df_top.columns:
                    df_top[col] = df_top[col].round(3)
            n_cos = df_top["cosolvent"].nunique() if not df_top.empty else 0
            summary = f"{len(df_top)} site(s) across {n_cos} cosolvent(s)"
            return df_top.to_dict("records"), summary

        # ── Voxel cloud ───────────────────────────────────────────────────────

        @app.callback(
            Output("voxel-graph", "figure"),
            [Input("voxel-cosolvent-dd", "value"),
             Input("topn-slider", "value")],
        )
        def update_voxel_view(cosolvent, top_n):
            if not cosolvent:
                return go.Figure()
            return self._load_voxel_figure(cosolvent, top_n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_server(self, host: str = "0.0.0.0", debug: bool = False):
        """Start the Dash development server.

        Parameters
        ----------
        host : str
            Network interface to bind to.  ``"0.0.0.0"`` (default) listens on
            all interfaces, which is required for SSH port forwarding.
        debug : bool
            Enable Dash debug mode (hot-reloading, verbose errors).
        """
        _print_startup_banner(self.port, host)
        self._app.run(host=host, port=self.port, debug=debug)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SSH-aware startup banner
# ---------------------------------------------------------------------------

def _print_startup_banner(port: int, host: str) -> None:
    """Print connection instructions, with SSH port-forwarding help when relevant."""
    try:
        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
    except Exception:
        hostname = "hpc-node"
        fqdn = hostname

    local_url = f"http://localhost:{port}/"
    is_ssh = any(k in os.environ for k in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"))
    user = os.environ.get("USER", "user")

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  CoSolvKit Hotspot Dashboard")
    print(f"{sep}")
    print(f"  Server   : {fqdn}  (host flag: {host})")
    print(f"  Port     : {port}")

    if is_ssh:
        tunnel_cmd = f"ssh -L {port}:localhost:{port} {user}@{fqdn} -N"
        print(f"\n  Detected SSH session — the browser must run on your")
        print(f"  local machine.  Open a NEW local terminal and run:\n")
        print(f"    {tunnel_cmd}\n")
        print(f"  Then open your browser at:  {local_url}")
        print(f"\n  Tip: add -f to the ssh command to background the tunnel:")
        print(f"    ssh -fNL {port}:localhost:{port} {user}@{fqdn}")
    else:
        print(f"\n  Open your browser at:  {local_url}")

    print(f"{sep}\n")


def launch_dashboard(
    out_path: str,
    pdb_path: Optional[str] = None,
    port: int = 8050,
    host: str = "0.0.0.0",
    agfe_cutoff: float = -1.0,
    debug: bool = False,
):
    """Create and immediately start a :class:`HotspotDashboard`.

    Parameters
    ----------
    out_path : str
        Analysis output directory.
    pdb_path : str, optional
        Reference PDB (auto-detected when *None*).
    port : int
        Dash server port (default 8050).
    host : str
        Network interface to bind to (default ``"0.0.0.0"``).
    agfe_cutoff : float
        AGFE cutoff shown in voxel plot titles (default −1.0 kcal/mol).
    debug : bool
        Enable Dash debug mode.
    """
    dashboard = HotspotDashboard(
        out_path=out_path,
        pdb_path=pdb_path,
        port=port,
        agfe_cutoff=agfe_cutoff,
    )
    dashboard.run_server(host=host, debug=debug)
