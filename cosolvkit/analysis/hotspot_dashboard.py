#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Interactive Plotly/Dash dashboard for visualizing hotspot detection results
#

import os
import json
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
    from dash import html, dcc, Input, Output, State, ctx
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
# Binding-site loading + reranking
# ---------------------------------------------------------------------------

DEFAULT_DASHBOARD_WEIGHTS = {
    "affinity": 3.0, "probe_coverage": 2.0, "volume": 1.0,
    "kinetics": 1.0, "shape": 1.0, "diversity": 1.0,
}

# (weight key, slider id, display label, default value) — drives both the
# controls-bar layout and the label-echo callbacks.
_WEIGHT_SPECS = [
    ("affinity", "weight-affinity", "Affinity", 3),
    ("probe_coverage", "weight-probe_coverage", "Probe cov.", 2),
    ("volume", "weight-volume", "Volume", 1),
    ("kinetics", "weight-kinetics", "Kinetics", 1),
    ("shape", "weight-shape", "Shape", 1),
    ("diversity", "weight-diversity", "Chem. diversity", 1),
]


def _load_binding_sites_csv(search_dir):
    """Load binding_sites.csv from search_dir (empty DataFrame if absent)."""
    path = os.path.join(search_dir, "binding_sites.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


class _BindingSiteRow:
    """Lightweight stand-in exposing the attributes score_binding_sites reads."""
    def __init__(self, row):
        self.site_id = int(row["site_id"])
        self.agfe_min = float(row["agfe_min"])
        self.probe_coverage = float(row["probe_coverage"])
        self.volume = float(row["volume"])
        self.solidity = float(row["solidity"])
        res = row.get("residence", None)
        self.residence = None if res is None or (isinstance(res, float) and not np.isfinite(res)) else float(res)
        fa = row.get("favorable_atomtypes", "")
        self.favorable_atomtypes = [a for a in str(fa).split(",") if a] if pd.notna(fa) else []


def rerank_binding_sites(df, weights):
    """Return a copy of df with combined/rank recomputed via score_binding_sites, sorted by rank."""
    from cosolvkit.analysis.core.scoring import score_binding_sites
    if df is None or df.empty:
        return df
    objs = [_BindingSiteRow(row) for _, row in df.iterrows()]
    score_binding_sites(objs, weights)          # sets .combined and .rank in place
    by_id = {o.site_id: (o.combined, o.rank) for o in objs}
    out = df.copy()
    out["combined"] = out["site_id"].map(lambda s: by_id[int(s)][0])
    out["rank"] = out["site_id"].map(lambda s: by_id[int(s)][1])
    return out.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lightweight Hotspot stand-in (for data loaded from CSV)
# ---------------------------------------------------------------------------

class _SiteLike:
    """Minimal Hotspot replacement constructed from CSV rows."""

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
            df = pd.read_csv(f)
        except Exception as exc:
            logger.warning(f"Could not read {f}: {exc}")
            continue
        # Auxiliary tables such as hotspot_sites_geom_*.csv (per-site geometric
        # descriptors) share the hotspot_sites_ prefix but are not sites tables:
        # they lack the 'cosolvent' column. Skip them so their rows don't pollute
        # the DataFrame with NaN cosolvent/rank/centroid values.
        if "cosolvent" not in df.columns:
            logger.debug(f"Skipping {f}: no 'cosolvent' column (not a hotspot-sites table).")
            continue
        dfs.append(df)

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
    """Interactive Plotly/Dash dashboard for CoSolvKit binding-site visualization.

    Shows a single "Binding Sites" view: the reference protein Cα backbone
    (coloured by RMSF) overlaid with cross-cosolvent binding-site centroids,
    colour-ranked by a re-rankable combined score (six weighted features),
    alongside a sortable/filterable table of binding-site metrics and
    per-site visibility toggles.

    Parameters
    ----------
    out_path : str
        Analysis output directory.  If a ``merged/`` subdirectory exists,
        ``binding_sites.csv``/DX maps are loaded from there; otherwise
        *out_path* itself is searched.
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
                "dash is required for the dashboard.\n"
                "Install with: pip install dash"
            )
        if not _PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")

        self.out_path = os.path.abspath(out_path)
        self.port = port
        self._agfe_cutoff = agfe_cutoff

        # Prefer merged/ subdirectory for maps and binding-site CSVs
        merged = os.path.join(self.out_path, "merged")
        self._map_dir = merged if os.path.isdir(merged) else self.out_path

        # Resolve PDB path
        self._pdb_path = pdb_path or self._find_pdb()

        # Load binding-site data
        self._bs_df = _load_binding_sites_csv(self._map_dir)
        if self._bs_df.empty:
            self._bs_df = _load_binding_sites_csv(self.out_path)

        # Optional per-site pharmacophore breakdown (site_id -> {cosolvent: {atype: agfe}}),
        # used by the drill-down panel (Task 3). Guarded: file may not exist.
        self._pharmacophore: Dict[int, Dict[str, Dict[str, float]]] = self._load_pharmacophore()

        # Parse PDB once at startup (used for the Cα backbone trace)
        if self._pdb_path and os.path.exists(self._pdb_path):
            logger.info(f"Parsing reference PDB: {self._pdb_path}")
            self._model_data = _parse_pdb_for_viewer(self._pdb_path)
        else:
            if self._pdb_path:
                logger.warning(f"Reference PDB not found: {self._pdb_path}")
            else:
                logger.warning("No reference PDB detected — protein viewer will be empty.")
            self._model_data = {"atoms": [], "bonds": []}

        # Surface missing-input problems loudly (a blank dashboard is confusing).
        for _w in self._data_warnings():
            logger.warning(_w)

        self._app = self._create_app()

    def _data_warnings(self) -> List[str]:
        """User-facing warnings about missing inputs (no binding sites / no PDB)."""
        msgs: List[str] = []
        if self._bs_df is None or self._bs_df.empty:
            msgs.append(
                f"No binding_sites.csv found in '{self._map_dir}' (or '{self.out_path}'). "
                "There are no binding sites to show — run the analysis "
                "(identify_binding_sites / MultiReport) to generate binding_sites.csv."
            )
        if not self._model_data.get("atoms"):
            msgs.append(
                "No protein structure (averaged_trajectory.pdb) found near "
                f"'{self.out_path}' — the protein backbone will be empty. "
                "Pass pdb_path=... or place averaged_trajectory.pdb in the results tree."
            )
        return msgs

    def _load_pharmacophore(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Load ``binding_sites_pharmacophore.json`` (if present) into a
        ``{site_id: {cosolvent: {atomtype: agfe}}}`` mapping."""
        for search_dir in (self._map_dir, self.out_path):
            path = os.path.join(search_dir, "binding_sites_pharmacophore.json")
            if os.path.exists(path):
                try:
                    with open(path) as fh:
                        records = json.load(fh)
                    return {
                        int(rec["site_id"]): rec.get("pharmacophore", {})
                        for rec in records
                    }
                except Exception as exc:
                    logger.warning(f"Could not read {path}: {exc}")
                    return {}
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pdb(self) -> Optional[str]:
        """Locate ``averaged_trajectory.pdb`` in common output locations.

        Also searches per-simulation sibling subdirectories: when the dashboard
        is pointed at a ``merged/`` directory, the averaged structure typically
        lives in the individual simulation folders (e.g.
        ``results/<sim>/averaged_trajectory.pdb``), not in ``merged/`` itself.
        Any per-simulation averaged structure works as the reference backbone.
        """
        parent = os.path.dirname(self.out_path)
        candidates = [
            os.path.join(self.out_path, "averaged_trajectory.pdb"),
            os.path.join(parent, "averaged_trajectory.pdb"),
        ]
        # child subdirs of out_path, then sibling subdirs (parent's children)
        candidates += sorted(glob(os.path.join(self.out_path, "*/averaged_trajectory.pdb")))
        candidates += sorted(glob(os.path.join(parent, "*/averaged_trajectory.pdb")))
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # Protein + Binding-sites Plotly figure
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

    def _build_binding_sites_figure(
        self,
        df: pd.DataFrame,
        visible_ids: set,
    ) -> "go.Figure":
        """Plotly figure: protein Cα backbone + binding-site centroid markers.

        Each binding site is its own trace so it can be hidden independently
        via the ``visible`` flag driven by the checklist. Marker colour maps
        rank (1 = best) onto a sequential "hot" colourscale; marker size is
        proportional to pocket volume.
        """
        traces = self._build_ca_traces()

        if not df.empty:
            n = len(df)
            for _, row in df.iterrows():
                sid = int(row["site_id"])
                rank = int(row["rank"])
                vol = float(row.get("volume", 1.0))
                msize = int(np.clip(vol ** (1.0 / 3.0) * 1.8, 12, 40))
                # rank -> color matching the "Best -> Worst" legend gradient:
                # rank 1 (frac 0) = deep red (#b40426), rank N (frac 1) = pale (#ffffcc).
                frac = 0.0 if n <= 1 else (rank - 1) / (n - 1)
                traces.append(go.Scatter3d(
                    x=[row["centroid_x"]], y=[row["centroid_y"]], z=[row["centroid_z"]],
                    mode="markers+text", name=f"Site {rank}", text=[f"{rank}"],
                    textposition="top center", textfont=dict(color="white", size=11),
                    marker=dict(size=msize, color=[frac],
                                colorscale=[[0.0, "#b40426"], [0.5, "#ffcc66"], [1.0, "#ffffcc"]],
                                cmin=0.0, cmax=1.0, opacity=0.85,
                                line=dict(width=1, color="white")),
                    visible=(sid in visible_ids),
                    hovertemplate=(
                        f"<b>Binding site {rank}</b><br>"
                        f"cosolvents: {row['cosolvents']}<br>"
                        f"combined: {float(row['combined']):.3f}<br>"
                        f"AGFE min: {float(row['agfe_min']):.3f} kcal/mol<br>"
                        f"volume: {vol:.1f} A^3<br>probes: {int(row['n_cosolvents'])}"
                        "<extra></extra>"),
                ))

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

    def _get_bs_table_columns(self) -> list:
        desired = [
            "site_id", "rank", "combined", "cosolvents", "n_cosolvents",
            "probe_coverage", "agfe_min", "volume", "solidity",
            "residence", "n_chemotypes",
        ]
        if self._bs_df.empty:
            return [{"name": c, "id": c} for c in desired]
        available = [c for c in desired if c in self._bs_df.columns]
        return [{"name": c, "id": c} for c in available]

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _create_app(self) -> "dash.Dash":
        app = dash.Dash(__name__, title="CoSolvKit Binding-Site Dashboard")

        label_style = {"fontWeight": "bold", "fontSize": "0.82em", "marginBottom": "4px"}
        btn_style = {
            "fontSize": "0.72em", "padding": "2px 10px", "marginRight": "6px",
            "border": "1px solid #cbd5e0", "borderRadius": "4px",
            "backgroundColor": "#edf2f7", "cursor": "pointer",
        }

        def _weight_slider(wid, label, default):
            return html.Div([
                html.Div(f"{label}: {default:+d}", id=f"{wid}-label", style=label_style),
                dcc.Slider(id=wid, min=-5, max=5, step=1, value=default,
                           marks={-5: "-5", 0: "0", 5: "+5"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"width": "150px"})

        weight_sliders = [
            _weight_slider(wid, label, default)
            for _, wid, label, default in _WEIGHT_SPECS
        ]

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f3f7", "minHeight": "100vh"},
            children=[

                # ---- Header ----
                html.Div(
                    style={"backgroundColor": "#1a3a5c", "color": "white", "padding": "12px 24px"},
                    children=[
                        html.H1("CoSolvKit Binding-Site Dashboard", style={"margin": "0", "fontSize": "1.5em"}),
                        html.P(f"Results: {self.out_path}",
                               style={"margin": "4px 0 0", "fontSize": "0.8em", "opacity": "0.75"}),
                    ],
                ),

                # ---- Data-warning banner (empty binding sites / missing PDB) ----
                html.Div(
                    id="data-warning-banner",
                    style=({
                        "backgroundColor": "#fff3cd", "color": "#7a5b00",
                        "borderBottom": "1px solid #ffe08a", "padding": "10px 24px",
                        "fontSize": "0.85em",
                    } if self._data_warnings() else {"display": "none"}),
                    children=[html.Div("⚠ " + w, style={"margin": "2px 0"})
                              for w in self._data_warnings()],
                ),

                # ---- Controls bar ----
                html.Div(
                    style={
                        "backgroundColor": "white", "padding": "10px 24px",
                        "display": "flex", "alignItems": "center", "gap": "24px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "flexWrap": "wrap",
                    },
                    children=weight_sliders + [
                        html.Div([
                            html.Div(id="topn-label", style=label_style, children="Top N sites: 3"),
                            html.Div(style={"width": "150px"}, children=[
                                dcc.Slider(id="topn-slider", min=1, max=10, step=1, value=3,
                                           marks={i: str(i) for i in range(1, 11)},
                                           tooltip={"placement": "bottom", "always_visible": False}),
                            ]),
                        ]),
                        html.Div([
                            html.Div("Rank", style=label_style),
                            html.Div(style={
                                "background": "linear-gradient(to right, #b40426, #ffcc66, #ffffcc)",
                                "width": "110px", "height": "10px", "borderRadius": "4px",
                            }),
                            html.Div(style={"display": "flex", "justifyContent": "space-between",
                                            "width": "110px", "fontSize": "0.72em", "color": "#555"},
                                     children=[html.Span("Best"), html.Span("Worst")]),
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
                                    value="binding_sites",
                                    style={"fontSize": "0.88em"},
                                    children=[

                                        # ── Binding Sites (Plotly): protein + ranked binding sites ──
                                        dcc.Tab(
                                            label="Binding Sites",
                                            value="binding_sites",
                                            children=[
                                                dcc.Graph(
                                                    id="binding-sites-graph",
                                                    style={"height": "calc(100vh - 220px)"},
                                                    config={"displayModeBar": True},
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

                                # Per-binding-site visibility checkboxes
                                html.Div([
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center",
                                               "justifyContent": "space-between", "marginBottom": "6px"},
                                        children=[
                                            html.H3("Binding-Site Visibility",
                                                    style={"margin": "0", "fontSize": "0.92em", "color": "#1a3a5c"}),
                                            html.Div([
                                                html.Button("All", id="check-all-btn", n_clicks=0, style=btn_style),
                                                html.Button("None", id="check-none-btn", n_clicks=0, style=btn_style),
                                            ]),
                                        ],
                                    ),
                                    dcc.Checklist(
                                        id="bs-checklist",
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
                                html.H3("Binding Sites",
                                        style={"margin": "0 0 4px", "fontSize": "0.92em", "color": "#1a3a5c"}),
                                html.P(id="table-summary",
                                       style={"fontSize": "0.78em", "color": "#666", "margin": "0 0 8px"}),
                                dash_table.DataTable(
                                    id="bs-table",
                                    columns=self._get_bs_table_columns(),
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

                                html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

                                # Drill-down detail panel (populated in Task 3)
                                html.Div(id="bs-detail"),
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

    def _ranked_topn(self, weight_values, top_n):
        """Map the six slider values (in ``_WEIGHT_SPECS`` order) to a weights
        dict, re-rank ``self._bs_df`` via ``rerank_binding_sites``, and return
        the top ``top_n`` rows (or an empty/None-ish frame if there's no data).
        """
        weights = {
            key: float(v)
            for (key, _wid, _label, _default), v in zip(_WEIGHT_SPECS, weight_values)
        }
        ranked = rerank_binding_sites(self._bs_df, weights)
        if ranked is None or ranked.empty:
            return ranked
        return ranked.head(int(top_n))

    def _register_callbacks(self, app: "dash.Dash"):
        # ── Slider labels ─────────────────────────────────────────────────────
        @app.callback(Output("topn-label", "children"), Input("topn-slider", "value"))
        def update_topn_label(value):
            return f"Top N sites: {value}"

        for _key, wid, label, _default in _WEIGHT_SPECS:
            def _make_update_weight_label(label=label):
                def update_weight_label(value):
                    return f"{label}: {value:+d}"
                return update_weight_label

            app.callback(
                Output(f"{wid}-label", "children"), Input(wid, "value")
            )(_make_update_weight_label())

        # ── Live re-rank → figure / table / checklist ───────────────────────────
        _weight_inputs = [Input(wid, "value") for _key, wid, _label, _default in _WEIGHT_SPECS]

        @app.callback(
            [Output("bs-checklist", "options"), Output("bs-checklist", "value")],
            _weight_inputs + [
                Input("topn-slider", "value"),
                Input("check-all-btn", "n_clicks"),
                Input("check-none-btn", "n_clicks"),
            ],
        )
        def _update_checklist(*args):
            weight_values, top_n = args[:6], args[6]
            top = self._ranked_topn(weight_values, top_n)
            options, values = [], []
            if top is not None and not top.empty:
                for _, r in top.iterrows():
                    sid = int(r["site_id"])
                    options.append({
                        "label": (f"Site {int(r['rank'])}  [{r['cosolvents']}]  "
                                  f"score={float(r['combined']):.2f}"),
                        "value": sid,
                    })
                    values.append(sid)
            if ctx.triggered_id == "check-none-btn":
                values = []
            return options, values

        @app.callback(
            Output("binding-sites-graph", "figure"),
            _weight_inputs + [Input("topn-slider", "value"), Input("bs-checklist", "value")],
        )
        def _update_figure(*args):
            weight_values, top_n, visible = args[:6], args[6], set(args[7] or [])
            top = self._ranked_topn(weight_values, top_n)
            return self._build_binding_sites_figure(
                top if top is not None else pd.DataFrame(), visible
            )

        @app.callback(
            [Output("bs-table", "data"), Output("table-summary", "children")],
            _weight_inputs + [Input("topn-slider", "value")],
        )
        def _update_table(*args):
            weight_values, top_n = args[:6], args[6]
            top = self._ranked_topn(weight_values, top_n)
            if top is None or top.empty:
                return [], "0 binding site(s) shown"
            cols = ["combined", "probe_coverage", "agfe_min", "volume", "solidity", "residence"]
            t = top.copy()
            for c in cols:
                if c in t.columns:
                    t[c] = t[c].round(3)
            return t.to_dict("records"), f"{len(top)} binding site(s) shown"

        @app.callback(
            Output("bs-detail", "children"),
            Input("bs-table", "active_cell"),
            Input("bs-table", "derived_virtual_data"),
        )
        def _detail(active_cell, rows):
            if not active_cell or not rows:
                return "Select a row to see member hotspots + pharmacophore."
            idx = active_cell.get("row")
            if idx is None or idx >= len(rows):
                return "Select a row to see member hotspots + pharmacophore."
            row = rows[idx]
            sid = int(row["site_id"]) if row.get("site_id") is not None else None
            pharm = self._pharmacophore.get(sid, {}) if sid is not None else {}
            lines = [
                html.Div(f"Binding site {row.get('rank')} — cosolvents {row.get('cosolvents')}"),
                html.Div(f"member hotspots: {row.get('member_hotspot_ids', '')}"),
            ]
            for cos, atypes in pharm.items():
                lines.append(html.Div(
                    f"{cos}: " + ", ".join(f"{a} {float(v):.2f}" for a, v in atypes.items())
                ))
            return lines

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
        try:
            self._app.run(host=host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            # Ctrl-C: terminate cleanly so the port is released immediately.
            print(f"\nDashboard stopped; port {self.port} released.")


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
    print(f"  CoSolvKit Binding-Site Dashboard")
    print(f"{sep}")
    print(f"  Server   : {fqdn}  (host flag: {host})")
    print(f"  Port     : {port}")

    if is_ssh:
        direct = f"ssh -fNL {port}:localhost:{port} {user}@{fqdn}"
        proxyjump = (
            f"ssh -fNL {port}:localhost:{port} "
            f"-J {user}@<login-node> {user}@{hostname}"
        )
        print(f"\n  Detected SSH session — the browser must run on your local")
        print(f"  machine.  Open a NEW local terminal and run ONE of:\n")
        print(f"  • If this host is directly reachable from your machine:")
        print(f"      {direct}\n")
        print(f"  • If this is an HPC compute node (reachable only via a login")
        print(f"    node — the usual case), jump through the login node:")
        print(f"      {proxyjump}")
        print(f"    (replace <login-node> with the cluster address you ssh into)\n")
        print(f"  Then open your browser at:  {local_url}")
    else:
        print(f"\n  Open your browser at:  {local_url}")

    print(f"\n  Stop with Ctrl-C  (do NOT use Ctrl-Z — that only suspends the")
    print(f"  server and leaves port {port} busy, so the next launch fails with")
    print(f"  'Address already in use'). To free a stuck port: fuser -k {port}/tcp")
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
