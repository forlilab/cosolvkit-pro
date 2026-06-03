#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Property computation over candidate binding pockets.
# Decoupled from hotspot detection — all functions here accept
# BindingSite objects and populate them in-place.
#

import os
import logging
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from . import hotspot_visualization as viz


# ---------------------------------------------------------------------------
# regionprops constants and helpers
# ---------------------------------------------------------------------------

# Standard skimage regionprops properties safe for 3D volumetric arrays that
# produce tabular (scalar or fixed-size array) output suitable for CSV/JSON.
#
# Excluded — raise NotImplementedError on 3D inputs:
#   eccentricity, orientation, perimeter, perimeter_crofton,
#   moments_hu, moments_weighted_hu
#
# Excluded — return variable-size per-region arrays or Python objects that
# break tabular export (pass them explicitly via regionprops_properties to opt in):
#   image, image_convex, image_filled, image_intensity, coords, coords_scaled, slice
#
# Note: 3D moments tensors are 4×4×4 = 64 columns each; the six moment
# properties in this list expand to ~384 columns in the output.
REGIONPROPS_ALL = [
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
    "centroid",
    "centroid_local",
    "centroid_weighted",
    "centroid_weighted_local",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
    "moments",
    "moments_central",
    "moments_normalized",
    "moments_weighted",
    "moments_weighted_central",
    "moments_weighted_normalized",
    "solidity",
]


def _serialize_regionprop_value(val):
    """Convert a regionprops_table cell to a JSON-safe Python scalar or list."""
    if isinstance(val, tuple) and any(isinstance(x, slice) for x in val):
        return [[x.start, x.stop, x.step] for x in val]
    if isinstance(val, slice):
        return [val.start, val.stop, val.step]
    if np.ndim(val) == 0:
        if isinstance(val, (np.integer, int)):
            return int(val)
        return float(val)
    return [float(v) for v in np.ravel(val)]


# ---------------------------------------------------------------------------
# Survival probability helpers
# ---------------------------------------------------------------------------

def _is_xyz(group):
    """Return True if group encodes an XYZ point (exactly 3 float-like values)."""
    return len(group) == 3 and all(isinstance(v, float) for v in group)


def _build_selection(cosolvent_name, group, radius):
    """Build an MDAnalysis selection string and a human-readable label for a zone."""
    if isinstance(group, int):
        return (
            f"resname {cosolvent_name} and sphzone {radius} resid {group}",
            str(group),
        )
    group = list(group)
    if _is_xyz(group):
        x, y, z = group
        return (
            f"resname {cosolvent_name} and point {x} {y} {z} {radius}",
            f"({x:.2f}, {y:.2f}, {z:.2f})",
        )
    resids = " or ".join(f"resid {r}" for r in group)
    return (
        f"resname {cosolvent_name} and sphzone {radius} ({resids})",
        " ".join(str(r) for r in group),
    )


# ---------------------------------------------------------------------------
# Curve-fitting helpers
# ---------------------------------------------------------------------------

def _single_exp(t, tau):
    return np.exp(-t / tau)


def _bi_exp(t, A, tau1, tau2):
    return A * np.exp(-t / tau1) + (1.0 - A) * np.exp(-t / tau2)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else 0.0


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Maps short weight-key aliases to the actual BindingSite attribute name.
_CORE_ATTR_ALIASES = {
    "favorability": "favorability_score",
    "diversity": "diversity_score",
    "volume": "volume_score",
    "favorability_score": "favorability_score",
    "diversity_score": "diversity_score",
    "volume_score": "volume_score",
}


def _get_site_value(site, key):
    """Retrieve a scoring component from a BindingSite.

    Checks core attribute aliases first, then site.properties.
    Returns None if the key is unknown or the value is None.
    """
    attr = _CORE_ATTR_ALIASES.get(key)
    if attr is not None:
        return getattr(site, attr, None)
    return site.properties.get(key)


def compute_composite_score(sites, score_weights):
    """Recompute composite scores for a list of BindingSite objects.

    Supports any combination of core field weights (``favorability``,
    ``diversity``, ``volume``, or their ``_score`` variants) and
    ``site.properties`` keys (``sp_mrt``, ``sp_tau_single``,
    ``geom_solidity``, etc.).

    Each component is min-max normalised across *sites* (higher = better).
    Sites with ``None`` or non-finite values for a component score 0 on that
    component.  Components where every site has a missing value are dropped
    and the remaining weights are re-normalised to sum to 1.

    Updates ``site.composite_score`` and ``site.rank`` in-place and
    re-ranks sites descending.

    Note
    ----
    Unlike the initial composite computed in ``HotspotDetector.detect()``,
    this function applies full min-max normalisation to *all* components,
    including diversity and volume.  Calling it with only the three core keys
    may therefore produce slightly different composite values than the initial
    detection pass.

    Parameters
    ----------
    sites : list[BindingSite]
    score_weights : dict[str, float]
        Weight keys resolved via ``_get_site_value``.  Need not sum to 1;
        the function normalises internally.
    """
    if not sites or not score_weights:
        return

    # Collect raw values for each weight key
    raw_values = {key: [_get_site_value(s, key) for s in sites]
                  for key in score_weights}

    # Determine per-key min/max over finite values; drop fully-missing keys
    active_keys = []
    key_mins, key_maxs = {}, {}
    for key, vals in raw_values.items():
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        if not finite:
            continue
        active_keys.append(key)
        key_mins[key] = min(finite)
        key_maxs[key] = max(finite)

    if not active_keys:
        return

    # Re-normalise weights over surviving keys
    total_w = sum(score_weights[k] for k in active_keys)
    norm_w = {k: score_weights[k] / total_w for k in active_keys}

    # Compute normalised component scores per site
    for i, site in enumerate(sites):
        composite = 0.0
        for key in active_keys:
            val = raw_values[key][i]
            lo, hi = key_mins[key], key_maxs[key]
            if val is None or not np.isfinite(val):
                component = 0.0
            elif (hi - lo) < 1e-20:
                component = 1.0
            else:
                component = (val - lo) / (hi - lo)
            composite += norm_w[key] * component
        site.composite_score = composite

    # Re-rank descending
    for rank, site in enumerate(sorted(sites, key=lambda s: s.composite_score,
                                       reverse=True), start=1):
        site.rank = rank


# ---------------------------------------------------------------------------
# PocketPropertyCalculator
# ---------------------------------------------------------------------------

class PocketPropertyCalculator:
    """Computes and attaches derived properties to :class:`BindingSite` objects.

    Handles three concerns independently of the hotspot-detection algorithm:

    * **Geometry descriptors** — scikit-image ``regionprops_table`` features
      attached as ``geom_*`` properties.
    * **Survival probability** — waterdynamics SP curves written to CSV/PNG.
    * **SP curve fitting** — kinetic metrics (MRT, half-life, τ constants)
      attached as ``sp_*`` properties.

    Parameters
    ----------
    out_path : str
        Directory for CSV/PNG output files.
    universe : MDAnalysis.Universe or None
        Required for :meth:`run_survival_probability`; may be ``None`` if SP
        is not used.
    gridsize : float
        Voxel size in Angstroms (default 0.5).  Not currently used by any
        method but retained for forward compatibility.
    regionprops_properties : list[str], optional
        Overrides :data:`REGIONPROPS_ALL` for geometry descriptor computation.
    regionprops_extra_properties : iterable of callable, optional
        Custom callables forwarded to ``regionprops_table``'s
        ``extra_properties`` argument.
    """

    def __init__(self, out_path, universe, gridsize=0.5,
                 regionprops_properties=None,
                 regionprops_extra_properties=None):
        self.out_path = out_path
        self.universe = universe
        self.gridsize = gridsize
        self.regionprops_properties = regionprops_properties
        self.regionprops_extra_properties = regionprops_extra_properties
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Geometry descriptors
    # ------------------------------------------------------------------

    def compute_regionprops(self, sites, labeled_array, intensity_image,
                            properties=None, extra_properties=None):
        """Compute per-region geometric descriptors and attach them to sites.

        Calls ``skimage.measure.regionprops_table`` on *labeled_array* and
        populates each site in *sites* with ``geom_*`` properties via
        :meth:`BindingSite.add_property`.

        Parameters
        ----------
        sites : list[BindingSite]
            Sites to annotate; each site's ``.site_id`` is used as the label
            key to look up its region in *labeled_array*.
        labeled_array : np.ndarray of int
            3-D labeled array (0 = background, positive integers = cluster ids).
        intensity_image : np.ndarray of float
            Intensity image passed to ``regionprops_table`` for weighted
            centroid and intensity properties (typically
            ``clip(-agfe_array, 0, None)``).
        properties : list[str], optional
            skimage property names to compute.  Overrides
            ``self.regionprops_properties``; ``None`` resolves to
            :data:`REGIONPROPS_ALL`.
        extra_properties : iterable of callable, optional
            Custom callables forwarded to ``regionprops_table``'s
            ``extra_properties`` argument.  Overrides
            ``self.regionprops_extra_properties``.
        """
        from skimage.measure import regionprops_table

        if properties is None:
            properties = self.regionprops_properties
        if properties is None:
            properties = REGIONPROPS_ALL

        try:
            from skimage.measure._regionprops import PROP_VALS
            safe = [p for p in properties if p in PROP_VALS]
            skipped = [p for p in properties if p not in PROP_VALS]
            if skipped:
                self.logger.debug(
                    "regionprops: skipped (not in this skimage version): %s", skipped
                )
            properties = safe
        except ImportError:
            pass

        requested = ["label"] + [p for p in properties if p != "label"]

        if extra_properties is None:
            extra_properties = self.regionprops_extra_properties

        props = regionprops_table(
            labeled_array,
            intensity_image=intensity_image,
            properties=requested,
            extra_properties=extra_properties or None,
        )

        n = len(props["label"])
        region_props = {}
        for i in range(n):
            lbl = int(props["label"][i])
            entry = {}
            for key, arr in props.items():
                if key == "label":
                    continue
                entry[f"geom_{key}"] = _serialize_regionprop_value(arr[i])
            region_props[lbl] = entry

        for site in sites:
            for k, v in region_props.get(site.site_id, {}).items():
                site.add_property(k, v)

    # ------------------------------------------------------------------
    # Survival probability
    # ------------------------------------------------------------------

    def run_survival_probability(self, cosolvent_names, candidate_zones,
                                 radius=6.0, max_tau=100, intermittency=2):
        """Compute the survival probability of cosolvents inside spherical zones.

        Each zone can be defined as a group of residue IDs or as an explicit
        XYZ coordinate; the two forms can be mixed.

        **Zone formats** — each element of ``candidate_zones`` is one zone:

        * ``[resid1, resid2, ...]`` — sphere centred at the COM of listed residues.
        * ``[x, y, z]`` (3 floats) — sphere centred at the explicit Angstrom point.
        * A bare ``int`` — treated as ``[resid]``.

        Results are saved as ``survival_probability_{cosolvent}.csv`` and
        ``survival_probability_{cosolvent}.png`` under ``self.out_path``.

        Parameters
        ----------
        cosolvent_names : list[str]
            Cosolvent residue names to analyse.
        candidate_zones : list
            Zones to analyse (see format description above).
        radius : float
            Sphere radius in Angstroms (default 6.0).
        max_tau : int
            Maximum lag time for the survival-probability calculation (default 100).
        intermittency : int
            Intermittency for ``waterdynamics.SurvivalProbability`` (default 2).
        """
        try:
            from waterdynamics import SurvivalProbability as SP
        except ImportError:
            raise ImportError(
                "waterdynamics package is required for survival probability analysis. "
                "Please install it."
            )

        if candidate_zones is None:
            raise ValueError("candidate_zones must be provided.")

        for cosolvent_name in cosolvent_names:
            data = []

            for zone_idx, zone in enumerate(candidate_zones):
                select, label_str = _build_selection(cosolvent_name, zone, radius)
                self.logger.info(
                    f"Zone {zone_idx} [{label_str}] — cosolvent {cosolvent_name}"
                )

                sp = SP(self.universe, select, verbose=True)
                sp.run(tau_max=max_tau, residues=False, intermittency=intermittency)

                for tau, sp_value in zip(sp.tau_timeseries, sp.sp_timeseries):
                    data.append({
                        "Group": zone_idx,
                        "Zone": label_str,
                        "Time": tau,
                        "SP": sp_value,
                        "Cosolvent": cosolvent_name,
                    })

            df_sp = pd.DataFrame(data)
            df_sp.to_csv(
                os.path.join(self.out_path, f"survival_probability_{cosolvent_name}.csv"),
                index=False,
            )
            viz.plot_sp_raw(cosolvent_name, df_sp, self.out_path)

    # ------------------------------------------------------------------
    # SP curve fitting
    # ------------------------------------------------------------------

    def fit_survival_probability(self, results, zone_to_site_rank=None):
        """Fit SP decay curves and store kinetic metrics in each BindingSite.

        Reads the ``survival_probability_{cosolvent}.csv`` files written by
        :meth:`run_survival_probability`, fits three decay models to each
        zone's curve, and stores the derived metrics in
        ``BindingSite.properties`` via :meth:`BindingSite.add_property`.

        **Stored properties** (prefixed ``sp_``):

        * ``sp_mrt``            — mean residence time (trapezoid integral of SP)
        * ``sp_half_life``      — time at SP = 0.5
        * ``sp_plateau``        — mean SP over the last 10 % of timepoints
        * ``sp_tau_single``     — single-exponential time constant τ
        * ``sp_r2_single``      — R² of single-exp fit
        * ``sp_amplitude_fast`` — fraction in the fast population (bi-exp A)
        * ``sp_tau_fast``       — fast time constant τ₁ (bi-exp)
        * ``sp_tau_slow``       — slow time constant τ₂ (bi-exp)
        * ``sp_r2_biexp``       — R² of bi-exponential fit

        Parameters
        ----------
        results : dict[str, list[BindingSite]]
            Output of :meth:`HotspotDetector.detect_all`.
        zone_to_site_rank : dict[int, int], optional
            Maps zone index (``Group`` column in CSV) to site rank.
            If ``None``, zone 0 → rank 1, zone 1 → rank 2, etc.
        """
        for cosolvent, sites in results.items():
            csv_path = os.path.join(
                self.out_path, f"survival_probability_{cosolvent}.csv"
            )
            if not os.path.exists(csv_path):
                self.logger.warning(
                    f"No SP CSV found for '{cosolvent}': {csv_path}. "
                    "Run run_survival_probability() first."
                )
                continue

            df = pd.read_csv(csv_path)
            site_by_rank = {site.rank: site for site in sites}

            for zone_idx, group_df in df.groupby("Group"):
                rank = (
                    zone_to_site_rank.get(int(zone_idx))
                    if zone_to_site_rank is not None
                    else int(zone_idx) + 1
                )
                site = site_by_rank.get(rank)
                if site is None:
                    self.logger.debug(
                        f"Zone {zone_idx} → rank {rank}: no matching site, skipping."
                    )
                    continue

                tau_arr = group_df["Time"].values.astype(float)
                sp_arr = group_df["SP"].values.astype(float)

                if len(tau_arr) < 3:
                    continue

                props = {}

                # MRT — trapezoidal integral
                props["sp_mrt"] = round(float(np.trapz(sp_arr, tau_arr)), 4)

                # Half-life — interpolate SP = 0.5
                try:
                    f_interp = interp1d(
                        sp_arr[::-1], tau_arr[::-1],
                        bounds_error=False, fill_value=np.nan,
                    )
                    hl = float(f_interp(0.5))
                    props["sp_half_life"] = round(hl, 4) if np.isfinite(hl) else None
                except Exception:
                    props["sp_half_life"] = None

                # Late-time plateau (mean of last 10 % of timepoints)
                n_tail = max(1, len(sp_arr) // 10)
                props["sp_plateau"] = round(float(np.mean(sp_arr[-n_tail:])), 4)

                # Single-exponential fit
                try:
                    p0 = [max(props["sp_mrt"], 1.0)]
                    popt, _ = curve_fit(
                        _single_exp, tau_arr, sp_arr,
                        p0=p0, bounds=(0, np.inf), maxfev=5000,
                    )
                    props["sp_tau_single"] = round(float(popt[0]), 4)
                    props["sp_r2_single"] = round(
                        _r2(sp_arr, _single_exp(tau_arr, *popt)), 4
                    )
                except Exception as exc:
                    self.logger.debug(f"Single-exp fit failed (zone {zone_idx}): {exc}")

                # Bi-exponential fit (requires at least 6 points)
                if len(tau_arr) >= 6:
                    try:
                        mrt = props["sp_mrt"]
                        p0 = [0.5, max(mrt * 0.1, 1.0), max(mrt, 1.0)]
                        popt, _ = curve_fit(
                            _bi_exp, tau_arr, sp_arr, p0=p0,
                            bounds=([0, 0, 0], [1, np.inf, np.inf]),
                            maxfev=10000,
                        )
                        A, tau1, tau2 = float(popt[0]), float(popt[1]), float(popt[2])
                        if tau1 > tau2:  # enforce fast < slow convention
                            A, tau1, tau2 = 1.0 - A, tau2, tau1
                        props["sp_amplitude_fast"] = round(A, 4)
                        props["sp_tau_fast"] = round(tau1, 4)
                        props["sp_tau_slow"] = round(tau2, 4)
                        props["sp_r2_biexp"] = round(
                            _r2(sp_arr, _bi_exp(tau_arr, *popt)), 4
                        )
                    except Exception as exc:
                        self.logger.debug(
                            f"Bi-exp fit failed (zone {zone_idx}): {exc}"
                        )

                for k, v in props.items():
                    site.add_property(k, v)

                self.logger.info(
                    f"Site rank {rank} ({cosolvent}): "
                    f"MRT={props['sp_mrt']:.2f}, "
                    f"plateau={props['sp_plateau']:.3f}, "
                    f"τ_single={props.get('sp_tau_single', 'N/A')}"
                )

            viz.plot_sp_fits(cosolvent, sites, df, self.out_path)
