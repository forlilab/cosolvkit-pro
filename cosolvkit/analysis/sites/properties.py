#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Property computation over candidate binding pockets.
# Moved verbatim from pocket_properties.py (Task 8).
#

import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from cosolvkit.analysis.core.models import PocketResidue


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


def detect_fused_residues(atomgroup):
    """Residues holding more selected atoms than the modal residue, as (resid, n_atoms).

    Symptom of a wrapped topology: the PDB resid field maxes at 9999, so the overflow is
    merged into one pseudo-residue.
    """
    if atomgroup is None or atomgroup.n_atoms == 0:
        return []
    uniq, counts = np.unique(atomgroup.resindices, return_counts=True)
    if len(uniq) < 2:
        return []
    modal = int(np.bincount(counts).argmax())
    residues = atomgroup.universe.residues
    return [(int(residues[ri].resid), int(c))
            for ri, c in zip(uniq, counts) if c > modal]


def warn_if_fused_residues(atomgroup, logger=None, context=""):
    """Log a warning if *atomgroup*'s topology has fused residues. True if any were found."""
    fused = detect_fused_residues(atomgroup)
    if not fused:
        return False
    if logger is not None:
        where = f" for {context}" if context else ""
        logger.warning(
            f"Topology has {len(fused)} over-sized residue(s) holding "
            f"{sum(n for _, n in fused)} atoms{where}: the resid field wrapped (PDB resid "
            "maxes at 9999), fusing distinct molecules into one residue. 'resid'-based "
            "selections are therefore ambiguous, and per-residue quantities are wrong for the "
            "fused residue. Use the prmtop as topology, not the PDB. Density maps and survival "
            "probability are unaffected (they key on positions and on unique atom ids)."
        )
    return True


def _finite_or_none(val):
    """Float, or None if non-finite (not valid JSON). QHull returns convex_area 0 for tiny
    blobs it cannot hull, making solidity inf."""
    f = float(val)
    return f if np.isfinite(f) else None


def _serialize_regionprop_value(val):
    """Convert a regionprops_table cell to a JSON-safe Python scalar or list."""
    if isinstance(val, tuple) and any(isinstance(x, slice) for x in val):
        return [[x.start, x.stop, x.step] for x in val]
    if isinstance(val, slice):
        return [val.start, val.stop, val.step]
    if np.ndim(val) == 0:
        if isinstance(val, (np.integer, int)):
            return int(val)
        return _finite_or_none(val)
    return [_finite_or_none(v) for v in np.ravel(val)]


# ---------------------------------------------------------------------------
# Survival probability helpers
# ---------------------------------------------------------------------------

def _is_xyz(group):
    """Return True if group encodes an XYZ point (exactly 3 float-like values)."""
    return len(group) == 3 and all(isinstance(v, float) for v in group)


def _zone_is_resid_based(zone):
    """True if *zone* selects by resid rather than by an explicit XYZ point."""
    if isinstance(zone, int):
        return True
    return not _is_xyz(list(zone))


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
# PLM embedding injection
# ---------------------------------------------------------------------------

def set_residue_embeddings(site, embeddings: Dict[int, Any], model_name: str = "") -> None:
    """Attach protein language model embeddings to pocket residues by resid.

    Call after :meth:`PocketPropertyCalculator.find_pocket_residues` has
    populated ``site.pocket_residues``.

    Parameters
    ----------
    site : Hotspot
        Site whose ``pocket_residues`` list to annotate.
    embeddings : dict[int, array-like]
        Mapping ``{resid: embedding_vector}``.  Resids that do not match any
        pocket residue are silently skipped with a warning.
    model_name : str
        Recorded as ``PocketResidue.embedding_model`` on each annotated residue.
    """
    logger = logging.getLogger(__name__)
    resid_to_pr = {pr.resid: pr for pr in site.pocket_residues}
    for resid, vec in embeddings.items():
        pr = resid_to_pr.get(int(resid))
        if pr is not None:
            pr.embedding = np.asarray(vec, dtype=np.float32)
            pr.embedding_model = model_name or None
        else:
            logger.warning(
                "set_residue_embeddings: resid %d not found in site.pocket_residues", resid
            )


# ---------------------------------------------------------------------------
# PocketPropertyCalculator
# ---------------------------------------------------------------------------

# vdW radii (A) for sizing an adaptive survival-probability zone from the probe itself.
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80, "P": 1.80,
              "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20}


def _element_of(atom):
    """Element symbol for an MDAnalysis atom, falling back to the atom name."""
    for attr in ("element", "type"):
        v = getattr(atom, attr, "")
        if isinstance(v, str) and v.strip():
            e = v.strip().upper()
            if e in _VDW_RADII:
                return e
            if e[:1] in _VDW_RADII:
                return e[:1]
    name = str(getattr(atom, "name", "")).strip().upper()
    return name[:1] if name[:1] in _VDW_RADII else "C"


class PocketPropertyCalculator:
    """Computes and attaches derived properties to :class:`Hotspot` objects.

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
        :meth:`Hotspot.add_property`.

        Parameters
        ----------
        sites : list[Hotspot]
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

        try:
            props = regionprops_table(
                labeled_array,
                intensity_image=intensity_image,
                properties=requested,
                extra_properties=extra_properties or None,
            )
        except ValueError:
            # feret_diameter_max calls marching_cubes on the convex hull image,
            # which fails for degenerate/tiny clusters where qhull returns an
            # empty image. Retry without it.
            fallback = [p for p in requested if p != "feret_diameter_max"]
            self.logger.warning(
                "regionprops_table failed (feret_diameter_max on degenerate cluster); "
                "retrying without feret_diameter_max."
            )
            props = regionprops_table(
                labeled_array,
                intensity_image=intensity_image,
                properties=fallback,
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

    @staticmethod
    def probe_zone_radius(universe, cosolvent_name, tolerance=1.7):
        """Survival-probability zone radius scaled to the PROBE: ``Rg + tolerance``.

        ``Rg`` is the probe's heavy-atom radius of gyration and *tolerance* is a contact
        buffer (default 1.7 A, a heavy-atom vdW radius).

        Scaling on the probe rather than on the hotspot is deliberate. Hotspot volume is
        "voxels above the AGFE cutoff, as grouped by the clustering", so deriving the
        radius from it would make the kinetics inherit both of those choices. Probe size is
        a fixed molecular property. It also keeps the radius constant within a probe, so
        residence stays comparable BETWEEN sites (a larger zone holds a molecule longer for
        purely geometric reasons), and puts the probes on a common physical footing.

        This replaces an earlier ``half the largest heavy-atom separation + mean vdW``
        footprint, which correlated with Rg at Pearson 0.95 while needing an element table
        and name-guessing to produce a term that varied by only 0.11 A across an 18-probe
        panel. For equivalent magnitudes to that formula use ``tolerance ~ 2.9``.

        Returns ``None`` when the cosolvent is absent from *universe*.
        """
        sel = universe.select_atoms(f"resname {cosolvent_name}")
        if sel.n_atoms == 0:
            return None
        atoms = sel.residues[0].atoms
        heavy = atoms[np.array([_element_of(a) for a in atoms]) != "H"]
        if heavy.n_atoms == 0:
            return None
        # Unweighted (number-average) Rg: the zone is a geometric criterion, so atom
        # identity should not tilt it. Differs from the mass-weighted value by <0.07 A
        # on this panel.
        pos = heavy.positions
        rg = float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))
        return rg + float(tolerance)

    def run_survival_probability(self, cosolvent_names, candidate_zones,
                                 radius=6.0, max_tau=100, intermittency=2,
                                 universes=None, radius_tolerance=1.7):
        """Compute the survival probability of cosolvents inside spherical zones.

        Each zone can be defined as a group of residue IDs or as an explicit
        XYZ coordinate; the two forms can be mixed.

        **Zone formats** — each element of ``candidate_zones`` is one zone:

        * ``[resid1, resid2, ...]`` — sphere centred at the COM of listed residues.
        * ``[x, y, z]`` (3 floats) — sphere centred at the explicit Angstrom point.
        * A bare ``int`` — treated as ``[resid]``.

        Results are saved as ``survival_probability_{cosolvent}.csv`` (the
        replica-averaged curve, which :meth:`fit_survival_probability` consumes) and
        ``survival_probability_{cosolvent}.png`` under ``self.out_path``. With more than
        one replica, ``survival_probability_{cosolvent}_per_replica.csv`` also keeps the
        individual curves so the spread is recoverable.

        Parameters
        ----------
        cosolvent_names : list[str]
            Cosolvent residue names to analyse.
        candidate_zones : list
            Zones to analyse (see format description above).
        radius : float or {"adaptive"}
            Sphere radius in Angstroms (default 6.0), or ``"adaptive"`` to scale it per
            probe via :meth:`probe_zone_radius` (footprint + *radius_tolerance*). A single
            fixed radius is a poor compromise on a mixed panel: 6.0 A is ~25x the median
            hotspot volume and measures the neighbourhood, while 3.5 A is small enough that
            replica agreement collapses to rho ~ 0.06.
        radius_tolerance : float
            Contact buffer added to the probe's radius of gyration when
            ``radius="adaptive"`` (default 1.7 A, a heavy-atom vdW radius).
        max_tau : int
            Maximum lag time for the survival-probability calculation (default 100).
        intermittency : int
            Intermittency for ``waterdynamics.SurvivalProbability`` (default 2).
        universes : list, optional
            Replica universes to average over. Defaults to ``[self.universe]``.

            Survival probability is a *dynamical* quantity, so replicas must be run
            SEPARATELY and their curves averaged — they cannot be concatenated into one
            trajectory. A join between two independent replicas is a discontinuity: a
            probe present in the last frame of one and absent from the first frame of the
            next registers as a departure that never happened, and the resids do not even
            refer to the same molecule across the join. With ``max_tau`` lag frames every
            join corrupts that many lag windows, biasing residence times downward.
            Averaging independent replicas is the correct ensemble average and is the only
            version that yields an uncertainty.
        """
        import cosolvkit.analysis.hotspot_visualization as viz
        try:
            from waterdynamics import SurvivalProbability as SP
        except ImportError:
            raise ImportError(
                "waterdynamics package is required for survival probability analysis. "
                "Please install it."
            )

        if candidate_zones is None:
            raise ValueError("candidate_zones must be provided.")

        replicas = list(universes) if universes else [self.universe]

        for cosolvent_name in cosolvent_names:
            data = []

            zone_radius = radius
            if isinstance(radius, str):
                if radius.strip().lower() != "adaptive":
                    raise ValueError(
                        f"radius must be a number or 'adaptive', got {radius!r}")
                zone_radius = self.probe_zone_radius(
                    replicas[0], cosolvent_name, tolerance=radius_tolerance)
                if zone_radius is None:
                    raise ValueError(
                        f"cannot size an adaptive zone: cosolvent {cosolvent_name!r} "
                        "not found in the trajectory topology")
                self.logger.info(
                    f"Adaptive SP zone radius for {cosolvent_name}: "
                    f"{zone_radius:.2f} A (probe Rg + {radius_tolerance:.2f})"
                )

            # 'resid N' is ambiguous on a wrapped topology, so only resid-style zones care.
            if any(_zone_is_resid_based(z) for z in candidate_zones):
                for universe in replicas:
                    warn_if_fused_residues(
                        universe.atoms, logger=self.logger,
                        context=f"resid-based SP zones of {cosolvent_name}")

            for rep_idx, universe in enumerate(replicas):
                for zone_idx, zone in enumerate(candidate_zones):
                    select, label_str = _build_selection(cosolvent_name, zone,
                                                        zone_radius)
                    self.logger.info(
                        f"Zone {zone_idx} [{label_str}] — cosolvent {cosolvent_name} "
                        f"— replica {rep_idx + 1}/{len(replicas)}"
                    )

                    sp = SP(universe, select, verbose=True)
                    sp.run(tau_max=max_tau, residues=False,
                           intermittency=intermittency)

                    for tau, sp_value in zip(sp.tau_timeseries, sp.sp_timeseries):
                        data.append({
                            "Group": zone_idx,
                            "Zone": label_str,
                            "Time": tau,
                            "SP": sp_value,
                            "Cosolvent": cosolvent_name,
                            "Replica": rep_idx,
                        })

            df_raw = pd.DataFrame(data)
            out_csv = os.path.join(
                self.out_path, f"survival_probability_{cosolvent_name}.csv")

            if len(replicas) > 1:
                df_raw.to_csv(
                    os.path.join(
                        self.out_path,
                        f"survival_probability_{cosolvent_name}_per_replica.csv"),
                    index=False,
                )
                # Average each zone's curve over replicas. Zones unoccupied in a replica
                # yield NaN there and are simply dropped from that point's mean, so a
                # missing replica lowers n rather than poisoning the average.
                df_sp = (
                    df_raw.groupby(["Group", "Zone", "Time", "Cosolvent"], as_index=False)
                    .agg(SP=("SP", "mean"), SP_sd=("SP", "std"), n_replicas=("SP", "count"))
                )
            else:
                df_sp = df_raw.drop(columns=["Replica"])

            df_sp.to_csv(out_csv, index=False)
            viz.plot_sp_raw(cosolvent_name, df_sp, self.out_path)

    # ------------------------------------------------------------------
    # SP curve fitting
    # ------------------------------------------------------------------

    def fit_survival_probability(self, results, zone_to_site_rank=None):
        """Fit SP decay curves and store kinetic metrics in each Hotspot.

        Reads the ``survival_probability_{cosolvent}.csv`` files written by
        :meth:`run_survival_probability`, fits three decay models to each
        zone's curve, and stores the derived metrics in
        ``Hotspot.properties`` via :meth:`Hotspot.add_property`.

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
        results : dict[str, list[Hotspot]]
            Output of :meth:`HotspotDetector.detect_all`.
        zone_to_site_rank : dict[int, int], optional
            Maps zone index (``Group`` column in CSV) to site rank.
            If ``None``, zone 0 → rank 1, zone 1 → rank 2, etc.
        """
        import cosolvkit.analysis.hotspot_visualization as viz
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

    # ------------------------------------------------------------------
    # Pocket residue identification
    # ------------------------------------------------------------------

    def find_pocket_residues(self, site, cutoff: float = 4.5) -> None:
        """Find protein residues that line the hotspot cavity and store them on *site*.

        Uses a KD-tree over the blob voxel coordinates (derived from
        ``site.voxel_mask``, ``site.grid_origin``, and ``site.grid_delta``) to
        efficiently identify protein heavy atoms within *cutoff* Å.  One
        :class:`PocketResidue` is appended to ``site.pocket_residues`` for each
        qualifying protein residue.

        Atom positions are read from the current frame of ``self.universe``
        (typically frame 0 or whatever frame the trajectory is at when this
        method is called).

        Parameters
        ----------
        site : Hotspot
            The hotspot to annotate.  Must have ``voxel_mask``, ``grid_origin``,
            and ``grid_delta`` already set (populated by
            :meth:`HotspotDetector.detect`).
        cutoff : float
            Distance threshold in Å (default 4.5).  Protein residues with any
            heavy atom within *cutoff* of any blob voxel are included.
        """
        if self.universe is None:
            raise ValueError(
                "find_pocket_residues requires a loaded MDAnalysis Universe "
                "(PocketPropertyCalculator.universe is None)."
            )
        if site.grid_origin is None or site.grid_delta is None:
            raise ValueError(
                "site.grid_origin / site.grid_delta are not set.  "
                "Call HotspotDetector.detect() before find_pocket_residues()."
            )

        # Reconstruct Angstrom coordinates of blob voxels
        voxel_indices = np.argwhere(site.voxel_mask)          # (N, 3) int
        voxel_coords = (
            site.grid_origin + voxel_indices * site.grid_delta  # (N, 3) float
        )
        n_voxels = len(voxel_coords)
        if n_voxels == 0:
            self.logger.warning("find_pocket_residues: site has no voxels, skipping.")
            return

        tree = cKDTree(voxel_coords)

        u = self.universe
        protein_ag = u.select_atoms("protein and not name H*")

        site.pocket_residues = []
        for res in protein_ag.residues:
            res_pos = res.atoms.positions  # (n_res_atoms, 3)

            # Nearest voxel distance for each heavy atom; keep residue if any is close
            dists, _ = tree.query(res_pos, k=1)
            min_dist = float(dists.min())
            if min_dist > cutoff:
                continue

            # Count unique blob voxels contacted by any heavy atom of this residue
            contacted = set()
            for voxel_id_list in tree.query_ball_point(res_pos, r=cutoff):
                contacted.update(voxel_id_list)
            n_contact = len(contacted)

            pr = PocketResidue(
                resid=int(res.resid),
                resindex=int(res.resindex),
                resname=str(res.resname),
                chain=str(res.segid),
                n_contact_voxels=n_contact,
                min_dist_ang=round(min_dist, 4),
                contact_fraction=round(n_contact / n_voxels, 4),
            )
            site.pocket_residues.append(pr)

        self.logger.info(
            "find_pocket_residues: %d residues within %.1f Å of site %d (%s)",
            len(site.pocket_residues), cutoff, site.site_id, site.cosolvent,
        )

    # ------------------------------------------------------------------
    # RMSF annotation
    # ------------------------------------------------------------------

    def annotate_residue_rmsf(self, site, rmsf_by_resid: Dict[int, float]) -> None:
        """Map pre-computed RMSF values onto pocket residues.

        Does **not** run any trajectory analysis — it just looks up each
        pocket residue's ``resid`` in *rmsf_by_resid* and stores the result.
        The expectation is that RMSF was already computed earlier in the
        analysis pipeline (e.g. via :class:`cosolvkit.analysis.analysis.Report`)
        and the caller passes the resulting mapping here.

        Call after :meth:`find_pocket_residues`.

        Parameters
        ----------
        site : Hotspot
            Site whose ``pocket_residues`` to annotate.
        rmsf_by_resid : dict[int, float]
            Mapping ``{resid: rmsf_angstroms}``.  Typically built from the
            ``RMSF`` result in the analysis pipeline::

                ca = universe.select_atoms("protein and name CA")
                rmsf_result = RMSF(ca).run()
                rmsf_by_resid = {
                    int(res.resid): float(val)
                    for res, val in zip(ca.residues, rmsf_result.results.rmsf)
                }

            Pocket residues whose ``resid`` is absent from the mapping receive
            ``rmsf = None``.
        """
        if not site.pocket_residues:
            return
        matched = 0
        for pr in site.pocket_residues:
            val = rmsf_by_resid.get(pr.resid)
            if val is not None:
                pr.rmsf = round(float(val), 4)
                matched += 1
        self.logger.info(
            "annotate_residue_rmsf: mapped RMSF for %d / %d pocket residues "
            "of site %d (%s)",
            matched, len(site.pocket_residues), site.site_id, site.cosolvent,
        )

    # ------------------------------------------------------------------
    # Per-frame cosolvent contact tracking
    # ------------------------------------------------------------------

    def compute_cosolvent_contacts(
        self,
        site,
        cosolvent_names: List[str],
        contact_cutoff: float = 4.0,
        step: int = 1,
    ) -> None:
        """Record trajectory frames in which each cosolvent molecule contacts each pocket residue.

        For every pocket residue in *site*, and for every individual cosolvent
        molecule (identified by its MDAnalysis ``resid``), this method iterates
        the trajectory and records the frame indices where any heavy atom of the
        molecule comes within *contact_cutoff* Å of any heavy atom of the
        residue.

        Results are stored in :attr:`PocketResidue.cosolvent_contacts` as::

            {cosolvent_name: {cosolvent_resid: [frame_idx, ...]}}

        Call after :meth:`find_pocket_residues`.

        Parameters
        ----------
        site : Hotspot
            Site whose ``pocket_residues`` to annotate.
        cosolvent_names : list[str]
            Residue names of cosolvent species to analyse.
        contact_cutoff : float
            Distance threshold in Å (default 4.0).
        step : int
            Trajectory stride — contacts are recorded only for sampled frames
            (default 1 = every frame).

        Notes
        -----
        Atom group objects are cached before the trajectory loop so that only
        ``.positions`` is accessed inside the hot path.  For very long
        trajectories with many cosolvent molecules, use ``step > 1`` to limit
        computation time.
        """
        if not site.pocket_residues:
            return
        if self.universe is None:
            raise ValueError(
                "compute_cosolvent_contacts requires a loaded MDAnalysis Universe."
            )

        u = self.universe

        # Cache residue AtomGroups once before the trajectory loop
        res_atom_groups: Dict[int, Any] = {
            pr.resindex: u.select_atoms(f"resindex {pr.resindex} and not name H*")
            for pr in site.pocket_residues
        }

        for cosolvent_name in cosolvent_names:
            cosol_heavy = u.select_atoms(f"resname {cosolvent_name} and not name H*")
            if len(cosol_heavy) == 0:
                self.logger.warning(
                    "compute_cosolvent_contacts: no atoms found for resname %s",
                    cosolvent_name,
                )
                continue

            cosol_resids: List[int] = [int(r) for r in np.unique(cosol_heavy.resids)]

            # Cache per-molecule AtomGroups
            mol_atom_groups: Dict[int, Any] = {
                rid: u.select_atoms(
                    f"resname {cosolvent_name} and resid {rid} and not name H*"
                )
                for rid in cosol_resids
            }

            n_frames = 0
            for ts in u.trajectory[::step]:
                frame = int(ts.frame)
                n_frames += 1
                for rid, mol_ag in mol_atom_groups.items():
                    mol_pos = mol_ag.positions  # (n_mol_atoms, 3)
                    for pr in site.pocket_residues:
                        res_pos = res_atom_groups[pr.resindex].positions  # (n_res_atoms, 3)
                        dists = np.linalg.norm(
                            mol_pos[:, np.newaxis, :] - res_pos[np.newaxis, :, :],
                            axis=-1,
                        )  # (n_mol_atoms, n_res_atoms)
                        if dists.min() <= contact_cutoff:
                            (
                                pr.cosolvent_contacts
                                .setdefault(cosolvent_name, {})
                                .setdefault(rid, [])
                                .append(frame)
                            )

            # Ensure frame lists are sorted (trajectory may not always be forward)
            for pr in site.pocket_residues:
                mol_dict = pr.cosolvent_contacts.get(cosolvent_name, {})
                for rid in mol_dict:
                    mol_dict[rid].sort()

            self.logger.info(
                "compute_cosolvent_contacts: %s — scanned %d frames, "
                "%d molecules, %d pocket residues for site %d",
                cosolvent_name, n_frames, len(cosol_resids),
                len(site.pocket_residues), site.site_id,
            )
