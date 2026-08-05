#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit — binding-site detection: group hotspots (any cosolvent) into
# pockets by connectivity of the union of their voxel masks.
#
import os
import json

import numpy as np
from scipy.ndimage import label as _ndlabel

from cosolvkit.analysis.core.grid import resample_mask_to_grid


def choose_reference_grid(hotspots):
    """Pick the member grid with the most voxels as the common reference."""
    best = max(hotspots, key=lambda h: int(np.prod(h.voxel_mask.shape)))
    return (np.asarray(best.grid_origin, dtype=float),
            np.asarray(best.grid_delta, dtype=float),
            tuple(best.voxel_mask.shape))


def _connectivity_structure(connectivity):
    if connectivity == 6:
        return np.array([[[0,0,0],[0,1,0],[0,0,0]],
                         [[0,1,0],[1,1,1],[0,1,0]],
                         [[0,0,0],[0,1,0],[0,0,0]]], dtype=int)
    return np.ones((3, 3, 3), dtype=int)  # 26-connectivity


def group_hotspots(probe_results, connectivity=6, merge_tolerance_ang=0.0):
    """Group hotspots across cosolvents into binding sites by mask connectivity.

    probe_results : dict[str, list[Hotspot]]
    merge_tolerance_ang : float
        Max surface gap (Å) between two hotspots that still merges them; the union is
        dilated by tol/2 per side before connected-components labeling. 0 = literal
        touch only. Dilation affects grouping ONLY — each group's stored ``union_mask``
        is the OR of members' original (undilated) resampled masks.
    Returns list of dicts: {members, union_mask, ref_origin, ref_delta, ref_shape}.
    """
    hotspots = [h for sites in probe_results.values() for h in sites]
    if not hotspots:
        return []

    ref_o, ref_d, ref_shape = choose_reference_grid(hotspots)

    resampled = [
        resample_mask_to_grid(h.voxel_mask, h.grid_origin, h.grid_delta,
                              ref_o, ref_d, ref_shape)
        for h in hotspots
    ]
    union = np.zeros(ref_shape, dtype=bool)
    for m in resampled:
        union |= m

    # Dilate ONLY for the grouping decision, to bridge gaps up to the tolerance.
    gridsize = float(ref_d[0])
    radius_vox = max(0, int(round((merge_tolerance_ang / 2.0) / gridsize)))
    if radius_vox > 0:
        from scipy.ndimage import binary_dilation
        from skimage.morphology import ball
        grouping_mask = binary_dilation(union, structure=ball(radius_vox))
    else:
        grouping_mask = union

    labels, n = _ndlabel(grouping_mask, structure=_connectivity_structure(connectivity))

    groups = {}
    for h, m in zip(hotspots, resampled):
        # Assign via the ORIGINAL mask's voxels, read from the (dilated) labeling.
        lab_counts = np.bincount(labels[m].ravel())
        if len(lab_counts) <= 1:
            continue  # hotspot has no voxels in the union (shouldn't happen)
        lab_counts[0] = 0  # ignore background
        lbl = int(lab_counts.argmax())
        groups.setdefault(lbl, {"members": [], "resampled": []})
        groups[lbl]["members"].append(h)
        groups[lbl]["resampled"].append(m)

    result = []
    for lbl, g in sorted(groups.items()):
        umask = np.zeros(ref_shape, dtype=bool)
        for m in g["resampled"]:
            umask |= m
        result.append({
            "members": g["members"],
            "union_mask": umask,
            "ref_origin": ref_o, "ref_delta": ref_d, "ref_shape": ref_shape,
        })
    return result


def _union_shape_features(union_mask):
    """solidity/extent/axis lengths of the single union region (0.0 if degenerate)."""
    try:
        from skimage.measure import regionprops
    except ImportError:
        return {"solidity": 0.0, "extent": 0.0,
                "axis_major_length": 0.0, "axis_minor_length": 0.0}
    labeled = union_mask.astype(int)
    props = regionprops(labeled)
    if not props:
        return {"solidity": 0.0, "extent": 0.0,
                "axis_major_length": 0.0, "axis_minor_length": 0.0}
    p = props[0]
    def _safe(name):
        try:
            return float(getattr(p, name))
        except Exception:
            return 0.0
    return {
        "solidity": _safe("solidity"),
        "extent": _safe("extent"),
        "axis_major_length": _safe("axis_major_length"),
        "axis_minor_length": _safe("axis_minor_length"),
    }


# Survival-probability metric feeding BindingSite.residence (the ``kinetics`` score
# feature): the lag at which survival falls to 0.5. Higher = longer residence = better.
# sp_mrt and sp_tau_single perform equivalently; half-life is the most interpretable.
KINETICS_METRIC = "sp_half_life"


def _finite_values(members, attr):
    """Member values of *attr* that are present and finite (may be empty)."""
    return _finite_from(members, lambda m: getattr(m, attr, None))


def _finite_properties(members, key):
    """Member ``properties[key]`` values that are present and finite (may be empty)."""
    return _finite_from(members, lambda m: m.properties.get(key))


def _finite_from(members, getter):
    out = []
    for m in members:
        v = getter(m)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def build_binding_site(site_id, group, n_total_cosolvents,
                       chemotype_map=None, n_total_probe_chemotypes=None):
    """Aggregate a group of member hotspots into a BindingSite.

    Members need not carry AGFE data: hotspots not derived from a density map yield a
    site with ``agfe_min=None``, an unweighted centroid and an empty pharmacophore.
    """
    from cosolvkit.analysis.core.models import BindingSite

    members = group["members"]
    ref_o = np.asarray(group["ref_origin"], dtype=float)
    ref_d = np.asarray(group["ref_delta"], dtype=float)
    union = group["union_mask"]

    # Centroid of member centroids weighted by |agfe_min|, or plain mean without AGFE.
    cents = np.array([m.centroid for m in members], dtype=float)
    agfe_vals = _finite_values(members, "agfe_min")
    if len(agfe_vals) == len(members):
        w = np.abs(np.array(agfe_vals, dtype=float))
    else:
        w = np.zeros(len(members), dtype=float)
    centroid = (cents.T @ w) / w.sum() if w.sum() > 0 else cents.mean(axis=0)

    agfe_min = min(agfe_vals) if agfe_vals else None
    top_vals = _finite_values(members, "agfe_mean_top_pct")
    agfe_mean_top_pct = min(top_vals) if top_vals else None
    gridsize = float(ref_d[0])
    volume = float(union.sum()) * (gridsize ** 3)

    shape = _union_shape_features(union)

    favorable_atomtypes = sorted({a for m in members for a in m.favorable_atomtypes})

    pharmacophore = {}
    for m in members:
        d = pharmacophore.setdefault(m.cosolvent, {})
        for atype, val in m.per_type_agfe.items():
            v = float(val)
            if atype not in d or v < d[atype]:
                d[atype] = round(v, 4)

    sp_vals = _finite_properties(members, KINETICS_METRIC)
    residence = max(sp_vals) if sp_vals else None

    cosolvents = sorted({m.cosolvent for m in members})

    from cosolvkit.analysis.core.chemotypes import probe_chemotypes
    chemotypes = probe_chemotypes(cosolvents, chemotype_map)

    return BindingSite(
        site_id=site_id, member_hotspots=members, voxel_mask=union, centroid=centroid,
        probe_chemotypes=chemotypes,
        n_total_probe_chemotypes=n_total_probe_chemotypes,
        agfe_min=agfe_min, agfe_mean_top_pct=agfe_mean_top_pct, volume=volume,
        solidity=shape["solidity"], extent=shape["extent"],
        axis_major_length=shape["axis_major_length"],
        axis_minor_length=shape["axis_minor_length"],
        favorable_atomtypes=favorable_atomtypes, pharmacophore=pharmacophore,
        residence=residence, residence_metric=KINETICS_METRIC,
        cosolvents=cosolvents,
        n_total_cosolvents=n_total_cosolvents,
        grid_origin=ref_o, grid_delta=ref_d,
    )


class BindingSiteDetector:
    """Detect binding sites by grouping hotspots (mask connectivity) and scoring them."""

    def __init__(self, probe_results, connectivity=6, weights=None,
                 merge_tolerance_ang=0.0, probe_chemotype_overrides=None,
                 field_maps=None):
        from cosolvkit.analysis.core.chemotypes import (
            n_available_chemotypes,
            resolve_probe_chemotypes,
        )
        self.probe_results = probe_results
        self.connectivity = connectivity
        self.weights = weights
        self.merge_tolerance_ang = merge_tolerance_ang
        self.field_maps = field_maps
        self.n_total_cosolvents = len(probe_results)
        self.chemotype_map = resolve_probe_chemotypes(probe_chemotype_overrides)
        # Denominator for probe_chemotype_coverage: what THIS panel can express.
        self.n_total_probe_chemotypes = n_available_chemotypes(
            list(probe_results.keys()), self.chemotype_map)

    def detect(self):
        from cosolvkit.analysis.core.scoring import score_binding_sites
        groups = group_hotspots(self.probe_results, connectivity=self.connectivity,
                                merge_tolerance_ang=self.merge_tolerance_ang)
        sites = [
            build_binding_site(
                site_id=i + 1, group=g,
                n_total_cosolvents=self.n_total_cosolvents,
                chemotype_map=self.chemotype_map,
                n_total_probe_chemotypes=self.n_total_probe_chemotypes,
            )
            for i, g in enumerate(groups)
        ]
        if self.field_maps:
            from cosolvkit.analysis.core.site_features import (
                ProbeFieldSampler, fused_site_features,
            )
            fused_site_features(sites, ProbeFieldSampler(self.field_maps))
        score_binding_sites(sites, self.weights)   # sets .combined and .rank
        sites.sort(key=lambda s: s.rank)
        return sites


def identify_binding_sites(probe_results, connectivity=6, weights=None,
                           merge_tolerance_ang=0.0, probe_chemotype_overrides=None,
                           field_maps=None):
    """Group per-cosolvent hotspots into ranked cross-cosolvent binding sites.

    *field_maps* is ``{cosolvent: (agfe_array, origin, delta)}``. Supplying it lets the
    scorer use count-normalised features fused over every probe at each site's point,
    instead of a best-of-members maximum biased by member count; omitting it warns.
    """
    return BindingSiteDetector(
        probe_results, connectivity=connectivity, weights=weights,
        merge_tolerance_ang=merge_tolerance_ang,
        probe_chemotype_overrides=probe_chemotype_overrides,
        field_maps=field_maps).detect()


def export_binding_sites(sites, out_path):
    """Write binding_sites.csv/.json, pharmacophore json, and a rank label .dx."""
    import pandas as pd
    os.makedirs(out_path, exist_ok=True)

    rows = [s.to_dict() for s in sites]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_path, "binding_sites.csv"), index=False)
    with open(os.path.join(out_path, "binding_sites.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    pharm = [{"site_id": s.site_id, "rank": s.rank,
              "cosolvents": s.cosolvents, "pharmacophore": s.pharmacophore}
             for s in sites]
    with open(os.path.join(out_path, "binding_sites_pharmacophore.json"), "w") as fh:
        json.dump(pharm, fh, indent=2)

    # Label map: voxel value = binding-site rank (0 = background). All sites share the
    # reference grid from grouping (same mask shape and grid_origin/grid_delta).
    if sites and sites[0].voxel_mask is not None and sites[0].grid_delta is not None:
        try:
            from gridData import Grid
            shape = sites[0].voxel_mask.shape
            origin = np.asarray(sites[0].grid_origin, dtype=float)
            delta = np.asarray(sites[0].grid_delta, dtype=float)
            rank_arr = np.zeros(shape, dtype=float)
            # Higher rank number = worse; paint worst first so rank 1 wins overlaps.
            for s in sorted(sites, key=lambda x: -x.rank):
                rank_arr[s.voxel_mask] = float(s.rank)
            edges = [origin[d] + np.arange(shape[d] + 1) * delta[d] for d in range(3)]
            Grid(rank_arr, edges=edges).export(
                os.path.join(out_path, "binding_site_labels.dx"))
        except Exception:
            pass
