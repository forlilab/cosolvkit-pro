#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit — binding-site detection: group hotspots (any cosolvent) into
# pockets by connectivity of the union of their voxel masks. Replaces the
# Jaccard-community consensus detector.
#
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


def group_hotspots(probe_results, connectivity=26):
    """Group hotspots across cosolvents into binding sites by mask connectivity.

    probe_results : dict[str, list[Hotspot]]
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

    labels, n = _ndlabel(union, structure=_connectivity_structure(connectivity))

    groups = {}
    for h, m in zip(hotspots, resampled):
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


def build_binding_site(site_id, group, n_total_cosolvents):
    """Aggregate a group of member hotspots into a BindingSite."""
    from cosolvkit.analysis.core.models import BindingSite

    members = group["members"]
    ref_o = np.asarray(group["ref_origin"], dtype=float)
    ref_d = np.asarray(group["ref_delta"], dtype=float)
    union = group["union_mask"]

    # Affinity-weighted centroid of member centroids (weight |agfe_min|).
    cents = np.array([m.centroid for m in members], dtype=float)
    w = np.array([abs(m.agfe_min) for m in members], dtype=float)
    centroid = (cents.T @ w) / w.sum() if w.sum() > 0 else cents.mean(axis=0)

    agfe_min = min(m.agfe_min for m in members)
    agfe_mean_top_pct = min(m.agfe_mean_top_pct for m in members)
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

    sp_vals = [m.properties.get("sp_mrt") for m in members
               if m.properties.get("sp_mrt") is not None
               and np.isfinite(m.properties.get("sp_mrt"))]
    residence = max(sp_vals) if sp_vals else None

    cosolvents = sorted({m.cosolvent for m in members})

    return BindingSite(
        site_id=site_id, member_hotspots=members, voxel_mask=union, centroid=centroid,
        agfe_min=agfe_min, agfe_mean_top_pct=agfe_mean_top_pct, volume=volume,
        solidity=shape["solidity"], extent=shape["extent"],
        axis_major_length=shape["axis_major_length"],
        axis_minor_length=shape["axis_minor_length"],
        favorable_atomtypes=favorable_atomtypes, pharmacophore=pharmacophore,
        residence=residence, cosolvents=cosolvents,
        n_total_cosolvents=n_total_cosolvents,
        grid_origin=ref_o, grid_delta=ref_d,
    )
