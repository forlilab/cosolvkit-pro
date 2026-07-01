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
