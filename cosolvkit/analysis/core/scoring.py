#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Scoring helpers: normalization primitives and binding-site ranking.
# Hotspots are raw-feature objects ranked by agfe_min directly in
# HotspotDetector.detect(); the composite-score machinery formerly here
# has been removed.
#

import numpy as np


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

def _inverted_minmax(arr):
    """Most-negative -> 1.0, least-negative -> 0.0. Flat -> all ones."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if (hi - lo) < 1e-20:
        return np.ones_like(arr)
    return (hi - arr) / (hi - lo)


def _divide_by_max(arr):
    """Divide by max; returns zeros when max <= 0."""
    arr = np.asarray(arr, dtype=float)
    m = arr.max()
    return arr / m if m > 0 else np.zeros_like(arr)


def _minmax(arr):
    """Standard min-max normalization. Flat -> all ones."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if (hi - lo) < 1e-20:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Binding-site scoring (signed weighted sum over global min-max, higher=better)
# ---------------------------------------------------------------------------

DEFAULT_BINDING_SITE_WEIGHTS = {
    "affinity": 3.0,
    "probe_coverage": 2.0,
    "volume": 1.0,
    "kinetics": 1.0,
    "shape": 1.0,
    "diversity": 1.0,
}

# Features whose raw value is "lower is better" -> inverted min-max (most-negative -> 1).
_BS_INVERTED_FEATURES = {"affinity"}


def _binding_site_feature_values(binding_sites):
    """Raw scalar per weightable feature (None allowed -> contributes 0)."""
    return {
        "affinity":       [s.agfe_min for s in binding_sites],
        "probe_coverage": [s.probe_coverage for s in binding_sites],
        "volume":         [s.volume for s in binding_sites],
        "kinetics":       [s.residence for s in binding_sites],
        "shape":          [s.solidity for s in binding_sites],
        "diversity":      [float(len(s.favorable_atomtypes)) for s in binding_sites],
    }


def score_binding_sites(binding_sites, weights=None):
    """Score and rank binding sites by a signed weighted sum of globally
    min-max-normalised features (higher = better), in place.

    Each weightable feature is normalised across *binding_sites* to [0,1],
    oriented higher=better (affinity inverts agfe_min). Sites missing a feature
    (None/non-finite) contribute 0 for it. ``weights`` may be negative and need
    not sum to 1; missing keys fall back to DEFAULT_BINDING_SITE_WEIGHTS.
    Sets ``.combined`` and ``.rank`` (1 = highest combined).
    """
    if not binding_sites:
        return
    weights = {**DEFAULT_BINDING_SITE_WEIGHTS, **(weights or {})}
    raw = _binding_site_feature_values(binding_sites)
    n = len(binding_sites)

    norm = {}
    for feat, vals in raw.items():
        finite_idx = [i for i, v in enumerate(vals)
                      if v is not None and np.isfinite(v)]
        full = np.zeros(n)
        if finite_idx:
            fv = np.array([vals[i] for i in finite_idx], dtype=float)
            normed = (_inverted_minmax(fv) if feat in _BS_INVERTED_FEATURES
                      else _minmax(fv))
            for pos, i in enumerate(finite_idx):
                full[i] = normed[pos]
        norm[feat] = full

    for i, site in enumerate(binding_sites):
        site.combined = float(sum(weights.get(f, 0.0) * norm[f][i] for f in raw))

    for rank, site in enumerate(sorted(binding_sites,
                                       key=lambda s: s.combined, reverse=True), start=1):
        site.rank = rank
