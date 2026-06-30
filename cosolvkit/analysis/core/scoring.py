#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Composite scoring helpers, moved verbatim from pocket_properties.py.
# Unification with HotspotDetector's inline scoring happens in Task 13.
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


def combine_detect_scores(raw_f, raw_d, raw_v, weights):
    """Reproduce HotspotDetector.detect() scoring exactly.

    favorability: inverted min-max; diversity: raw; volume: divide-by-max.
    ``weights`` is the already-sum-1-normalised dict {favorability, diversity, volume}.
    Returns (composite, f_norm, v_norm) as np.ndarrays.
    """
    raw_d = np.asarray(raw_d, dtype=float)
    f_norm = _inverted_minmax(raw_f)
    v_norm = _divide_by_max(raw_v)
    composite = (weights["favorability"] * f_norm
                 + weights["diversity"] * raw_d
                 + weights["volume"] * v_norm)
    return composite, f_norm, v_norm


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Maps short weight-key aliases to the actual Hotspot attribute name.
_CORE_ATTR_ALIASES = {
    "favorability": "favorability_score",
    "diversity": "diversity_score",
    "volume": "volume_score",
    "favorability_score": "favorability_score",
    "diversity_score": "diversity_score",
    "volume_score": "volume_score",
}


def _get_site_value(site, key):
    """Retrieve a scoring component from a Hotspot.

    Checks core attribute aliases first, then site.properties.
    Returns None if the key is unknown or the value is None.
    """
    attr = _CORE_ATTR_ALIASES.get(key)
    if attr is not None:
        return getattr(site, attr, None)
    return site.properties.get(key)


def compute_composite_score(sites, score_weights):
    """Recompute composite scores for a list of Hotspot objects.

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
    sites : list[Hotspot]
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

    # Compute normalised component scores per site.
    # For each active key, normalise only the finite values via _minmax
    # (flat key -> ones, matching (hi-lo)<1e-20 -> 1.0); None/non-finite
    # positions stay 0.0.
    key_norm = {}
    for key in active_keys:
        vals = raw_values[key]
        finite_indices = [i for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
        finite_vals = np.array([vals[i] for i in finite_indices], dtype=float)
        normed = _minmax(finite_vals)
        full = np.zeros(len(vals))
        for pos, i in enumerate(finite_indices):
            full[i] = normed[pos]
        key_norm[key] = full

    for i, site in enumerate(sites):
        composite = 0.0
        for key in active_keys:
            composite += norm_w[key] * key_norm[key][i]
        site.composite_score = composite

    # Re-rank descending
    for rank, site in enumerate(sorted(sites, key=lambda s: s.composite_score,
                                       reverse=True), start=1):
        site.rank = rank
