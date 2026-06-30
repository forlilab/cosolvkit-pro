#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Composite scoring helpers, moved verbatim from pocket_properties.py.
# Unification with HotspotDetector's inline scoring happens in Task 11.
#

import numpy as np


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
