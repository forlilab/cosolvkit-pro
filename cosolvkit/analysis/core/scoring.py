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

import warnings

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
    # Most-negative AGFE at the site (lower is better). See _affinity_values.
    "affinity": 3.0,
    # How many PROBES hit the site (not atom types — that is chemotype_diversity below).
    # Effectively a member count, so biased toward big sites, but useful on average.
    "probe_coverage": 2.0,
    "volume": 1.0,
    "kinetics": 1.0,
    "shape": 1.0,
    # Number of favourable pharmacophoric ATOM TYPES (HBD/HBA/Car/Cal/Hal) at the site.
    # Requires atom-type-split density maps (``density_maps.use_atomtypes: true``);
    # without them every site has zero favourable atom types and this contributes
    # nothing. Do not confuse it with probe diversity — that is ``probe_coverage``
    # (how many probes) and ``probe_chemotype_coverage`` (how many probe chemotypes).
    "chemotype_diversity": 1.0,
    # Fraction of probe chemotype classes (aromatic/aliphatic/HBD/HBA/anionic/cationic)
    # represented among the probes that hit the site. Opt-in: weight 0.0 by default,
    # pending validation across more than one target.
    "probe_chemotype_coverage": 0.0,
    # Read from the AGFE map as a FIELD at member-hotspot centroids (see core/field.py),
    # not from the thresholded blob. Opt-in at 0.0, pending a weight sweep.
    "field_contrast": 0.0,
    "field_sharpness": 0.0,
    # Enclosure proxy: protein heavy atoms within 8 A of the site, averaged (not maxed) over
    # member hotspots so member count cannot inflate it. Opt-in at 0.0, pending a weight sweep.
    "buriedness": 0.0,
}

# Features whose raw value is "lower is better" -> inverted min-max (most-negative -> 1).
# `shape` (solidity) is inverted because known sites are LESS convex than novel ones — real
# pockets are irregular clefts.
_BS_INVERTED_FEATURES = {"affinity", "field_contrast", "shape"}

# ``diversity`` was renamed to ``chemotype_diversity`` because it scores atom types, not
# probes, and readers reliably assumed the latter. Accepted with a warning rather than
# rejected so that saved weight sets and the dashboard keep working.
_LEGACY_WEIGHT_ALIASES = {"diversity": "chemotype_diversity"}


def _site_property(site, name):
    """A property attached to the site itself (fused features), or None."""
    v = (getattr(site, "properties", None) or {}).get(name)
    return float(v) if v is not None and np.isfinite(v) else None


def _best_member_property(site, name, prefer="min"):
    """Best value of *name* over a site's member hotspots, or None if none carry it.

    *prefer* is ``"min"`` (most-negative, e.g. contrast) or ``"max"`` (e.g. sharpness).
    """
    vals = []
    for hs in getattr(site, "member_hotspots", None) or []:
        v = (getattr(hs, "properties", None) or {}).get(name)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return None
    return min(vals) if prefer == "min" else max(vals)


#: Use the count-normalised fused affinity instead of best-member ``agfe_min``. Off by
#: default: it removes the member-count bias but ranked worse. The fused values are still
#: computed and exported whenever the maps are supplied.
USE_FUSED_AFFINITY = False


def _mean_member_property(site, name):
    """Mean of *name* over member hotspots, or None. Mean, not max: a best-of-members
    summary is inflated by member count."""
    vals = []
    for hs in getattr(site, "member_hotspots", None) or []:
        v = (getattr(hs, "properties", None) or {}).get(name)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _affinity_values(binding_sites):
    """Fused, count-normalised affinity when available; best-member ``agfe_min`` otherwise.

    ``agfe_min`` is a best-of-members minimum and so is inflated by member count; the fused
    value samples every probe at the site's point (see core/site_features.py). The fallback
    warns, since a count-biased affinity at weight 3.0 dominates the score.
    """
    if not USE_FUSED_AFFINITY:
        return [s.agfe_min for s in binding_sites]
    fused = [_site_property(s, "fused_affinity") for s in binding_sites]
    if any(v is not None for v in fused):
        return fused
    warnings.warn(
        "Binding-site 'affinity' is falling back to best-member agfe_min, which correlates "
        "with member count at rho -0.82 (best-of-n inflation). Pass the per-probe maps so "
        "fused_affinity can be computed — see core/site_features.fused_site_features.",
        UserWarning, stacklevel=2,
    )
    return [s.agfe_min for s in binding_sites]


def _binding_site_feature_values(binding_sites):
    """Raw scalar per weightable feature (None allowed -> contributes 0)."""
    return {
        "affinity":       _affinity_values(binding_sites),
        "probe_coverage": [s.probe_coverage for s in binding_sites],
        "volume":         [s.volume for s in binding_sites],
        "kinetics":       [s.residence for s in binding_sites],
        "shape":          [s.solidity for s in binding_sites],
        "chemotype_diversity": [float(len(s.favorable_atomtypes))
                                for s in binding_sites],
        "probe_chemotype_coverage": [getattr(s, "probe_chemotype_coverage", None)
                                     for s in binding_sites],
        # Fused (count-normalised) when present, else best-of-members.
        "field_contrast":  [_site_property(s, "fused_contrast")
                            if _site_property(s, "fused_contrast") is not None
                            else _best_member_property(s, "field_contrast", prefer="min")
                            for s in binding_sites],
        "buriedness":     [_mean_member_property(s, "buriedness") for s in binding_sites],
        "field_sharpness": [_site_property(s, "fused_sharpness")
                            if _site_property(s, "fused_sharpness") is not None
                            else _best_member_property(s, "field_sharpness", prefer="max")
                            for s in binding_sites],
    }


def normalize_weights(weights):
    """Resolve a user weights dict against the defaults.

    Applies the legacy aliases (with a ``DeprecationWarning``) and rejects any key that
    is not a real feature — a silently-ignored weight would look like a working knob
    while doing nothing.
    """
    if not weights:
        return dict(DEFAULT_BINDING_SITE_WEIGHTS)
    resolved = {}
    for key, value in weights.items():
        canonical = _LEGACY_WEIGHT_ALIASES.get(key, key)
        if canonical != key:
            warnings.warn(
                f"Binding-site weight {key!r} is deprecated; use {canonical!r} "
                "(it scores favourable ATOM TYPES, not probes).",
                DeprecationWarning, stacklevel=3,
            )
        if canonical not in DEFAULT_BINDING_SITE_WEIGHTS:
            raise ValueError(
                f"Unknown binding-site weight {key!r}. Valid weights: "
                f"{sorted(DEFAULT_BINDING_SITE_WEIGHTS)}"
            )
        resolved[canonical] = value
    return {**DEFAULT_BINDING_SITE_WEIGHTS, **resolved}


def score_binding_sites(binding_sites, weights=None):
    """Score and rank binding sites by a signed weighted sum of globally
    min-max-normalised features (higher = better), in place.

    Each weightable feature is normalised across *binding_sites* to [0,1],
    oriented higher=better (affinity inverts agfe_min). Sites missing a feature
    (None/non-finite) contribute 0 for it. ``weights`` may be negative and need
    not sum to 1; missing keys fall back to DEFAULT_BINDING_SITE_WEIGHTS, and an
    unrecognised key raises (see :func:`normalize_weights`).
    Sets ``.combined`` and ``.rank`` (1 = highest combined).
    """
    if not binding_sites:
        return
    weights = normalize_weights(weights)
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
