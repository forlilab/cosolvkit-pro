#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Scoring helpers: normalization primitives and binding-site ranking.
# Hotspots themselves are ranked by agfe_min in HotspotDetector.detect(), not here.
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
    # How many PROBES hit the site. Effectively a member count, so biased toward
    # large sites.
    "probe_coverage": 2.0,
    "volume": 1.0,
    "kinetics": 1.0,
    "shape": 1.0,
    # Number of favourable pharmacophoric ATOM TYPES (HBD/HBA/Car/Cal/Hal) at the site.
    # Needs atom-type-split density maps (``density_maps.use_atomtypes: true``);
    # without them it is zero everywhere and contributes nothing. Not probe diversity:
    # that is ``probe_coverage`` (how many probes) and ``probe_chemotype_coverage``.
    "chemotype_diversity": 1.0,
    # Fraction of probe chemotype classes represented among the probes hitting the
    # site. Opt-in (0.0).
    "probe_chemotype_coverage": 0.0,
    # Read from the AGFE map as a FIELD at member-hotspot centroids (see core/field.py),
    # not from the thresholded blob. Opt-in (0.0).
    "field_contrast": 0.0,
    "field_sharpness": 0.0,
    # Normalised enclosure: fraction of a ball around the site that solvent can reach, averaged
    # over member hotspots. Prefer this over `buriedness`, which is an unbounded atom count whose
    # AUC rises monotonically with its radius (0.524 -> 0.806 over 4-20 A on FosAKP) and therefore
    # reports centrality rather than enclosure. This one is bounded in [0,1], plateaus with radius,
    # is nearly volume-independent (|rho| <= 0.13 vs volume), and retains AUC 0.685 after
    # residualising on buriedness AND volume. Inverted: LOWER accessible fraction = more enclosed.
    # Requires the accessible-volume mask, so it is only populated when the detector can find
    # `solvent_accessible_map.dx`. Opt-in (0.0) pending validation on a second target.
    "accessible_fraction": 0.0,
}

# Features whose raw value is "lower is better" -> inverted min-max (most-negative -> 1).
# `shape` (solidity) is inverted because real pockets are irregular clefts, i.e. less convex.
_BS_INVERTED_FEATURES = {"affinity", "field_contrast", "shape",
                         "accessible_fraction"}   # lower = more enclosed

# ``diversity`` scored atom types, not probes, and was renamed. Accepted with a warning
# rather than rejected so saved weight sets keep working.
_LEGACY_WEIGHT_ALIASES = {"diversity": "chemotype_diversity"}

# Weights that have been REMOVED. Accepted-with-warning and dropped rather than raising, so a
# saved weight set keeps loading; deliberately NOT aliased to a replacement, because the
# replacement measures a different quantity on a different scale with the opposite sign, and
# silently remapping would change a score without telling anyone.
#   buriedness -> superseded by `accessible_fraction`. It was a raw, unbounded count of protein
#   heavy atoms in an 8 A ball whose AUC rose monotonically with its radius (0.524 at 4 A to
#   0.806 at 20 A, counting a quarter of the protein at the top), so it reported centrality
#   rather than enclosure.
_RETIRED_WEIGHTS = {
    "buriedness": "accessible_fraction",
}


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
#: default: it removes the member-count bias but ranked worse. Fused values are still
#: computed and exported when the maps are supplied.
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

    ``agfe_min`` is a best-of-members minimum and so is inflated by member count, which
    matters because affinity carries the largest weight; the fused value instead samples
    every probe at the site's point (see core/site_features.py).
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
        "accessible_fraction": [_mean_member_property(s, "accessible_fraction")
                                for s in binding_sites],
        "field_sharpness": [_site_property(s, "fused_sharpness")
                            if _site_property(s, "fused_sharpness") is not None
                            else _best_member_property(s, "field_sharpness", prefer="max")
                            for s in binding_sites],
    }


def normalize_weights(weights):
    """Resolve a user weights dict against the defaults.

    Applies legacy aliases (with a ``DeprecationWarning``) and raises on any unknown key,
    since a silently-ignored weight would look like a working knob.

    :param weights: partial ``{feature: weight}`` dict, or None/empty for the defaults.
    :return: full weights dict.
    """
    if not weights:
        return dict(DEFAULT_BINDING_SITE_WEIGHTS)
    resolved = {}
    for key, value in weights.items():
        if key in _RETIRED_WEIGHTS:
            warnings.warn(
                f"Binding-site weight {key!r} has been REMOVED and is ignored; use "
                f"{_RETIRED_WEIGHTS[key]!r} instead. It is not an alias -- the replacement is "
                "normalised, bounded and inverted, so port the weight deliberately rather than "
                "copying its value across.",
                DeprecationWarning, stacklevel=3,
            )
            continue
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
    """Rank binding sites in place by a weighted sum of normalised features.

    Each feature is min-max normalised across *binding_sites* and oriented so that
    higher = better; a site missing a feature (None/non-finite) contributes 0 for it.
    Sets ``.combined`` and ``.rank`` (1 = highest combined).

    :param binding_sites: sites to score; mutated in place.
    :param weights: partial weights dict; weights may be negative and need not sum
        to 1 (see :func:`normalize_weights`).
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

    # A weight on a feature that is constant across every site cannot change the ranking.
    # Two of the default weights have historically been in exactly this state --
    # `chemotype_diversity` is identically zero without `density_maps.use_atomtypes: true`, and
    # `kinetics` without `sp_top_n > 0` -- so the weight vector claimed to use six features while
    # ranking on four. Warn rather than silently mis-describe the model.
    for feat, vals in raw.items():
        w = weights.get(feat, 0.0)
        if w == 0.0:
            continue
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        if not finite:
            warnings.warn(
                f"Binding-site weight {feat!r}={w} has no effect: the feature is missing on "
                f"every site. Set the weight to 0.0 or supply the data it needs.",
                RuntimeWarning, stacklevel=2)
        elif len(binding_sites) > 1 and np.ptp(np.asarray(finite, dtype=float)) == 0.0:
            warnings.warn(
                f"Binding-site weight {feat!r}={w} has no effect: the feature is constant "
                f"({finite[0]!r}) across all {len(binding_sites)} sites, so it cannot change the "
                f"ranking. For 'chemotype_diversity' this usually means "
                f"'density_maps.use_atomtypes' is not enabled; for 'kinetics', 'sp_top_n' is 0.",
                RuntimeWarning, stacklevel=2)

    for i, site in enumerate(binding_sites):
        site.combined = float(sum(weights.get(f, 0.0) * norm[f][i] for f in raw))

    for rank, site in enumerate(sorted(binding_sites,
                                       key=lambda s: s.combined, reverse=True), start=1):
        site.rank = rank
