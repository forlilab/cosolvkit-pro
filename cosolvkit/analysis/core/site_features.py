"""Count-normalised binding-site features: sample the field, fuse across a fixed probe set.

A binding site is a union of hotspots from several probes, so any feature summarised as the *best
member* is inflated by how many members it has.
Instead the AGFE field is sampled at the site's representative point for **every probe in the
panel**, not just the members, so the per-site denominator is constant.

Fusion is a per-probe z-score across sites before summing, so a probe with a deeper AGFE scale
cannot outvote the rest.
"""

import numpy as np

from cosolvkit.analysis.core.field import field_descriptors

#: Descriptors fused into site-level features, and the site property each becomes.
FUSED_FEATURES = {
    "field_min_ball": "fused_affinity",
    "field_contrast": "fused_contrast",
    "field_sharpness": "fused_sharpness",
}

#: Descriptors where a more negative raw value means more site-like.
_LOWER_IS_BETTER = {"field_min_ball", "field_contrast"}


class ProbeFieldSampler:
    """Samples every probe's AGFE map at a physical point.

    Parameters
    ----------
    maps : dict
        ``{cosolvent: (array, origin, delta)}``. Grids may differ between probes — each map is
        sampled in its own frame, so no resampling is needed.
    """

    def __init__(self, maps):
        self.maps = {}
        for name, entry in (maps or {}).items():
            arr, origin, delta = entry
            self.maps[name] = (np.asarray(arr), np.asarray(origin, dtype=float),
                               np.asarray(delta, dtype=float))

    @property
    def probes(self):
        return sorted(self.maps)

    def descriptors_at(self, xyz):
        """``{cosolvent: {descriptor: value or None}}`` at *xyz*, for every probe."""
        return {name: field_descriptors(arr, origin, delta, xyz)
                for name, (arr, origin, delta) in self.maps.items()}


def _zscore_columns(v):
    """Column-wise z-score, ignoring NaNs; all-NaN or constant columns become 0."""
    out = np.zeros_like(v)
    for j in range(v.shape[1]):
        col = v[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        sd = np.nanstd(col[ok])
        if sd <= 0:
            continue
        z = np.zeros_like(col)
        z[ok] = (col[ok] - np.nanmean(col[ok])) / sd
        out[:, j] = z
    return out


def fused_site_features(sites, sampler, features=None):
    """Attach count-normalised fused field features to each site, in place.

    For each descriptor a ``(n_sites, n_probes)`` matrix is built by sampling every probe at every
    site's representative point, z-scored per probe across sites, then summed over probes. Higher
    is always better in the result. Returns the list of property names written.
    """
    features = features or FUSED_FEATURES
    sites = list(sites)
    if not sites or not sampler.probes:
        return []

    probes = sampler.probes
    per_site = []
    for s in sites:
        point = getattr(s, "centroid", None)
        per_site.append(None if point is None else sampler.descriptors_at(point))

    written = []
    for desc, prop in features.items():
        v = np.full((len(sites), len(probes)), np.nan)
        for i, d in enumerate(per_site):
            if d is None:
                continue
            for j, p in enumerate(probes):
                raw = d[p].get(desc)
                if raw is not None and np.isfinite(raw):
                    # orient so larger = more site-like before z-scoring
                    v[i, j] = -raw if desc in _LOWER_IS_BETTER else raw
        fused = _zscore_columns(v).sum(axis=1)
        for s, value in zip(sites, fused):
            s.add_property(prop, float(value))
        written.append(prop)
    return written
