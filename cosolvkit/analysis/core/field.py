"""Descriptors read from the AGFE map as a continuous FIELD, at a point.

Nothing here depends on a cutoff or a clustering, so the values cannot move when those
settings change. These are a *scoring* substrate, not a *detector*: they score points the
pipeline already proposed.
"""

import numpy as np

# Sampling radii in Angstroms.
_MIN_BALL_R = 3.0
_CONTRAST_R, _CONTRAST_SHELL = 3.0, 2.0
_SHARP_R, _SHARP_SHELL_LO, _SHARP_SHELL_HI = 1.5, 2.0, 3.0

FIELD_DESCRIPTORS = ("field_min_ball", "field_mean_ball", "field_contrast",
                     "field_sharpness")


def _ball(arr, origin, delta, xyz, radius):
    """(values, distances) for voxels within *radius* of *xyz*; (None, None) if off-grid."""
    arr = np.asarray(arr)
    origin = np.asarray(origin, dtype=float)
    delta = np.asarray(delta, dtype=float)
    c = np.rint((np.asarray(xyz, dtype=float) - origin) / delta).astype(int)
    if np.any(c < 0) or np.any(c >= np.array(arr.shape)):
        return None, None
    rv = np.ceil(radius / delta).astype(int)
    lo = np.maximum(c - rv, 0)
    hi = np.minimum(c + rv + 1, np.array(arr.shape))
    if np.any(lo >= hi):
        return None, None
    sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    gi = np.indices(sub.shape)
    off = np.stack([(lo[d] + gi[d]) * delta[d] + origin[d] - float(xyz[d])
                    for d in range(3)])
    return sub, np.sqrt((off ** 2).sum(axis=0))


def field_descriptors(agfe, origin, delta, xyz):
    """Field descriptors at *xyz*, as ``{name: float or None}``.

    ``None`` for every descriptor when the point lies outside the grid.
    """
    out = {name: None for name in FIELD_DESCRIPTORS}

    sub, dist = _ball(agfe, origin, delta, xyz, _MIN_BALL_R)
    if sub is None:
        return out
    m = dist <= _MIN_BALL_R
    if m.any():
        out["field_min_ball"] = float(sub[m].min())
        out["field_mean_ball"] = float(sub[m].mean())

    sub, dist = _ball(agfe, origin, delta, xyz, _CONTRAST_R + _CONTRAST_SHELL)
    if sub is not None:
        inner = dist <= _CONTRAST_R
        outer = (dist > _CONTRAST_R) & (dist <= _CONTRAST_R + _CONTRAST_SHELL)
        if inner.any() and outer.any():
            out["field_contrast"] = float(sub[inner].min() - sub[outer].mean())

    sub, dist = _ball(agfe, origin, delta, xyz, _SHARP_SHELL_HI)
    if sub is not None:
        core = dist <= _SHARP_R
        shell = (dist > _SHARP_SHELL_LO) & (dist <= _SHARP_SHELL_HI)
        if core.any() and shell.any():
            out["field_sharpness"] = float(
                (sub[shell].mean() - sub[core].min()) / (_SHARP_R ** 2))
    return out


def accessible_fraction(points, mask, origin, delta, radius=10.0):
    """Fraction of voxels within *radius* of each point that are solvent-accessible.

    The enclosure measure. It replaces a removed `buriedness`, which was a raw count of
    protein heavy atoms in a ball -- unbounded, and its
    value depends entirely on the radius: measured on FosAKP its AUC climbs monotonically from
    0.524 at 4 A to 0.806 at 20 A without ever flattening, and at 20 A it counts a quarter of
    the protein — at which point it reports how CENTRAL a point is, not how enclosed. This
    quantity is bounded in [0, 1] and **plateaus** with radius (0.741 / 0.780 / 0.794 / 0.823 /
    0.818 at R = 8 / 10 / 12 / 16 / 20), which is the behaviour an enclosure measure should have.

    Lower means more enclosed, so it is scored inverted (see `_BS_INVERTED_FEATURES`).

    *mask* is the boolean solvent-accessible volume from
    :meth:`GridAnalysis._build_accessible_mask` — note that this is an EMPIRICAL accessible
    volume (voxels water oxygens and cosolvent heavy atoms were observed to visit, dilated by a
    probe radius, minus protein-occupied voxels), not an analytic SASA. One consequence: a pocket
    solvent rarely entered reads as inaccessible for want of sampling rather than want of space.

    *origin* is the centre of voxel 0, matching gridData and
    ``HotspotDetector._voxel_to_angstrom``, so the inverse map rounds rather than floors.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    mask = np.asarray(mask, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    delta = np.asarray(delta, dtype=float)
    shape = np.asarray(mask.shape)

    rv = np.ceil(radius / delta).astype(int)
    ranges = [np.arange(-rv[d], rv[d] + 1) for d in range(3)]
    gi, gj, gk = np.meshgrid(*ranges, indexing="ij")
    within = (np.sqrt((gi * delta[0]) ** 2 + (gj * delta[1]) ** 2
                      + (gk * delta[2]) ** 2) <= radius)
    offs = np.stack([gi[within], gj[within], gk[within]], axis=1)

    out = np.full(len(pts), np.nan)
    idx = np.rint((pts - origin) / delta).astype(int)
    for i, c in enumerate(idx):
        v = c + offs
        ok = np.all((v >= 0) & (v < shape), axis=1)
        if ok.any():
            v = v[ok]
            out[i] = mask[v[:, 0], v[:, 1], v[:, 2]].mean()
    return out


def attach_accessible_fraction(hotspots, mask, origin, delta, radius=16.0):
    """Attach ``accessible_fraction`` at each hotspot centroid. Returns the count attached.

    R = 16 A, chosen for **replica robustness**, not for the best central AUC. The mask is
    occupancy-derived, so it differs between replicas of the same box, and that propagates into
    the feature. Measured on five benzene replicas (own-probe masks, 13 usable probes):

        R (A)     8      10      12      16      20
        worst  0.655   0.696   0.727   0.775   0.775
        spread 0.086   0.085   0.067   0.048   0.043

    At R = 10 the worst replica (0.696) falls BELOW the deterministic `buriedness` this feature
    replaced (0.744) — so at that radius the advantage is inside replica noise. At R = 16 even the
    worst replica clears it and the spread halves, because a larger ball averages over more voxels
    and mask noise cancels. An earlier version of this function defaulted to 10 A on the grounds
    that it is where the measure has a physical length scale (the mean fraction is non-monotone in
    R, bottoming out at 8-10 A: small balls sit in the solvent void, intermediate balls capture the
    protein wall, large balls reach back into bulk). That reasoning is sound but loses to
    reproducibility here.

    Two further findings worth knowing when supplying *mask*:
      * the cosolvent term contributes almost nothing — a water-only mask has Jaccard **0.997**
        with the water+cosolvent mask and scores the same (0.778 vs 0.780). Prefer water-only: it
        is probe-independent, so ONE mask per target is legitimate, and it removes the coupling
        whereby a probe's own mask marks exactly where that probe went.
      * probe-to-probe mask choice barely matters (each probe scored with its own mask 0.786 vs
        all scored with benzene's 0.780; Jaccard median 0.825). **Replica** choice matters far
        more than probe choice, so pool replicas when building the mask if you can.
    """
    n = 0
    for hs in hotspots:
        if getattr(hs, "centroid", None) is None:
            continue
        v = accessible_fraction(hs.centroid, mask, origin, delta, radius)[0]
        if np.isfinite(v):
            hs.add_property("accessible_fraction", float(v))
            n += 1
    return n


def attach_field_descriptors(hotspots, agfe, origin, delta):
    """Compute and attach field descriptors at each hotspot centroid. Returns the count."""
    n = 0
    for hs in hotspots:
        if getattr(hs, "centroid", None) is None:
            continue
        for name, value in field_descriptors(agfe, origin, delta, hs.centroid).items():
            hs.add_property(name, value)
        n += 1
    return n
