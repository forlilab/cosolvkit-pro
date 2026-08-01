"""Descriptors read from the AGFE map as a continuous FIELD, at a point.

Measured against crystallographic ground truth on FosAKP, descriptors sampled this way reach
AUC 0.805 at fixed query points, while descriptors of the thresholded-and-clustered blob sit at
0.50-0.70 and swing across segmentation settings by as much as their whole effect. Nothing here
depends on a cutoff or a clustering, so the values cannot move when those settings change.

Field descriptors are the better *scoring* substrate but not a better *detector*: a prototype
that generated candidates from field minima instead of segmentation tied or lost on recall@top5.
So these are meant to score points the pipeline already proposed.
"""

import numpy as np

# Sampling radii in Angstroms. Chosen to match the offline evaluation that produced the AUCs
# above, so pipeline values are directly comparable to it.
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

    ``None`` for every descriptor when the point lies outside the grid, so a caller can attach
    the result unconditionally and let the scorer treat missing values as no contribution.
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


def buriedness(points, protein_xyz, radius=8.0):
    """Protein heavy atoms within *radius* of each point — a simple enclosure proxy.

    Worth having because it is **independent** of the density terms, not because it is strong on
    its own: measured against competing hotspots, buriedness scores 0.686 and volume 0.697, but
    rho(volume, buriedness) = +0.044 while rho(volume, field_min_ball) = -0.948. Combining volume
    and buriedness reaches 0.726; adding the field term does not help.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    prot = np.asarray(protein_xyz, dtype=float)
    if prot.size == 0:
        return np.zeros(len(pts))
    out = np.empty(len(pts))
    for i, q in enumerate(pts):
        out[i] = int((np.sqrt(((prot - q) ** 2).sum(axis=1)) <= radius).sum())
    return out


def attach_buriedness(hotspots, protein_xyz, radius=8.0):
    """Attach ``buriedness`` at each hotspot centroid. Returns the count attached."""
    n = 0
    for hs in hotspots:
        if getattr(hs, "centroid", None) is None:
            continue
        hs.add_property("buriedness", float(buriedness(hs.centroid, protein_xyz, radius)[0]))
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
