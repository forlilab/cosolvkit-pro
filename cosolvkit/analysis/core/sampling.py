"""
Sampling adequacy of an AGFE map.
AGFE is ``-kT * ln(c / b)`` with ``c`` the counts in a voxel and ``b`` the expected counts if
occupancy were uniform. For a dilute cosolvent ``b`` is a small fraction of one count, so a
single visit is already a deep-looking well and the map's precision is set by Poisson statistics
rather than by trajectory length. :func:`sampling_report` makes that comparison explicit.
"""

import math

BOLTZMANN_CONSTANT_KB = 0.0019872041      # kcal/(mol*K)
KT_300 = BOLTZMANN_CONSTANT_KB * 300.0    # 0.596 kcal/mol


def expected_bulk_counts(n_atoms, n_frames, n_accessible_voxels):
    """Counts a voxel would collect over the whole run at uniform occupancy."""
    if n_accessible_voxels <= 0:
        raise ValueError("n_accessible_voxels must be positive")
    return float(n_atoms) * float(n_frames) / float(n_accessible_voxels)


def agfe_from_counts(counts, bulk_counts, temperature=300.0):
    """AGFE of a voxel holding *counts*, given the bulk expectation. -inf counts -> +inf."""
    kt = BOLTZMANN_CONSTANT_KB * temperature
    if counts <= 0:
        return float("inf")
    return -kt * math.log(float(counts) / float(bulk_counts))


def counts_from_agfe(agfe, bulk_counts, temperature=300.0):
    """Counts implied by an AGFE value — the inverse of :func:`agfe_from_counts`."""
    kt = BOLTZMANN_CONSTANT_KB * temperature
    return float(bulk_counts) * math.exp(-float(agfe) / kt)


def effective_pooled_voxels(sigma_voxels, ndim=3):
    """Independent voxels averaged by a Gaussian kernel of width *sigma_voxels*.

    ``1 / sum(w_i^2)`` for normalised Gaussian weights, which is ``(2*sqrt(pi)*sigma)^ndim``.
    """
    if sigma_voxels <= 0:
        return 1.0
    return float((2.0 * math.sqrt(math.pi) * float(sigma_voxels)) ** ndim)


def agfe_noise_sigma(pooled_counts, temperature=300.0):
    """Poisson uncertainty on AGFE, ``kT / sqrt(counts)``.

    Approximate: the pipeline smooths the AGFE field rather than the histogram, so this treats
    the smoothed value as an average of *pooled_counts* independent observations.
    """
    kt = BOLTZMANN_CONSTANT_KB * temperature
    if pooled_counts <= 0:
        return float("inf")
    return kt / math.sqrt(float(pooled_counts))


def expected_max_z(n_samples):
    """Expected maximum of *n_samples* standard normals, ``sqrt(2 ln n)``."""
    if n_samples <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(float(n_samples)))


def noise_floor_agfe(n_atoms, n_frames, n_accessible_voxels,
                     sigma_voxels, temperature=300.0):
    """Most favourable AGFE reachable by Poisson noise alone.

    Takes the best of ``n_accessible_voxels / pooled`` independent patches, each holding
    ``pooled * b`` counts on average. Any well shallower than this is indistinguishable from a
    fluctuation.
    """
    b = expected_bulk_counts(n_atoms, n_frames, n_accessible_voxels)
    pooled = effective_pooled_voxels(sigma_voxels)
    pooled_bulk = b * pooled
    if pooled_bulk <= 0:
        return float("-inf")
    n_patches = max(1.0, float(n_accessible_voxels) / pooled)
    best = pooled_bulk + expected_max_z(n_patches) * math.sqrt(pooled_bulk)
    return agfe_from_counts(best / pooled, b, temperature)


def sampling_report(n_atoms, n_frames, n_accessible_voxels, sigma_voxels,
                    n_kt=1.0, temperature=300.0):
    """Sampling adequacy of one map, as a dict plus a one-line summary.

    ``cutoff_below_noise_floor`` is the verdict: True means the favourability cutoff selects
    voxels that noise alone could produce, so the favourable set is not trustworthy.
    """
    kt = BOLTZMANN_CONSTANT_KB * temperature
    b = expected_bulk_counts(n_atoms, n_frames, n_accessible_voxels)
    pooled = effective_pooled_voxels(sigma_voxels)
    pooled_bulk = b * pooled
    cutoff = -float(n_kt) * kt
    floor = noise_floor_agfe(n_atoms, n_frames, n_accessible_voxels,
                             sigma_voxels, temperature)
    below = bool(cutoff > floor)     # both negative; "above the floor" means less favourable
    summary = (
        f"sampling: {b:.3f} counts/voxel bulk ({pooled_bulk:.1f} pooled over {pooled:.0f} "
        f"voxels), 1 visit = {agfe_from_counts(1.0, b, temperature):+.2f} kcal/mol, "
        f"noise floor {floor:+.2f} kcal/mol vs cutoff {cutoff:+.2f} kcal/mol"
        + ("  [CUTOFF BELOW NOISE FLOOR — favourable voxels are not trustworthy]"
           if below else "  [ok]")
    )
    return {
        "bulk_counts_per_voxel": b,
        "pooled_voxels": pooled,
        "pooled_bulk_counts": pooled_bulk,
        "agfe_of_one_visit": agfe_from_counts(1.0, b, temperature),
        "counts_at_cutoff": counts_from_agfe(cutoff, b, temperature),
        "agfe_noise_sigma": agfe_noise_sigma(pooled_bulk, temperature),
        "noise_floor_agfe": floor,
        "cutoff_agfe": cutoff,
        "cutoff_below_noise_floor": below,
        "summary": summary,
    }
