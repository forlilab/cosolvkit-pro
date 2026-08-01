"""Smooth the occupancy, not the free energy — which removes the 1e-10 floor artifact.

The old order was: Boltzmann-invert the raw histogram, then Gaussian-filter the resulting energy
field. Two things go wrong.

* ``log(0)`` is undefined, so the code floored the local probability at ``N = 1e-10``. For
  FosAKP benzene that maps every unvisited voxel to **+8.49 kcal/mol** — a number with no
  physical meaning, chosen only to keep the logarithm finite.
* The Gaussian filter then ran over a field whose background *was* that arbitrary constant, so
  it mixed +8.49 into every neighbourhood. Which voxels survived as "favourable" therefore
  depended on the choice of epsilon: moving the floor 1e-10 -> 1e-8 shifts the background
  8.49 -> 5.74 and changes the answer.

Smoothing the histogram first fixes both, and is the physically correct order anyway:

* the kernel width is ``atom_radius / 3``, i.e. it represents the physical size of an atom —
  that is a statement about where density is, not about energy;
* density averages linearly, and the log of a locally-averaged density is a well-defined free
  energy, whereas an average of logs is not;
* no floor is needed to keep the log finite, only a *resolution limit*: a single observation
  spread by the kernel peaks at ``1/((2*pi)^(d/2) sigma^d)`` counts, and anything below that is
  less than one observation's worth. That limit is derived from the kernel, not invented.

The behavioural payoff, tested below: an isolated single visit becomes **unfavourable**, because
one count spread over ~36 voxels is below bulk density. So the -1 kT cutoff stops selecting every
voxel a probe ever touched and starts requiring ~5 coincident visits.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from cosolvkit.analysis.core.grid import (
    BOLTZMANN_CONSTANT_KB,
    _detection_floor_counts,
    _grid_free_energy,
    _grid_free_energy_density_smoothed,
)

KT = BOLTZMANN_CONSTANT_KB * 300.0

# Real FosAKP benzene numbers.
N_ATOMS, N_FRAMES, N_ACC = 444, 1000, 2_921_804
SIGMA = (1.4 / 3.0) / 0.5          # 0.9333 voxels
SHAPE = (25, 25, 25)
MID = (12, 12, 12)


def _agfe(hist, sigma=SIGMA):
    return _grid_free_energy_density_smoothed(hist, N_ATOMS, N_FRAMES, N_ACC, sigma, 300.0)


class TestDetectionFloor:

    def test_matches_the_gaussian_peak_of_a_single_count(self):
        expected = 1.0 / ((2 * np.pi) ** 1.5 * SIGMA ** 3)
        assert _detection_floor_counts(SIGMA) == pytest.approx(expected)

    def test_a_single_count_really_does_peak_at_the_floor(self):
        """Empirical check of the formula against scipy's own kernel."""
        h = np.zeros(SHAPE)
        h[MID] = 1.0
        peak = gaussian_filter(h, sigma=SIGMA, mode="constant", cval=0.0).max()
        assert peak == pytest.approx(_detection_floor_counts(SIGMA), rel=0.02)

    def test_no_smoothing_means_the_limit_is_one_count(self):
        assert _detection_floor_counts(0.0) == 1.0

    def test_wider_kernel_has_a_lower_limit(self):
        assert _detection_floor_counts(2 * SIGMA) < _detection_floor_counts(SIGMA)


class TestBackgroundIsBoundedAndDerived:

    def test_unvisited_background_is_not_the_old_8_5(self):
        agfe = _agfe(np.zeros(SHAPE))
        assert agfe.max() < 1.0, "background must be a resolution limit, not +8.5 kcal/mol"

    def test_background_equals_the_detection_limit_value(self):
        agfe = _agfe(np.zeros(SHAPE))
        floor = _detection_floor_counts(SIGMA)
        expected = -KT * np.log((floor / N_FRAMES) / (N_ATOMS / N_ACC))
        assert agfe.max() == pytest.approx(expected)

    def test_background_moves_with_sigma_not_with_any_epsilon(self):
        """The cap is a property of the kernel, so changing the kernel must change it.

        A wider kernel has a LOWER detection limit (fewer counts are resolvable), so its cap sits
        further from bulk: +0.40 at sigma=0.93 vs +1.64 at sigma=1.87. Both remain far below the
        legacy +8.49, and both are derived rather than chosen.
        """
        a = _agfe(np.zeros(SHAPE), sigma=SIGMA).max()
        b = _agfe(np.zeros(SHAPE), sigma=2 * SIGMA).max()
        assert a != pytest.approx(b)
        assert b > a, "a wider kernel resolves less, so its cap is further from bulk"
        legacy = _grid_free_energy(np.zeros(SHAPE), N_ATOMS, N_FRAMES, N_ACC, 300.0).max()
        assert a < legacy / 5 and b < legacy / 4

    def test_the_old_path_still_shows_the_artifact(self):
        """Regression guard: the legacy estimator is unchanged, so old results reproduce."""
        legacy = _grid_free_energy(np.zeros(SHAPE), N_ATOMS, N_FRAMES, N_ACC, 300.0)
        assert legacy.max() > 8.0, "legacy 1e-10 floor should still give ~+8.5"


class TestShotNoiseIsFiltered:

    def test_an_isolated_single_visit_is_not_favourable(self):
        """The whole point. One visit used to read -1.12 kcal/mol and pass the -1 kT cutoff."""
        h = np.zeros(SHAPE)
        h[MID] = 1.0
        assert _agfe(h)[MID] > 0.0

    def test_an_isolated_pair_of_visits_is_still_not_favourable(self):
        h = np.zeros(SHAPE)
        h[MID] = 2.0
        assert _agfe(h)[MID] > -KT

    def test_a_real_hotspot_is_still_favourable(self):
        """30 coincident visits — the depth of a genuine FosAKP hotspot."""
        h = np.zeros(SHAPE)
        h[MID] = 30.0
        assert _agfe(h)[MID] < -KT

    def test_the_cutoff_now_needs_roughly_five_coincident_visits(self):
        """Documents the new sensitivity: ~5 counts in a kernel, not 1 anywhere."""
        passing = []
        for k in range(1, 12):
            h = np.zeros(SHAPE)
            h[MID] = float(k)
            passing.append(_agfe(h)[MID] < -KT)
        first = passing.index(True) + 1
        assert 4 <= first <= 7, f"cutoff first passed at {first} visits"

    def test_spatially_spread_counts_beat_the_same_counts_scattered(self):
        """A coincidence filter: clustered visits should read deeper than dispersed ones."""
        clustered = np.zeros(SHAPE)
        clustered[11:14, 11:14, 11:14] = 30.0 / 27
        scattered = np.zeros(SHAPE)
        rng = np.random.default_rng(0)
        idx = rng.choice(np.arange(5, 20), size=(27, 3))
        for i, j, k in idx:
            scattered[i, j, k] += 30.0 / 27
        assert _agfe(clustered).min() < _agfe(scattered).min()


class TestEstimatorInvariants:

    def test_uniform_occupancy_still_gives_zero(self):
        b = N_ATOMS * N_FRAMES / N_ACC          # expected counts per voxel
        h = np.full(SHAPE, b)
        interior = _agfe(h)[5:-5, 5:-5, 5:-5]
        np.testing.assert_allclose(interior, 0.0, atol=1e-6)

    def test_smoothing_the_density_conserves_total_counts(self):
        """Linear in the observable — unlike smoothing a log field."""
        h = np.zeros(SHAPE)
        h[10:15, 10:15, 10:15] = 3.0
        sm = gaussian_filter(h, sigma=SIGMA, mode="constant", cval=0.0)
        assert sm.sum() == pytest.approx(h.sum(), rel=1e-6)

    def test_enrichment_is_still_negative_and_depletion_positive(self):
        b = N_ATOMS * N_FRAMES / N_ACC
        rich = np.full(SHAPE, b * 20)
        poor = np.full(SHAPE, b / 20)
        assert _agfe(rich)[MID] < 0 < _agfe(poor)[MID]

    def test_output_is_finite_everywhere(self):
        h = np.zeros(SHAPE)
        h[MID] = 7.0
        assert np.all(np.isfinite(_agfe(h)))
