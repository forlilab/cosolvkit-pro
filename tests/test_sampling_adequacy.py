"""How much of an AGFE map is real, and how much is counting noise?

AGFE is a Boltzmann inversion of an occupancy histogram, so its precision is set by how many
times a probe actually visited each voxel. On a real FosAKP benzene run (444 probe atoms,
1000 frames, 2,921,804 accessible voxels) the expected occupancy is **0.152 counts per voxel**,
one single visit already reads −1.12 kcal/mol, and the deepest voxel in the whole map is only
~30 visits. Nothing in the pipeline reported this, so there was no way to tell a real well from a
Poisson fluctuation.

These are the arithmetic pieces of that check. The headline they combine into: for benzene the
most favourable AGFE reachable by noise alone is more favourable than the −1 kT favourability
cutoff, so the cutoff sits *below* the noise floor. Water (11,924 oxygens, 27x better sampled)
comfortably clears it.

Definitions used throughout, with c = total counts in a voxel over the run:

    b     = N_o * n_frames = n_atoms * n_frames / n_accessible_voxels   (expected counts/voxel)
    AGFE  = -kT * ln(c / b)
    n_eff = (2*sqrt(pi)*sigma)^ndim                                     (Gaussian pooling)
"""

import math

import numpy as np
import pytest

from cosolvkit.analysis.core.sampling import (
    KT_300,
    agfe_from_counts,
    agfe_noise_sigma,
    counts_from_agfe,
    effective_pooled_voxels,
    expected_max_z,
    noise_floor_agfe,
    sampling_report,
)

# Real numbers from FosAKP benzene replica r0.
BEN = dict(n_atoms=444, n_frames=1000, n_accessible_voxels=2_921_804)
WATER = dict(n_atoms=11_924, n_frames=1000, n_accessible_voxels=2_921_804)
SIGMA_VOX = (1.4 / 3.0) / 0.5          # the pipeline's smoothing sigma, in voxels


def _bulk(d):
    from cosolvkit.analysis.core.sampling import expected_bulk_counts
    return expected_bulk_counts(**d)


class TestExpectedBulkCounts:

    def test_benzene_is_a_fifth_of_a_count_per_voxel(self):
        assert _bulk(BEN) == pytest.approx(0.15196, rel=1e-3)


class TestAgfeCountsConversion:

    def test_uniform_occupancy_is_zero_agfe(self):
        b = _bulk(BEN)
        assert agfe_from_counts(b, b) == pytest.approx(0.0, abs=1e-12)

    def test_one_single_visit_already_passes_the_1kt_cutoff(self):
        """The core problem: 1 visit reads -1.12, well past the -0.596 cutoff."""
        agfe = agfe_from_counts(1.0, _bulk(BEN))
        assert agfe == pytest.approx(-1.123, abs=5e-3)
        assert agfe < -KT_300

    def test_round_trip(self):
        """Inverse consistency, and the depth of the deepest FosAKP benzene voxel (~30 visits)."""
        b = _bulk(BEN)
        for c in (0.5, 1.0, 7.0, 30.0):
            assert counts_from_agfe(agfe_from_counts(c, b), b) == pytest.approx(c)

    def test_zero_counts_is_not_a_finite_agfe(self):
        assert not np.isfinite(agfe_from_counts(0.0, _bulk(BEN)))


class TestEffectivePooling:

    def test_pipeline_sigma_pools_about_36_voxels(self):
        assert effective_pooled_voxels(SIGMA_VOX) == pytest.approx(36.2, rel=0.02)

    def test_no_smoothing_pools_a_single_voxel(self):
        """sigma=0 would make the closed form collapse to 0 voxels; it is special-cased to 1."""
        assert effective_pooled_voxels(0.0) == pytest.approx(1.0)


class TestNoiseMagnitude:

    def test_sigma_is_kt_over_sqrt_counts(self):
        assert agfe_noise_sigma(100.0) == pytest.approx(KT_300 / 10.0)

    def test_benzene_pooled_noise_matches_the_measured_prominence(self):
        """Measured basin prominence was 0.11-0.15 kcal/mol; pooled noise should be that order."""
        pooled = _bulk(BEN) * effective_pooled_voxels(SIGMA_VOX)
        assert agfe_noise_sigma(pooled) == pytest.approx(0.25, abs=0.05)


class TestExpectedMaxZ:

    def test_sqrt_two_log_n_with_degenerate_counts_clamped_to_zero(self):
        """n<=1 has no log; it must clamp rather than return nan/negative."""
        assert expected_max_z(80_000) == pytest.approx(math.sqrt(2 * math.log(80_000)))
        assert expected_max_z(1) == 0.0
        assert expected_max_z(0) == 0.0


class TestNoiseFloor:

    def test_benzene_noise_floor_beats_the_favourability_cutoff(self):
        """The headline: noise alone reaches a more favourable AGFE than the -1 kT cutoff."""
        floor = noise_floor_agfe(**BEN, sigma_voxels=SIGMA_VOX)
        assert floor < 0
        assert floor < -KT_300, "cutoff sits below the noise floor for benzene"

    def test_water_noise_floor_does_not_reach_the_cutoff(self):
        floor = noise_floor_agfe(**WATER, sigma_voxels=SIGMA_VOX)
        assert -KT_300 < floor < 0, "water is well enough sampled for the cutoff to mean something"


class TestSamplingReport:

    def test_reports_the_quantities_and_the_verdict(self):
        r = sampling_report(**BEN, sigma_voxels=SIGMA_VOX, n_kt=1.0)
        assert r["bulk_counts_per_voxel"] == pytest.approx(0.15196, rel=1e-3)
        assert r["pooled_bulk_counts"] == pytest.approx(5.5, rel=0.05)
        assert r["agfe_of_one_visit"] == pytest.approx(-1.123, abs=5e-3)
        assert r["cutoff_agfe"] == pytest.approx(-KT_300)
        assert r["cutoff_below_noise_floor"] is True

    def test_a_stricter_cutoff_can_clear_the_noise_floor(self):
        """Actionable: the report should show that raising n_kt fixes the comparison."""
        loose = sampling_report(**BEN, sigma_voxels=SIGMA_VOX, n_kt=1.0)
        strict = sampling_report(**BEN, sigma_voxels=SIGMA_VOX, n_kt=4.0)
        assert loose["cutoff_below_noise_floor"] is True
        assert strict["cutoff_below_noise_floor"] is False
