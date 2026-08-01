"""Field descriptors sampled from the AGFE map, and their use as binding-site score features.

Measured on FosAKP at fixed query points against buriedness-matched decoys, field descriptors
reach AUC 0.805 while the thresholded-blob descriptors sit at 0.50-0.70 — so the field is the
better scoring substrate even though it is *not* a better detector (a field-identification
prototype tied or lost on recall@top5 against the existing pipeline).

They enter as opt-in weights (default 0.0). The 0.805 was measured at crystallographic ligand
positions, not at hotspot centroids, and a hotspot centroid sits a median 1.4 A from the ligand it
represents — so a non-zero default would change every ranking on the strength of a measurement
made somewhere else. Enabling them is a weight sweep, not an assumption.
"""

import numpy as np
import pytest

from cosolvkit.analysis.core.field import FIELD_DESCRIPTORS, field_descriptors


def _well(shape=(41, 41, 41), depth=-3.0, sigma=3.0):
    """A single smooth Gaussian well centred in the box, on a 0.5 A grid."""
    c = np.array([(s - 1) / 2 for s in shape])
    idx = np.indices(shape).astype(float)
    r2 = sum((idx[d] - c[d]) ** 2 for d in range(3))
    return depth * np.exp(-r2 / (2 * sigma ** 2))


ORIGIN = np.zeros(3)
DELTA = np.full(3, 0.5)


def _centre_xyz(arr):
    return (np.array([(s - 1) / 2 for s in arr.shape]) * DELTA) + ORIGIN


class TestDescriptorValues:

    def test_returns_the_documented_set(self):
        arr = _well()
        d = field_descriptors(arr, ORIGIN, DELTA, _centre_xyz(arr))
        assert set(d) == set(FIELD_DESCRIPTORS)

    def test_min_ball_finds_the_well_bottom(self):
        arr = _well(depth=-3.0)
        d = field_descriptors(arr, ORIGIN, DELTA, _centre_xyz(arr))
        assert d["field_min_ball"] == pytest.approx(-3.0, abs=0.05)

    def test_contrast_is_negative_at_a_well(self):
        """Inside is more favourable than the surrounding shell."""
        arr = _well()
        d = field_descriptors(arr, ORIGIN, DELTA, _centre_xyz(arr))
        assert d["field_contrast"] < 0

    def test_sharpness_is_positive_at_a_well_and_zero_on_a_flat_field(self):
        arr = _well()
        sharp = field_descriptors(arr, ORIGIN, DELTA, _centre_xyz(arr))["field_sharpness"]
        flat = field_descriptors(np.zeros((41, 41, 41)), ORIGIN, DELTA,
                                 _centre_xyz(arr))["field_sharpness"]
        assert sharp > 0
        assert flat == pytest.approx(0.0, abs=1e-9)

    def test_a_deeper_well_scores_stronger_on_every_descriptor(self):
        shallow = field_descriptors(_well(depth=-1.0), ORIGIN, DELTA, _centre_xyz(_well()))
        deep = field_descriptors(_well(depth=-3.0), ORIGIN, DELTA, _centre_xyz(_well()))
        assert deep["field_min_ball"] < shallow["field_min_ball"]
        assert deep["field_contrast"] < shallow["field_contrast"]
        assert deep["field_sharpness"] > shallow["field_sharpness"]

    def test_a_narrow_well_is_sharper_than_a_broad_one_of_equal_depth(self):
        narrow = field_descriptors(_well(sigma=1.5), ORIGIN, DELTA, _centre_xyz(_well()))
        broad = field_descriptors(_well(sigma=6.0), ORIGIN, DELTA, _centre_xyz(_well()))
        assert narrow["field_sharpness"] > broad["field_sharpness"]

    def test_a_point_outside_the_grid_yields_nones_rather_than_raising(self):
        arr = _well()
        d = field_descriptors(arr, ORIGIN, DELTA, np.array([-500.0, -500.0, -500.0]))
        assert all(v is None for v in d.values())


class TestScoringIntegration:

    def test_new_features_are_weightable_and_default_to_zero(self):
        from cosolvkit.analysis.core.scoring import (
            DEFAULT_BINDING_SITE_WEIGHTS, normalize_weights,
        )
        for name in ("field_contrast", "field_sharpness"):
            assert DEFAULT_BINDING_SITE_WEIGHTS[name] == 0.0
            assert normalize_weights({name: 2.0})[name] == 2.0

    def test_site_takes_the_best_field_value_over_its_member_hotspots(self, make_hotspot):
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import _binding_site_feature_values
        a, b = make_hotspot(rank=1), make_hotspot(rank=2)
        a.add_property("field_contrast", -0.4)
        b.add_property("field_contrast", -1.9)          # better
        a.add_property("field_sharpness", 0.10)
        b.add_property("field_sharpness", 0.75)         # better
        site = BindingSite(site_id=1, member_hotspots=[a, b])
        vals = _binding_site_feature_values([site])
        assert vals["field_contrast"][0] == pytest.approx(-1.9)
        assert vals["field_sharpness"][0] == pytest.approx(0.75)

    def test_missing_field_properties_give_none_not_a_crash(self, make_hotspot):
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import (
            _binding_site_feature_values, score_binding_sites,
        )
        site = BindingSite(site_id=1, member_hotspots=[make_hotspot()])
        vals = _binding_site_feature_values([site])
        assert vals["field_contrast"][0] is None
        score_binding_sites([site])                     # must not raise
        assert site.rank == 1

    def test_contrast_is_treated_as_lower_is_better(self):
        """field_contrast is negative at a real well, so it must be inverted like affinity."""
        from cosolvkit.analysis.core.scoring import _BS_INVERTED_FEATURES
        assert "field_contrast" in _BS_INVERTED_FEATURES
        assert "field_sharpness" not in _BS_INVERTED_FEATURES

    def test_enabling_the_weight_changes_the_ranking(self, make_hotspot):
        """The knob must actually do something — a silently inert weight is worse than none."""
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import score_binding_sites
        weak, strong = make_hotspot(rank=1), make_hotspot(rank=2)
        weak.add_property("field_contrast", -0.1)
        strong.add_property("field_contrast", -2.0)
        s1 = BindingSite(site_id=1, member_hotspots=[weak], volume=100.0, agfe_min=-1.0)
        s2 = BindingSite(site_id=2, member_hotspots=[strong], volume=100.0, agfe_min=-1.0)
        score_binding_sites([s1, s2], weights={"field_contrast": 5.0})
        assert s2.rank == 1, "the site with the stronger field contrast should rank first"
