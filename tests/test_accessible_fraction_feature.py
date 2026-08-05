"""``accessible_fraction`` as a binding-site score feature — the enclosure axis.

Replaces ``tests/test_buriedness_feature.py``. ``buriedness`` was a raw count of protein heavy
atoms in a ball: unbounded, and its AUC climbed monotonically with radius on FosAKP (0.524 at
4 A to 0.806 at 20 A) without ever flattening, so at large radius it reported how *central* a
point is rather than how *enclosed*. ``accessible_fraction`` is bounded in [0, 1] and plateaus
with radius (0.741 / 0.780 / 0.794 / 0.823 / 0.818 at R = 8 / 10 / 12 / 16 / 20), which is what
an enclosure measure should do. On matched 250 ns data it scored 0.724 against buriedness 0.667.

Lower means more enclosed, so the feature is scored inverted (``_BS_INVERTED_FEATURES``), and it
is aggregated as the **mean** over member hotspots rather than the max, because a best-of-members
summary is inflated by member count (rho -0.82 on this benchmark).
"""

import numpy as np
import pytest

from cosolvkit.analysis.core.field import accessible_fraction

DELTA = np.array([1.0, 1.0, 1.0])
ORIGIN = np.array([0.0, 0.0, 0.0])


class TestAccessibleFraction:

    def test_fully_accessible_ball_is_one_and_fully_blocked_is_zero(self):
        pt = np.array([[5.0, 5.0, 5.0]])
        open_mask = np.ones((11, 11, 11), dtype=bool)
        shut_mask = np.zeros((11, 11, 11), dtype=bool)
        assert accessible_fraction(pt, open_mask, ORIGIN, DELTA, radius=3.0)[0] == 1.0
        assert accessible_fraction(pt, shut_mask, ORIGIN, DELTA, radius=3.0)[0] == 0.0

    def test_is_bounded_in_zero_one(self):
        rng = np.random.default_rng(0)
        mask = rng.random((15, 15, 15)) > 0.5
        pts = np.array([[7.0, 7.0, 7.0], [3.0, 9.0, 5.0]])
        v = accessible_fraction(pts, mask, ORIGIN, DELTA, radius=4.0)
        assert np.all((v >= 0.0) & (v <= 1.0))

    def test_an_enclosed_point_scores_below_an_exposed_one(self):
        """The discriminating behaviour: half-blocked neighbourhood < open neighbourhood."""
        mask = np.ones((21, 21, 21), dtype=bool)
        mask[:11, :, :] = False           # wall on one side
        enclosed = accessible_fraction(np.array([[11.0, 10.0, 10.0]]), mask,
                                       ORIGIN, DELTA, radius=5.0)[0]
        exposed = accessible_fraction(np.array([[17.0, 10.0, 10.0]]), mask,
                                      ORIGIN, DELTA, radius=5.0)[0]
        assert enclosed < exposed

    def test_origin_is_the_centre_of_voxel_zero_so_the_inverse_map_rounds(self):
        """A point 0.4 A off a voxel centre must land on that voxel, not its neighbour."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        v = accessible_fraction(np.array([[2.4, 2.0, 2.0]]), mask, ORIGIN, DELTA, radius=0.1)
        assert v[0] == 1.0

    def test_point_outside_the_grid_is_nan_not_an_error(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        v = accessible_fraction(np.array([[500.0, 500.0, 500.0]]), mask,
                                ORIGIN, DELTA, radius=2.0)
        assert np.isnan(v[0])


class TestScoringIntegration:

    def test_site_takes_the_MEAN_over_members_not_the_max(self, make_hotspot):
        """A max would reintroduce the member-count inflation (rho -0.82)."""
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import _binding_site_feature_values
        a, b = make_hotspot(rank=1), make_hotspot(rank=2)
        a.add_property("accessible_fraction", 0.40)
        b.add_property("accessible_fraction", 0.60)
        site = BindingSite(site_id=1, member_hotspots=[a, b])
        got = _binding_site_feature_values([site])["accessible_fraction"][0]
        assert got == pytest.approx(0.50)

    def test_missing_value_is_none_not_a_crash(self, make_hotspot):
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import (
            _binding_site_feature_values, score_binding_sites,
        )
        site = BindingSite(site_id=1, member_hotspots=[make_hotspot()])
        assert _binding_site_feature_values([site])["accessible_fraction"][0] is None
        score_binding_sites([site])
        assert site.rank == 1

    def test_more_enclosed_ranks_first_when_weighted(self, make_hotspot):
        """Inverted feature: the LOWER accessible_fraction must win."""
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import score_binding_sites
        enclosed, exposed = make_hotspot(rank=1), make_hotspot(rank=2)
        enclosed.add_property("accessible_fraction", 0.30)
        exposed.add_property("accessible_fraction", 0.70)
        s1 = BindingSite(site_id=1, member_hotspots=[enclosed], volume=100.0, agfe_min=-1.0)
        s2 = BindingSite(site_id=2, member_hotspots=[exposed], volume=100.0, agfe_min=-1.0)
        score_binding_sites([s1, s2], weights={"accessible_fraction": 5.0})
        assert s1.rank == 1

    def test_retired_buriedness_weight_is_rejected(self, make_hotspot):
        """`buriedness` was removed; naming it now raises rather than being dropped silently.

        It is deliberately NOT aliased to `accessible_fraction`: the replacement measures a
        different quantity, on a different scale, with the opposite sign, so a config carrying the
        old name has to be ported deliberately rather than reinterpreted.
        """
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import score_binding_sites
        site = BindingSite(site_id=1, member_hotspots=[make_hotspot()],
                           volume=100.0, agfe_min=-1.0)
        with pytest.raises(ValueError, match="buriedness"):
            score_binding_sites([site], weights={"buriedness": 5.0})
