"""Buriedness as a binding-site score feature — the first genuinely independent axis in a while.

Measured on analysis_v3 against the negatives that matter (competing hotspots, 77 known / 328
novel, within probe):

    volume                0.697
    buriedness            0.686        known sites are more enclosed: 53.8 vs 45.8 neighbours
    field_min_ball        0.681
    volume + buriedness   0.726        <- buriedness is worth +0.029

The reason it is worth adding is decorrelation, not its solo AUC:

    rho(volume, buriedness)     = +0.044      independent
    rho(volume, field_min_ball) = -0.948      the density-depth term is a size proxy

So affinity/field/volume all live on roughly one axis, and buriedness is a second one. It is also
aggregated as the **mean** over member hotspots rather than the max, because a best-of-members
summary is inflated by member count (rho -0.82 on this benchmark).
"""

import numpy as np
import pytest

from cosolvkit.analysis.core.field import buriedness


class TestBuriedness:

    def test_counts_protein_atoms_within_the_radius(self):
        """Both sides of the cutoff: 1.0 A counts, 50 A does not, and the radius is honoured."""
        prot = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        pt = np.array([[0.0, 0.0, 0.0]])
        assert buriedness(pt, prot, radius=8.0)[0] == 2
        assert buriedness(pt, prot, radius=0.5)[0] == 1
        assert buriedness(pt, prot, radius=60.0)[0] == 3

    def test_empty_protein_gives_zero_not_an_error(self):
        assert buriedness(np.array([[0.0, 0.0, 0.0]]), np.zeros((0, 3)))[0] == 0


class TestScoringIntegration:

    def test_site_takes_the_MEAN_over_members_not_the_max(self, make_hotspot):
        """A max would reintroduce the member-count inflation (rho -0.82)."""
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import _binding_site_feature_values
        a, b = make_hotspot(rank=1), make_hotspot(rank=2)
        a.add_property("buriedness", 40.0)
        b.add_property("buriedness", 60.0)
        site = BindingSite(site_id=1, member_hotspots=[a, b])
        assert _binding_site_feature_values([site])["buriedness"][0] == pytest.approx(50.0)

    def test_missing_buriedness_is_none_not_a_crash(self, make_hotspot):
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import (
            _binding_site_feature_values, score_binding_sites,
        )
        site = BindingSite(site_id=1, member_hotspots=[make_hotspot()])
        assert _binding_site_feature_values([site])["buriedness"][0] is None
        score_binding_sites([site])
        assert site.rank == 1

    def test_higher_buriedness_ranks_first_when_weighted(self, make_hotspot):
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import score_binding_sites
        lo, hi = make_hotspot(rank=1), make_hotspot(rank=2)
        lo.add_property("buriedness", 30.0)
        hi.add_property("buriedness", 70.0)
        s1 = BindingSite(site_id=1, member_hotspots=[lo], volume=100.0, agfe_min=-1.0)
        s2 = BindingSite(site_id=2, member_hotspots=[hi], volume=100.0, agfe_min=-1.0)
        score_binding_sites([s1, s2], weights={"buriedness": 5.0})
        assert s2.rank == 1
