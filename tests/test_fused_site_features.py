"""Binding-site features must be count-normalised fusions, not best-of-members maxima.

Measured on the FosAKP combined binding sites (33 sites, 1-35 hotspots each):

    rho(n_hotspots, agfe_min)       = -0.820      <- "best over members", lower is better
    rho(n_hotspots, probe_coverage) = +0.978

So 82% of the affinity term and essentially all of probe_coverage are member count. It is pure
extreme-value statistics, not quality: drawing best-of-n from ONE distribution with no quality
difference gives mean -0.57 at n=2, -1.17 at n=5, -1.87 at n=20. Since `affinity` carries weight
3.0 and `probe_coverage` 2.0, five of nine weight units were measuring how many hotspots landed
somewhere — and member count is anti-predictive for the sites that matter (the 23-fragment main
pocket is hit by 4 probes, a 6-fragment site by 13).

The fix is to sample the field at the site's own representative point across a FIXED probe set —
every probe, not just the members — and fuse across probes. The denominator is then constant, so
there is no best-of-n inflation, and it is the regime that measured +0.25 AUC in this project.
"""

import numpy as np
import pytest

from cosolvkit.analysis.core.site_features import (
    ProbeFieldSampler,
    fused_site_features,
)


def _map(depth, centre=(10.0, 10.0, 10.0), shape=(41, 41, 41), gridsize=0.5, sigma=3.0):
    """A Gaussian well of given depth centred at *centre* (Angstroms)."""
    origin = np.zeros(3)
    delta = np.full(3, gridsize)
    c = (np.asarray(centre) - origin) / delta
    idx = np.indices(shape).astype(float)
    r2 = sum((idx[d] - c[d]) ** 2 for d in range(3))
    return depth * np.exp(-r2 / (2 * sigma ** 2)), origin, delta


class _Site:
    """Minimal stand-in: the fusion only needs a point and (for the bias test) members."""

    def __init__(self, site_id, centroid, n_members=1):
        self.site_id = site_id
        self.centroid = np.asarray(centroid, dtype=float)
        self.member_hotspots = [object()] * n_members
        self.properties = {}

    def add_property(self, k, v):
        self.properties[k] = v


class TestSampler:

    def test_samples_every_probe_including_ones_with_no_density_there(self):
        s = ProbeFieldSampler({"BEN": _map(-3.0), "ACT": _map(0.0)})
        vals = s.descriptors_at(np.array([10.0, 10.0, 10.0]))
        assert set(vals) == {"BEN", "ACT"}
        assert vals["BEN"]["field_min_ball"] < vals["ACT"]["field_min_ball"]

    def test_a_point_off_one_probes_grid_gives_none_for_that_probe_only(self):
        s = ProbeFieldSampler({"BEN": _map(-3.0), "ACT": _map(-3.0, shape=(5, 5, 5))})
        vals = s.descriptors_at(np.array([10.0, 10.0, 10.0]))
        assert vals["BEN"]["field_min_ball"] is not None
        assert vals["ACT"]["field_min_ball"] is None


class TestCountNormalisation:
    """The property the whole change exists for."""

    def test_two_sites_with_identical_fields_score_equally_regardless_of_member_count(self):
        s = ProbeFieldSampler({"BEN": _map(-3.0), "PHN": _map(-3.0)})
        one = _Site(1, (10.0, 10.0, 10.0), n_members=1)
        many = _Site(2, (10.0, 10.0, 10.0), n_members=20)
        fused_site_features([one, many], s)
        assert one.properties["fused_affinity"] == pytest.approx(
            many.properties["fused_affinity"]), "member count must not enter the value"

    def test_a_genuinely_deeper_site_still_scores_higher(self):
        """Count-normalising must not flatten real differences."""
        s = ProbeFieldSampler({"BEN": _map(-3.0, centre=(10.0, 10.0, 10.0))})
        at_well = _Site(1, (10.0, 10.0, 10.0))
        off_well = _Site(2, (17.0, 17.0, 17.0))
        fused_site_features([at_well, off_well], s)
        assert at_well.properties["fused_affinity"] > off_well.properties["fused_affinity"]

    def test_fusion_is_z_scored_per_probe_so_one_probes_scale_cannot_dominate(self):
        """A probe with a 100x deeper scale must not swamp the other probes' votes."""
        huge = _map(-300.0, centre=(4.0, 4.0, 4.0))
        small_a = _map(-3.0, centre=(16.0, 16.0, 16.0))
        small_b = _map(-3.0, centre=(16.0, 16.0, 16.0))
        s = ProbeFieldSampler({"HUGE": huge, "A": small_a, "B": small_b})
        at_huge = _Site(1, (4.0, 4.0, 4.0))
        at_small = _Site(2, (16.0, 16.0, 16.0))
        fused_site_features([at_huge, at_small], s)
        # two probes agree on the small well, one disagrees -> the majority should win
        assert at_small.properties["fused_affinity"] > at_huge.properties["fused_affinity"]


class TestScoringDefaults:

    def test_affinity_uses_the_fused_value_when_opted_in(self, make_hotspot, monkeypatch):
        from cosolvkit.analysis.core import scoring
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import _binding_site_feature_values
        monkeypatch.setattr(scoring, "USE_FUSED_AFFINITY", True)
        hs = make_hotspot(agfe_min=-9.0)                 # a deep member, best-of-members bait
        site = BindingSite(site_id=1, member_hotspots=[hs], agfe_min=-9.0)
        site.add_property("fused_affinity", 1.5)
        vals = _binding_site_feature_values([site])
        assert vals["affinity"][0] == pytest.approx(1.5), \
            "fused value must take precedence over the count-biased best-of-members"

    def test_affinity_falls_back_to_agfe_min_with_a_warning(self, make_hotspot, monkeypatch):
        from cosolvkit.analysis.core import scoring
        from cosolvkit.analysis.core.models import BindingSite
        from cosolvkit.analysis.core.scoring import _binding_site_feature_values
        monkeypatch.setattr(scoring, "USE_FUSED_AFFINITY", True)
        site = BindingSite(site_id=1, member_hotspots=[make_hotspot()], agfe_min=-2.0)
        with pytest.warns(UserWarning, match="count"):
            vals = _binding_site_feature_values([site])
        assert vals["affinity"][0] == pytest.approx(-2.0)

    def test_probe_coverage_keeps_its_weight_despite_the_count_bias(self):
        """It IS the member count (rho +0.978), but dropping it measured worse: mean rank of
        true sites 8.25 -> 11.13 over 43 matched sites. Biased and still useful on average."""
        from cosolvkit.analysis.core.scoring import DEFAULT_BINDING_SITE_WEIGHTS
        assert DEFAULT_BINDING_SITE_WEIGHTS["probe_coverage"] == 2.0

    def test_shape_keeps_its_weight(self):
        """Re-measured on v3: solidity is the best per-hotspot discriminator, 0.700 after
        controlling for volume, where agfe_min falls to 0.500."""
        from cosolvkit.analysis.core.scoring import DEFAULT_BINDING_SITE_WEIGHTS
        assert DEFAULT_BINDING_SITE_WEIGHTS["shape"] == 1.0

    def test_fused_affinity_is_higher_is_better(self):
        from cosolvkit.analysis.core.scoring import _BS_INVERTED_FEATURES
        assert "fused_affinity" not in _BS_INVERTED_FEATURES
