"""Default weights must not name features that cannot move the ranking.

Three incoherences shipped in ``DEFAULT_BINDING_SITE_WEIGHTS`` at once:

* ``kinetics: 1.0`` — the leave-one-probe-out fit on the 13 FosAKP probes with >=3 true sites
  came out NEGATIVE in **13/13 folds**, so the shipped +1.0 was pointed the wrong way. It is
  also inert unless ``survival_kwargs.sp_top_n > 0``. Zeroed rather than flipped: flipping
  asserts a direction from one target, zeroing only declines to assert the wrong one.
* ``chemotype_diversity: 1.0`` — identically zero without ``density_maps.use_atomtypes``, so in
  the default configuration it was a weight on a constant.
* ``accessible_fraction: 0.0`` — the one feature that survived residualising on both buriedness
  and volume (AUC 0.685 residual, 0.724 vs 0.667 for buriedness on matched 250 ns data,
  |rho| <= 0.13 with volume) was computed and then ignored.

Fixed by adopting the fitted ``tier_b_2026_08`` set as the default. ONE TARGET, 6 pockets, so
the SIGNS are the durable part and the magnitudes are provisional — that caveat belongs in the
docstring, not in a weight that contradicts its own fit.
"""

import pytest

from cosolvkit.analysis.core.scoring import (
    DEFAULT_BINDING_SITE_WEIGHTS, LEGACY_BINDING_SITE_WEIGHTS_2026_07,
    _BS_INVERTED_FEATURES,
)


def test_no_default_weight_is_pointed_backwards_or_inert():
    """kinetics and chemotype_diversity must not carry weight by default."""
    assert DEFAULT_BINDING_SITE_WEIGHTS["kinetics"] == 0.0, \
        "kinetics fitted NEGATIVE 13/13 folds and is inert without sp_top_n > 0"
    assert DEFAULT_BINDING_SITE_WEIGHTS["chemotype_diversity"] == 0.0, \
        "chemotype_diversity is identically zero without density_maps.use_atomtypes"


def test_accessible_fraction_is_actually_used():
    """It is computed whenever the accessible mask exists; weighting it at 0 wasted it."""
    assert DEFAULT_BINDING_SITE_WEIGHTS["accessible_fraction"] > 0.0
    assert "accessible_fraction" in _BS_INVERTED_FEATURES, \
        "lower accessible fraction = more enclosed, so it must be inverted"


def test_shape_outranks_affinity_as_fitted():
    """13/13 folds agreed on the sign and ordering: shape > affinity."""
    w = DEFAULT_BINDING_SITE_WEIGHTS
    assert w["shape"] > w["affinity"] > 0.0


def test_volume_is_dropped_as_redundant():
    """Fitted ~0 with the sign flipping in 3/13 folds; redundant with shape + enclosure."""
    assert DEFAULT_BINDING_SITE_WEIGHTS["volume"] == 0.0


def test_legacy_weight_set_is_kept_for_reproducibility():
    """Earlier runs must remain reproducible without editing the library."""
    legacy = LEGACY_BINDING_SITE_WEIGHTS_2026_07
    assert legacy["kinetics"] == 1.0
    assert legacy["chemotype_diversity"] == 1.0
    assert legacy["accessible_fraction"] == 0.0
    assert legacy["volume"] == 1.0
    assert legacy["affinity"] == 3.0 and legacy["shape"] == 1.0
    # and it must still be a complete, usable set
    assert set(legacy) == set(DEFAULT_BINDING_SITE_WEIGHTS)


def test_every_default_weighted_feature_is_scorable(make_hotspot):
    """A non-zero default weight must name a feature the scorer actually reads."""
    from cosolvkit.analysis.core.models import BindingSite
    from cosolvkit.analysis.core.scoring import _binding_site_feature_values
    site = BindingSite(site_id=1, member_hotspots=[make_hotspot()])
    available = set(_binding_site_feature_values([site]))
    weighted = {k for k, v in DEFAULT_BINDING_SITE_WEIGHTS.items() if v != 0.0}
    assert weighted <= available, f"weighted but unscorable: {weighted - available}"


# ---------------------------------------------------------------------------------------
# A default-weighted feature must also be EXPORTED and reachable from the dashboard,
# or the weight is invisible downstream.
# ---------------------------------------------------------------------------------------

def test_accessible_fraction_is_exported_to_the_binding_sites_table(make_hotspot):
    """It was scored but absent from binding_sites.csv, so nothing downstream could see it."""
    from cosolvkit.analysis.core.models import BindingSite
    h = make_hotspot()
    h.add_property("accessible_fraction", 0.42)
    site = BindingSite(site_id=1, member_hotspots=[h])
    d = site.to_dict()
    assert "accessible_fraction" in d, "binding_sites.csv schema omits accessible_fraction"
    assert d["accessible_fraction"] == pytest.approx(0.42, abs=1e-6)


def test_missing_accessible_fraction_exports_as_none(make_hotspot):
    from cosolvkit.analysis.core.models import BindingSite
    site = BindingSite(site_id=1, member_hotspots=[make_hotspot()])
    assert site.to_dict()["accessible_fraction"] is None


def test_dashboard_weights_do_not_drift_from_the_core_defaults():
    """Two hand-maintained weight sets had already diverged; keep one source of truth."""
    from cosolvkit.analysis.hotspot_dashboard import DEFAULT_DASHBOARD_WEIGHTS
    assert DEFAULT_DASHBOARD_WEIGHTS == DEFAULT_BINDING_SITE_WEIGHTS


def test_dashboard_row_adapter_exposes_accessible_fraction():
    """Otherwise the dashboard silently reranks without a feature it weights at 2.0."""
    import pandas as pd
    from cosolvkit.analysis.hotspot_dashboard import _BindingSiteRow
    row = pd.Series({"site_id": 1, "agfe_min": -3.0, "probe_coverage": 1.0, "volume": 100.0,
                     "solidity": 0.6, "residence": 20.0, "favorable_atomtypes": "Car",
                     "accessible_fraction": 0.42})
    assert _BindingSiteRow(row).accessible_fraction == pytest.approx(0.42)
    row2 = row.drop("accessible_fraction")
    assert _BindingSiteRow(row2).accessible_fraction is None
