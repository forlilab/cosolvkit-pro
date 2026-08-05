import numpy as np
import pytest
from cosolvkit.analysis.core.models import BindingSite
from cosolvkit.analysis.core.scoring import score_binding_sites


def _bs(site_id, agfe_min, volume, n_cos, residence, solidity, atomtypes):
    return BindingSite(
        site_id=site_id, member_hotspots=[], voxel_mask=None,
        centroid=np.zeros(3), agfe_min=agfe_min, agfe_mean_top_pct=agfe_min,
        volume=volume, solidity=solidity, extent=0.5, axis_major_length=1.0,
        axis_minor_length=1.0, favorable_atomtypes=list(atomtypes), pharmacophore={},
        residence=residence, cosolvents=[f"C{i}" for i in range(n_cos)], n_total_cosolvents=2,
    )


def test_score_binding_sites_default_weights_and_rank():
    # 2 sites. Defaults (the fitted tier_b set): affinity+3.0, shape+3.5, accessible_fraction+2.0,
    # probe_coverage+2.0. volume, kinetics and chemotype_diversity are now 0.0 -- see
    # tests/test_default_weight_coherence.py for why each was zeroed.
    # A: agfe_min -3 (best), vol 100, 2 cos (coverage 1.0), residence 20, solidity 0.6, 2 atomtypes
    # B: agfe_min -1 (worst), vol 50, 1 cos (coverage 0.5), residence 10, solidity 0.9, 1 atomtype
    # solidity: A=0.6, B=0.9 — known sites are LESS convex (0.742 vs 0.835 measured), so the
    # all-best site takes the lower value.
    a = _bs(1, -3.0, 100.0, 2, 20.0, 0.6, ["Car", "HBA"])
    b = _bs(2, -1.0, 50.0, 1, 10.0, 0.9, ["Car"])
    score_binding_sites([a, b])
    # Every scorable feature: A=1.0, B=0.0 after normalization (A better on all; affinity and
    # shape inverted). These fixtures carry no accessible_fraction (it needs the accessible-volume
    # mask), so its 2.0 contributes nothing here and the dead-weight guard warns; it is covered in
    # tests/test_accessible_fraction_feature.py.
    # combined_A = 3.0 + 3.5 + 2.0 = 8.5 ; combined_B = 0.0
    assert a.combined == pytest.approx(8.5, abs=1e-9)
    assert b.combined == pytest.approx(0.0, abs=1e-9)
    assert a.rank == 1 and b.rank == 2


def test_score_binding_sites_negative_weight_flips_preference():
    a = _bs(1, -3.0, 100.0, 2, 20.0, 0.9, ["Car", "HBA"])  # all-best
    b = _bs(2, -1.0, 50.0, 1, 10.0, 0.6, ["Car"])          # all-worst
    # Only volume matters, negatively: prefer SMALLER volume -> B should win.
    score_binding_sites([a, b], {"affinity": 0, "probe_coverage": 0, "kinetics": 0,
                                 "shape": 0, "chemotype_diversity": 0, "volume": -1})
    assert a.combined == pytest.approx(-1.0, abs=1e-9)  # vol_norm 1.0 * -1
    assert b.combined == pytest.approx(0.0, abs=1e-9)
    assert b.rank == 1 and a.rank == 2


def test_score_binding_sites_none_kinetics_contributes_zero():
    a = _bs(1, -3.0, 100.0, 2, None, 0.9, ["Car"])   # residence missing
    b = _bs(2, -2.0, 100.0, 2, 10.0, 0.9, ["Car"])
    score_binding_sites([a, b], {"affinity": 0, "probe_coverage": 0, "volume": 0,
                                 "shape": 0, "chemotype_diversity": 0, "kinetics": 1})
    # b has the only finite residence -> minmax over {10.0} flat -> 1.0; a None -> 0.0
    assert b.combined == pytest.approx(1.0, abs=1e-9)
    assert a.combined == pytest.approx(0.0, abs=1e-9)


def test_shape_scores_lower_solidity_first():
    """`shape` is lower-is-better and was once scored with the wrong sign.

    Re-measured on FosAKP analysis_v3 over 405 hotspots (51 known): solidity separates known
    from novel at within-probe AUC 0.734 (0.700 after controlling for volume), and known sites
    are the *less* convex ones — mean 0.742 vs 0.835. Real pockets are irregular clefts.
    """
    irregular = _bs(1, -1.0, 100.0, 1, 1.0, 0.74, ["Car"])
    round_ = _bs(2, -1.0, 100.0, 1, 1.0, 0.84, ["Car"])
    score_binding_sites([irregular, round_],
                        {"shape": 1.0, "affinity": 0.0, "probe_coverage": 0.0,
                         "volume": 0.0, "kinetics": 0.0, "chemotype_diversity": 0.0})
    assert irregular.rank == 1
