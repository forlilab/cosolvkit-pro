"""Probe chemotype classes, and the score feature built on them.

``probe_coverage`` counts how many probes hit a site; it says nothing about how
chemically different they are. Four hydrophobes and one anion-plus-one-cation give very
different chemical evidence for the same probe count, so
``probe_chemotype_coverage`` counts CLASSES instead.
"""

import warnings

import numpy as np
import pytest

from cosolvkit.analysis.core.chemotypes import (
    CHEMOTYPE_CLASSES,
    DEFAULT_PROBE_CHEMOTYPES,
    n_available_chemotypes,
    probe_chemotypes,
    resolve_probe_chemotypes,
)
from cosolvkit.analysis.core.models import BindingSite, Hotspot
from cosolvkit.analysis.core.scoring import (
    DEFAULT_BINDING_SITE_WEIGHTS,
    normalize_weights,
    score_binding_sites,
)
from cosolvkit.analysis.sites.binding_sites import identify_binding_sites


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_every_default_class_is_valid():
    for resname, classes in DEFAULT_PROBE_CHEMOTYPES.items():
        for c in classes:
            assert c in CHEMOTYPE_CLASSES, f"{resname} -> {c}"


def test_hydrophobes_span_fewer_classes_than_a_charge_pair():
    """The whole point of the feature, as a test."""
    four_hydrophobes = probe_chemotypes(["BEN", "PHN", "PRP", "FBZ"])
    charge_pair = probe_chemotypes(["ACT", "MAM"])
    assert "anionic" not in four_hydrophobes
    assert set(charge_pair) >= {"anionic", "cationic"}


def test_benzene_is_not_an_hbond_partner():
    """Benzene is aromatic and hydrophobic, and neither donates nor accepts.

    It was tagged ``aliphatic``, which is wrong on its face: benzene has no non-aromatic carbon.
    ``aliphatic`` means a saturated carbon skeleton, ``hydrophobic`` means an apolar surface --
    see ``tests/test_probe_chemotype_chemistry.py``, which now derives this from the structure.
    """
    assert probe_chemotypes(["BEN"]) == ["aromatic", "hydrophobic"]


def test_unknown_resnames_contribute_nothing():
    assert probe_chemotypes(["NOPE", "ZZZ"]) == []


def test_classes_come_back_in_canonical_order():
    got = probe_chemotypes(["MAM", "ACT", "BEN"])
    assert got == [c for c in CHEMOTYPE_CLASSES if c in got]


def test_resname_lookup_is_case_and_space_insensitive():
    assert probe_chemotypes([" ben "]) == probe_chemotypes(["BEN"])


def test_overrides_beat_the_builtin_table():
    m = resolve_probe_chemotypes({"BEN": ["anionic"]})
    assert probe_chemotypes(["BEN"], m) == ["anionic"]
    # a bare string is accepted in place of a one-element list
    m = resolve_probe_chemotypes({"XYZ": "cationic"})
    assert probe_chemotypes(["XYZ"], m) == ["cationic"]


def test_override_with_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown chemotype class"):
        resolve_probe_chemotypes({"BEN": ["not_a_class"]})


def test_available_classes_uses_the_panel():
    # A hydrophobe-only panel cannot express charges, so the denominator shrinks.
    assert n_available_chemotypes(["BEN", "PRP"]) == len(probe_chemotypes(["BEN", "PRP"]))


def test_available_classes_falls_back_when_panel_is_unclassified():
    assert n_available_chemotypes(["NOPE"]) == len(CHEMOTYPE_CLASSES)


# ---------------------------------------------------------------------------
# BindingSite integration
# ---------------------------------------------------------------------------

def test_binding_site_derives_chemotypes_from_cosolvents():
    bs = BindingSite(site_id=1, cosolvents=["BEN", "ACT"])
    assert "aromatic" in bs.probe_chemotypes
    assert "anionic" in bs.probe_chemotypes
    assert bs.n_probe_chemotypes == len(bs.probe_chemotypes)
    # and they reach the exported record
    d = bs.to_dict()
    assert "probe_chemotypes" in d
    assert d["n_probe_chemotypes"] > 0
    assert d["probe_chemotype_coverage"] is not None


def test_probe_chemotype_coverage_is_a_fraction():
    bs = BindingSite(site_id=1, cosolvents=["ACT"], n_total_probe_chemotypes=4)
    assert bs.probe_chemotype_coverage == pytest.approx(len(bs.probe_chemotypes) / 4)


def test_coverage_falls_back_to_full_class_list():
    bs = BindingSite(site_id=1, cosolvents=["BEN"])
    assert bs.probe_chemotype_coverage == pytest.approx(
        len(bs.probe_chemotypes) / len(CHEMOTYPE_CLASSES))


def test_probe_coverage_and_chemotype_coverage_can_disagree():
    """Four hydrophobes: high probe_coverage, low chemotype coverage."""
    many = BindingSite(site_id=1, cosolvents=["BEN", "FBZ", "PRP", "PHN"],
                       n_total_cosolvents=4, n_total_probe_chemotypes=6)
    few = BindingSite(site_id=2, cosolvents=["ACT", "MAM"],
                      n_total_cosolvents=4, n_total_probe_chemotypes=6)
    assert many.probe_coverage > few.probe_coverage
    assert set(few.probe_chemotypes) - set(many.probe_chemotypes)


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

def test_legacy_diversity_alias_is_rejected():
    """The `diversity` -> `chemotype_diversity` alias was removed, so the old name now raises.

    Rejecting beats silently remapping: the two names scored different things (atom types vs
    probes), and a weight that is quietly renamed changes a score without saying so.
    """
    with pytest.raises(ValueError, match="diversity"):
        normalize_weights({"diversity": 7.0})


def test_unknown_weight_key_raises():
    with pytest.raises(ValueError, match="Unknown binding-site weight"):
        normalize_weights({"favourability": 1.0})


def test_missing_keys_fall_back_to_defaults():
    w = normalize_weights({"affinity": 9.0})
    assert w["affinity"] == 9.0
    assert w["volume"] == DEFAULT_BINDING_SITE_WEIGHTS["volume"]
    assert normalize_weights(None) == DEFAULT_BINDING_SITE_WEIGHTS


def test_new_feature_is_opt_in():
    assert DEFAULT_BINDING_SITE_WEIGHTS["probe_chemotype_coverage"] == 0.0


def test_chemotype_coverage_weight_changes_the_ranking():
    """With a positive weight the chemically broader site wins; negative flips it."""
    def _bs(site_id, cosolvents):
        return BindingSite(site_id=site_id, cosolvents=cosolvents,
                           n_total_cosolvents=4, n_total_probe_chemotypes=6,
                           centroid=np.zeros(3))
    only = {k: 0.0 for k in DEFAULT_BINDING_SITE_WEIGHTS}

    broad, narrow = _bs(1, ["ACT", "MAM", "BEN"]), _bs(2, ["PRP"])
    score_binding_sites([broad, narrow], {**only, "probe_chemotype_coverage": 1.0})
    assert broad.rank == 1

    broad, narrow = _bs(1, ["ACT", "MAM", "BEN"]), _bs(2, ["PRP"])
    score_binding_sites([broad, narrow], {**only, "probe_chemotype_coverage": -1.0})
    assert narrow.rank == 1


# ---------------------------------------------------------------------------
# End-to-end through identify_binding_sites
# ---------------------------------------------------------------------------

def _hs(cosolvent, site_id, blob, shape=(20, 20, 20)):
    mask = np.zeros(shape, dtype=bool)
    mask[blob] = True
    h = Hotspot(rank=1, site_id=site_id, cosolvent=cosolvent, n_voxels=int(mask.sum()),
                centroid=np.zeros(3), agfe_min=-2.0, agfe_mean_top_pct=-2.0,
                voxel_mask=mask, favorable_atomtypes=["Car"],
                per_type_agfe={"Car": -2.0})
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    return h


def test_identify_binding_sites_populates_chemotypes():
    a = _hs("BEN", 1, np.s_[5:9, 5:9, 5:9])
    b = _hs("ACT", 2, np.s_[7:11, 7:11, 7:11])   # overlaps a -> one site
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sites = identify_binding_sites({"BEN": [a], "ACT": [b]})
    assert len(sites) == 1
    s = sites[0]
    assert "aromatic" in s.probe_chemotypes
    assert "anionic" in s.probe_chemotypes
    assert 0.0 < s.probe_chemotype_coverage <= 1.0


def test_identify_binding_sites_honours_chemotype_overrides():
    a = _hs("XYZ", 1, np.s_[5:9, 5:9, 5:9])
    sites = identify_binding_sites({"XYZ": [a]},
                                   probe_chemotype_overrides={"XYZ": ["cationic"]})
    assert sites[0].probe_chemotypes == ["cationic"]
