"""`shape` (solidity) must score lower-is-better, matching the measurement.

Re-measured on FosAKP `analysis_v3` over 405 hotspots (51 known): solidity separates known from
novel sites at within-probe AUC 0.734, and **0.700 after controlling for cluster volume** — the
strongest feature available, where `agfe_min` falls to exactly 0.500. The direction is
unambiguous: known sites have **lower** solidity (mean 0.742) than novel ones (0.835). Real
pockets are irregular clefts; the rounder blobs are the spurious ones.

The scorer treated `shape` as higher-is-better, so the best-measured feature was contributing with
the wrong sign.
"""

import numpy as np
import pytest

from cosolvkit.analysis.core.models import BindingSite
from cosolvkit.analysis.core.scoring import _BS_INVERTED_FEATURES, score_binding_sites


def _site(site_id, solidity):
    return BindingSite(site_id=site_id, centroid=np.zeros(3), agfe_min=-1.0, volume=100.0,
                       solidity=solidity, extent=0.5, axis_major_length=1.0,
                       axis_minor_length=1.0, favorable_atomtypes=["Car"], residence=1.0,
                       cosolvents=["C0"], n_total_cosolvents=1)


def test_shape_is_registered_as_lower_is_better():
    assert "shape" in _BS_INVERTED_FEATURES


def test_the_less_convex_site_ranks_first():
    """Known sites average solidity 0.742 vs 0.835 for novel ones."""
    irregular, round_ = _site(1, 0.74), _site(2, 0.84)
    score_binding_sites([irregular, round_], weights={"shape": 1.0, "affinity": 0.0,
                                                      "volume": 0.0, "kinetics": 0.0,
                                                      "chemotype_diversity": 0.0})
    assert irregular.rank == 1


def test_a_negative_shape_weight_flips_it_back():
    """Guard that the term is still a working knob after inverting."""
    irregular, round_ = _site(1, 0.74), _site(2, 0.84)
    score_binding_sites([irregular, round_], weights={"shape": -1.0, "affinity": 0.0,
                                                      "volume": 0.0, "kinetics": 0.0,
                                                      "chemotype_diversity": 0.0})
    assert round_.rank == 1
