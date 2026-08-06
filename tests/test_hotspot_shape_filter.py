"""Discard implausibly convex / diffuse hotspots before they reach binding-site grouping.

Hotspot-level discrimination is the stronger axis in this pipeline, and spurious hotspots are what
drag spurious binding sites into existence, so filtering before the merge is worth more than
re-weighting after it.

Measured on FosAKP (18 probes, 205 hotspots at 20 A^3, `scripts/sweep_hotspot_filter.py`), keeping
hotspots with ``geom_solidity <= max_solidity`` -- real pockets are irregular clefts, so LOW
solidity is the pocket-like end:

    max_solidity   novel hotspots cut   ground-truth pockets kept
      (off)                 0                    6/6
      0.910                19  (13%)             6/6
      0.886                35  (24%)             6/6
      0.869                54  (36%)             6/6
      0.851                69  (47%)             6/6      <- last safe step
      0.815                86  (58%)             5/6      <- loses a pocket

``field_sharpness`` was evaluated as a second criterion and does NOT earn a place: its known and
novel medians are 0.411 vs 0.387, only one threshold is safe at all (0.2931, cutting 16), and the
best combined setting (solidity 0.869 + sharpness 0.2931) cuts 67 novels -- fewer than solidity
alone at 0.851 (69). It is implemented so it can be enabled deliberately, but it buys nothing here.

Both thresholds default to None (off). 0.851 sits one grid step from losing a pocket, and a
threshold tuned to the edge of 6 pockets on a single target is not a default; 0.910 is the
conservative value with margin.
"""

import numpy as np
import pytest


class _Site:
    """Minimal hotspot stand-in: the filter only reads ``properties``."""

    def __init__(self, site_id, solidity=None, sharpness=None):
        self.site_id = site_id
        self.rank = site_id
        self.properties = {}
        if solidity is not None:
            self.properties["geom_solidity"] = solidity
        if sharpness is not None:
            self.properties["field_sharpness"] = sharpness


def _filter(sites, **kw):
    from cosolvkit.analysis.sites.detect import filter_hotspots_by_shape
    return filter_hotspots_by_shape(sites, **kw)


def test_off_by_default_keeps_everything():
    sites = [_Site(1, 0.99, 0.01), _Site(2, 0.10, 0.99)]
    assert _filter(sites) == sites


def test_solidity_ceiling_drops_the_convex_ones():
    """Real pockets are irregular clefts, so a near-sphere (solidity ~1) is the suspect one."""
    blobby, cleft = _Site(1, 0.98), _Site(2, 0.62)
    kept = _filter([blobby, cleft], max_solidity=0.91)
    assert kept == [cleft]


def test_sharpness_floor_drops_the_diffuse_ones():
    diffuse, sharp = _Site(1, sharpness=0.05), _Site(2, sharpness=0.80)
    kept = _filter([diffuse, sharp], min_field_sharpness=0.29)
    assert kept == [sharp]


def test_both_criteria_must_pass():
    good = _Site(1, 0.60, 0.80)
    bad_shape = _Site(2, 0.99, 0.80)
    bad_sharp = _Site(3, 0.60, 0.01)
    kept = _filter([good, bad_shape, bad_sharp], max_solidity=0.91, min_field_sharpness=0.29)
    assert kept == [good]


def test_a_missing_property_never_silently_discards():
    """A hotspot with no geom_solidity means regionprops was off, not that the site is bad.

    Dropping it would make the filter's effect depend on an unrelated setting.
    """
    no_props = _Site(1)
    assert _filter([no_props], max_solidity=0.5, min_field_sharpness=0.9) == [no_props]


def test_non_finite_values_are_treated_as_missing():
    nan_site = _Site(1, float("nan"), float("nan"))
    assert _filter([nan_site], max_solidity=0.5, min_field_sharpness=0.9) == [nan_site]


def test_boundary_is_inclusive():
    on_edge = _Site(1, 0.91, 0.29)
    assert _filter([on_edge], max_solidity=0.91, min_field_sharpness=0.29) == [on_edge]


def test_order_is_preserved():
    sites = [_Site(i, 0.5) for i in range(5)]
    assert [s.site_id for s in _filter(sites, max_solidity=0.9)] == [0, 1, 2, 3, 4]


def test_config_exposes_both_knobs_and_defaults_them_off():
    from cosolvkit.analysis.config import ClusteringConfig
    c = ClusteringConfig()
    assert c.max_solidity is None
    assert c.min_field_sharpness is None


@pytest.mark.parametrize("smax,expect", [(None, 3), (0.910, 2), (0.851, 1), (0.5, 0)])
def test_threshold_monotonically_tightens(smax, expect):
    """Solidity 0.80 / 0.90 / 0.95: a lower ceiling keeps strictly fewer, never more."""
    sites = [_Site(1, 0.80), _Site(2, 0.90), _Site(3, 0.95)]
    assert len(_filter(sites, max_solidity=smax)) == expect
