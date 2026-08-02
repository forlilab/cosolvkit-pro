"""sp_top_n gating of the default (hotspot-centroid) zone list.

The cap itself and float coercion are covered by
test_sp_supplied_zones.py::test_default_is_top_n_hotspot_centroids and
::test_supplied_zones_are_coerced_to_float; only the under-cap case is unique here.
"""

from cosolvkit.analysis.multi_report import _sp_candidate_zones


class _FakeSite:
    def __init__(self, centroid):
        self.centroid = centroid


def test_fewer_sites_than_cap_returns_all():
    sites = [_FakeSite([1.0, 2.0, 3.0])]
    assert _sp_candidate_zones(sites, sp_top_n=5) == [[1.0, 2.0, 3.0]]
