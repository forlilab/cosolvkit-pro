from cosolvkit.analysis.multi_report import _sp_candidate_zones


class _FakeSite:
    def __init__(self, centroid):
        self.centroid = centroid


def test_caps_candidate_zones_to_sp_top_n():
    sites = [_FakeSite([float(i), 0.0, 0.0]) for i in range(10)]
    zones = _sp_candidate_zones(sites, sp_top_n=3)
    assert zones == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_fewer_sites_than_cap_returns_all():
    sites = [_FakeSite([1.0, 2.0, 3.0])]
    assert _sp_candidate_zones(sites, sp_top_n=5) == [[1.0, 2.0, 3.0]]


def test_coerces_to_float():
    import numpy as np
    zones = _sp_candidate_zones([_FakeSite(np.array([1, 2, 3]))], sp_top_n=1)
    assert zones == [[1.0, 2.0, 3.0]]
    assert all(isinstance(v, float) for v in zones[0])
