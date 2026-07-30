"""Kinetics must be profilable at SUPPLIED points, not only at hotspot centroids.

The default zones are hotspot centroids, which are thresholded-and-clustered objects. Measured
against crystallographic ligand positions they sit a median 1.4 A (max 3.8 A) from the ligand
they represent, and profiling the same trajectories at ligand-centred points instead moved
sp_half_life from AUC ~0.5 (rho ~0.15) to AUC 0.90 (rho 0.66). So where the zone is placed is
part of the measurement, and callers need to control it.
"""

import numpy as np
import pytest

from cosolvkit.analysis.multi_report import (
    _load_zones_csv,
    _sp_candidate_zones,
    _zone_to_site_rank,
)


class _Site:
    def __init__(self, centroid, rank):
        self.centroid = np.asarray(centroid, dtype=float)
        self.rank = rank


def _sites():
    return [_Site([0.0, 0.0, 0.0], 1), _Site([10.0, 0.0, 0.0], 2),
            _Site([20.0, 0.0, 0.0], 3)]


# ---------------------------------------------------------------------------
# Zone selection
# ---------------------------------------------------------------------------

def test_default_is_top_n_hotspot_centroids():
    z = _sp_candidate_zones(_sites(), 2)
    assert z == [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]


def test_supplied_zones_replace_the_centroids():
    z = _sp_candidate_zones(_sites(), 2, zones=[[1.5, 2.5, 3.5]])
    assert z == [[1.5, 2.5, 3.5]]


def test_supplied_zones_ignore_sp_top_n():
    """sp_top_n must not truncate an explicit list the caller asked for."""
    zones = [[float(i), 0.0, 0.0] for i in range(7)]
    assert len(_sp_candidate_zones(_sites(), 2, zones=zones)) == 7


def test_supplied_zones_are_coerced_to_float():
    z = _sp_candidate_zones(_sites(), 1, zones=[(1, 2, 3)])
    assert z == [[1.0, 2.0, 3.0]]
    assert all(isinstance(v, float) for v in z[0])


def test_empty_supplied_list_is_honoured_not_treated_as_absent():
    """[] means 'no zones', which must not silently fall back to centroids."""
    assert _sp_candidate_zones(_sites(), 2, zones=[]) == []


# ---------------------------------------------------------------------------
# Zone -> site attribution
# ---------------------------------------------------------------------------

def test_zone_maps_to_nearest_hotspot_rank():
    m = _zone_to_site_rank([[0.4, 0.0, 0.0], [9.6, 0.0, 0.0]], _sites(), 4.0)
    assert m == {0: 1, 1: 2}


def test_zone_beyond_cutoff_is_left_unmapped():
    """Otherwise a distant zone's kinetics would be misattributed to a real site."""
    m = _zone_to_site_rank([[5.0, 0.0, 0.0]], _sites(), 2.0)
    assert m == {}


def test_mapping_is_by_distance_not_by_index():
    """The default behaviour (zone i -> rank i+1) is wrong for supplied zones."""
    zones = [[20.0, 0.0, 0.0], [0.0, 0.0, 0.0]]      # reversed w.r.t. rank order
    assert _zone_to_site_rank(zones, _sites(), 1.0) == {0: 3, 1: 1}


def test_mapping_with_no_sites_is_empty():
    assert _zone_to_site_rank([[0.0, 0.0, 0.0]], [], 4.0) == {}


# ---------------------------------------------------------------------------
# zones_csv loading
# ---------------------------------------------------------------------------

def test_zones_csv_roundtrip(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    assert _load_zones_csv(str(p)) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_zones_csv_accepts_extra_columns_and_any_case(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("label,X,Y,Z,known\nsiteA,1,2,3,True\n")
    assert _load_zones_csv(str(p)) == [[1.0, 2.0, 3.0]]


def test_zones_csv_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="needs x, y, z columns"):
        _load_zones_csv(str(p))
