"""The minimum cluster size should be specifiable in A^3, not only in voxels.

``min_cluster_voxels`` is a count, so its physical meaning moves with ``gridsize``. Changing
gridsize 0.5 -> 0.8 A on FosAKP silently made the filter 4.1x stricter in volume
(10 voxels: 1.25 A^3 -> 5.12 A^3) and dropped a real hotspot: for benzene the site-3 blob passes
the AGFE cutoff (voxels 2.88 A from the ligand) but its cluster no longer survived, so the nearest
voxel belonging to *any* hotspot moved to 7.05 A and the site scored a miss. Re-running with
``min_cluster_voxels=2`` — the same physical volume as 10 voxels at 0.5 A — brought it back
(2.88 A, hit) and raised the hotspot count 32 -> 40.

A threshold expressed in A^3 is invariant to the grid, so a resolution change stops silently
changing which hotspots exist.
"""

import pytest

from cosolvkit.analysis.sites.clustering import min_cluster_voxels_for_volume


class TestVoxelsForVolume:

    def test_converts_volume_to_a_voxel_count(self):
        # 1.25 A^3 at 0.5 A voxels = 10 voxels
        assert min_cluster_voxels_for_volume(1.25, 0.5) == 10

    def test_same_volume_gives_fewer_voxels_on_a_coarser_grid(self):
        """The whole point: the physical threshold is preserved across gridsizes."""
        fine = min_cluster_voxels_for_volume(1.25, 0.5)
        coarse = min_cluster_voxels_for_volume(1.25, 0.8)
        assert fine == 10
        assert coarse == 2                      # 1.25 / 0.512 = 2.44 -> 2

    def test_physical_volume_is_preserved_within_one_voxel(self):
        for gs in (0.4, 0.5, 0.6, 0.8, 1.0):
            n = min_cluster_voxels_for_volume(20.0, gs)
            assert abs(n * gs ** 3 - 20.0) <= gs ** 3

    def test_never_returns_less_than_one_voxel(self):
        assert min_cluster_voxels_for_volume(0.001, 1.0) == 1
        assert min_cluster_voxels_for_volume(0.0, 0.5) == 1

    def test_rejects_a_nonpositive_gridsize(self):
        with pytest.raises(ValueError):
            min_cluster_voxels_for_volume(10.0, 0.0)


class TestConfigWiring:

    def test_config_exposes_the_volume_option_defaulting_off(self):
        from cosolvkit.analysis.config import ClusteringConfig
        c = ClusteringConfig()
        assert c.min_cluster_volume_ang3 is None, "voxel count stays the default behaviour"
        assert c.min_cluster_voxels == 20

    def test_volume_setting_overrides_the_voxel_count(self):
        from cosolvkit.analysis.config import ClusteringConfig
        c = ClusteringConfig(min_cluster_voxels=999, min_cluster_volume_ang3=1.25)
        assert c.resolve_min_cluster_voxels(0.5) == 10
        assert c.resolve_min_cluster_voxels(0.8) == 2

    def test_voxel_count_is_used_when_no_volume_is_given(self):
        from cosolvkit.analysis.config import ClusteringConfig
        c = ClusteringConfig(min_cluster_voxels=17)
        assert c.resolve_min_cluster_voxels(0.5) == 17
        assert c.resolve_min_cluster_voxels(0.8) == 17, "unchanged by gridsize, as before"
