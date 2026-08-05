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

from cosolvkit.analysis.sites.clustering import min_cluster_voxels_for_volume


class TestVoxelsForVolume:

    def test_same_volume_gives_fewer_voxels_on_a_coarser_grid(self):
        """The whole point: the physical threshold is preserved across gridsizes."""
        fine = min_cluster_voxels_for_volume(1.25, 0.5)
        coarse = min_cluster_voxels_for_volume(1.25, 0.8)
        assert fine == 10                       # 1.25 / 0.125 = 10
        assert coarse == 2                      # 1.25 / 0.512 = 2.44 -> 2

    def test_never_returns_less_than_one_voxel(self):
        assert min_cluster_voxels_for_volume(0.001, 1.0) == 1
        assert min_cluster_voxels_for_volume(0.0, 0.5) == 1


class TestConfigWiring:

    def test_the_volume_is_the_default_and_is_grid_independent(self):
        from cosolvkit.analysis.config import ClusteringConfig
        c = ClusteringConfig()
        assert c.min_cluster_volume_ang3 == 20.0
        for gs in (0.5, 0.8, 1.0):
            n = c.resolve_min_cluster_voxels(gs)
            assert abs(n * gs ** 3 - 20.0) < gs ** 3, f"threshold moved at gridsize {gs}"

    def test_there_is_no_voxel_count_knob(self):
        """The volume is the only size setting. A raw count used to sit beside it as an override,
        which meant two ways to say the same thing and a precedence rule to get wrong."""
        from cosolvkit.analysis.config import ClusteringConfig
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ClusteringConfig)}
        assert "min_cluster_volume_ang3" in fields
        assert "min_cluster_voxels" not in fields
        import pytest
        with pytest.raises(TypeError):
            ClusteringConfig(min_cluster_voxels=17)
