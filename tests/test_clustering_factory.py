import pytest
from cosolvkit.analysis.sites.clustering import (
    build_clustering_strategy,
    SkimageWatershedClustering,
    ConnectedComponentsClustering,
)
from cosolvkit.analysis.config import ClusteringConfig


def test_default_strategy_is_skimage_watershed():
    strat = build_clustering_strategy(ClusteringConfig(), gridsize=0.8)
    assert isinstance(strat, SkimageWatershedClustering)
    # 20 A^3 / 0.8**3 = 39 voxels. The count is DERIVED now, so it moves with gridsize while the
    # physical threshold stays put -- that is the point.
    assert strat.min_cluster_voxels == 39


def test_gridsize_is_required():
    """It used to default to None, and passing nothing silently ignored the volume threshold."""
    with pytest.raises(TypeError):
        build_clustering_strategy(ClusteringConfig())


def test_the_same_volume_gives_a_different_count_per_gridsize():
    fine = build_clustering_strategy(ClusteringConfig(), gridsize=0.5)
    coarse = build_clustering_strategy(ClusteringConfig(), gridsize=1.0)
    assert fine.min_cluster_voxels == 160 and coarse.min_cluster_voxels == 20
    assert fine.min_cluster_voxels * 0.5**3 == coarse.min_cluster_voxels * 1.0**3 == 20.0


# skimage_watershed is omitted: it is the default and is covered above.
@pytest.mark.parametrize("name,cls", [
    ("connected_components", ConnectedComponentsClustering),
])
def test_each_strategy_builds(name, cls):
    strat = build_clustering_strategy(
        ClusteringConfig(strategy=name, min_cluster_volume_ang3=3.584), gridsize=0.8)
    assert isinstance(strat, cls)
    assert strat.min_cluster_voxels == 7  # 3.584 / 0.8**3


def test_strategy_kwargs_applied():
    strat = build_clustering_strategy(
        ClusteringConfig(strategy="connected_components", strategy_kwargs={"connectivity": 6}),
        gridsize=0.8,
    )
    assert isinstance(strat, ConnectedComponentsClustering)
    assert strat.connectivity == 6


def test_registry_holds_only_the_two_live_strategies():
    """`watershed` and `dbscan` were removed: neither was ever a default.

    `watershed` seeded with peak_local_max(min_distance=...), a grid-dependent geometric spacing,
    where skimage_watershed uses an h_maxima contrast threshold in kcal/mol -- strictly better on
    a smooth AGFE field. `dbscan` ignored both voxel adjacency and AGFE intensity.
    """
    for gone in ("watershed", "dbscan"):
        with pytest.raises(ValueError, match="Unknown clustering strategy"):
            build_clustering_strategy(ClusteringConfig(strategy=gone), gridsize=0.8)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown clustering strategy"):
        build_clustering_strategy(ClusteringConfig(strategy="nope"), gridsize=0.8)
