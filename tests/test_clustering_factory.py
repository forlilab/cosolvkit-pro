import pytest
from cosolvkit.analysis.sites.clustering import (
    build_clustering_strategy,
    SkimageWatershedClustering,
    ConnectedComponentsClustering,
    WatershedClustering,
    DBSCANClustering,
)
from cosolvkit.analysis.config import ClusteringConfig


def test_default_strategy_is_skimage_watershed():
    strat = build_clustering_strategy(ClusteringConfig())
    assert isinstance(strat, SkimageWatershedClustering)
    assert strat.min_cluster_voxels == 20


# skimage_watershed is omitted: it is the default and is covered above.
@pytest.mark.parametrize("name,cls", [
    ("connected_components", ConnectedComponentsClustering),
    ("watershed", WatershedClustering),
    ("dbscan", DBSCANClustering),
])
def test_each_strategy_builds(name, cls):
    strat = build_clustering_strategy(ClusteringConfig(strategy=name, min_cluster_voxels=7))
    assert isinstance(strat, cls)
    assert strat.min_cluster_voxels == 7


def test_strategy_kwargs_applied():
    strat = build_clustering_strategy(
        ClusteringConfig(strategy="dbscan", strategy_kwargs={"eps_angstrom": 2.5})
    )
    assert isinstance(strat, DBSCANClustering)
    assert strat.eps_angstrom == 2.5


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown clustering strategy"):
        build_clustering_strategy(ClusteringConfig(strategy="nope"))
