# tests/test_binding_sites_wiring.py
def test_config_defaults():
    from cosolvkit.analysis.config import BindingSitesConfig, HotspotsConfig
    assert BindingSitesConfig().enabled is True
    # 6, not 26: 26 links corner-touching voxels and fused distinct pockets.
    assert BindingSitesConfig().connectivity == 6
    assert BindingSitesConfig().merge_tolerance_ang == 0.0
    assert HotspotsConfig().n_kt == 1.0
    assert HotspotsConfig().clustering.strategy == "skimage_watershed"
    # The size threshold is a VOLUME now; the voxel count is a derived escape hatch.
    assert HotspotsConfig().clustering.min_cluster_voxels is None
    assert HotspotsConfig().clustering.min_cluster_volume_ang3 == 20.0
    assert HotspotsConfig().survival_kwargs == {}


def test_public_exports_binding_site_api():
    import cosolvkit
    from cosolvkit import identify_binding_sites, BindingSiteDetector, BindingSite
    assert "identify_binding_sites" in cosolvkit.__all__
    assert "ConsensusSite" not in cosolvkit.__all__
