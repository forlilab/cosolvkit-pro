# tests/test_binding_sites_wiring.py
def test_config_has_binding_sites_defaults():
    from cosolvkit.analysis.config import BindingSitesConfig
    assert BindingSitesConfig().enabled is True
    assert BindingSitesConfig().connectivity == 26


def test_public_exports_binding_site_api():
    import cosolvkit
    from cosolvkit import identify_binding_sites, BindingSiteDetector, BindingSite
    assert "identify_binding_sites" in cosolvkit.__all__
    assert "ConsensusSite" not in cosolvkit.__all__


def test_hotspots_config_defaults():
    from cosolvkit.analysis.config import HotspotsConfig
    assert HotspotsConfig().n_kt == 1.0
    assert HotspotsConfig().clustering.strategy == "skimage_watershed"
    assert HotspotsConfig().clustering.min_cluster_voxels == 20
    assert HotspotsConfig().survival_kwargs == {}
