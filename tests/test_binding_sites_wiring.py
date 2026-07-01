# tests/test_binding_sites_wiring.py
def test_config_has_binding_sites_defaults():
    from cosolvkit.analysis.config import AnalysisConfig, BindingSitesConfig, HotspotsConfig
    assert BindingSitesConfig().enabled is True
    assert BindingSitesConfig().connectivity == 26
    assert HotspotsConfig().compute_survival_probability is True


def test_public_exports_binding_site_api():
    import cosolvkit
    from cosolvkit import identify_binding_sites, BindingSiteDetector, BindingSite
    assert "identify_binding_sites" in cosolvkit.__all__
    assert "ConsensusSite" not in cosolvkit.__all__


def test_hotspots_config_kt_defaults():
    from cosolvkit.analysis.config import HotspotsConfig
    assert HotspotsConfig().cutoff_mode == "kt"
    assert HotspotsConfig().n_kt == 1.0
