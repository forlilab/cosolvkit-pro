import pytest
from cosolvkit.analysis.config import HotspotsConfig, resolve_agfe_cutoff
from cosolvkit.analysis.core.grid import BOLTZMANN_CONSTANT_KB


def test_default_is_minus_one_kt():
    cfg = HotspotsConfig()
    assert cfg.n_kt == 1.0
    got = resolve_agfe_cutoff(cfg, temperature=300.0)
    assert got == pytest.approx(-1.0 * BOLTZMANN_CONSTANT_KB * 300.0, abs=1e-9)  # ~ -0.5962


def test_scales_with_temperature_and_nkt():
    cfg = HotspotsConfig(n_kt=2.0)
    assert resolve_agfe_cutoff(cfg, 310.0) == pytest.approx(
        -2.0 * BOLTZMANN_CONSTANT_KB * 310.0, abs=1e-9
    )
