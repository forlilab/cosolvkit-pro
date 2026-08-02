"""The sampling verdict must be judged against the cutoff the run will actually use.

``_log_sampling_adequacy`` compares each map's Poisson noise floor against the favourability
cutoff, but it defaulted to ``n_kt=1.0`` while the config's ``hotspots.n_kt`` is what hotspot
detection applies. On the FosAKP re-run with ``n_kt: 2.0`` that produced

    EOH sampling: ... noise floor -0.72 kcal/mol vs cutoff -0.60 kcal/mol
    [CUTOFF BELOW NOISE FLOOR - favourable voxels are not trustworthy]

in every log, when the configured cutoff was -1.19 and cleared the floor comfortably. A warning
that cries wolf on a correctly-configured run is worse than no warning.
"""

import logging
import os

import pytest

pytest.importorskip("MDAnalysis")

from tests.test_grid_analysis import HAS_MDA, _make_universe          # noqa: E402
from tests.test_raw_agfe_export import _report                        # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

COSOLVENT = "BEN"


def _run(tmp_path, caplog, **kwargs):
    r = _report(_make_universe(), tmp_path / "out")
    with caplog.at_level(logging.INFO):
        r.generate_density_maps(cosolvent_names=[COSOLVENT], use_atomtypes=False,
                                gridsize=1.0, temperature=300.0, export_raw=False,
                                **kwargs)
    return caplog.text


def test_reported_cutoff_follows_n_kt(tmp_cwd, tmp_path, caplog):
    """kT = 0.596, so n_kt=2 must be judged against -1.19, not the hardcoded -0.60."""
    text = _run(tmp_path, caplog, n_kt=2.0)
    assert "sampling:" in text
    assert "-1.19" in text, f"expected the n_kt=2 cutoff in the log, got: {text[-400:]}"
    assert "cutoff -0.60" not in text


def test_a_strict_cutoff_does_not_warn_when_it_clears_the_floor(tmp_cwd, tmp_path, caplog):
    """The false alarm: a very strict cutoff is by definition above the noise floor."""
    text = _run(tmp_path, caplog, n_kt=12.0)
    assert "CUTOFF BELOW NOISE FLOOR" not in text
    assert "[ok]" in text
