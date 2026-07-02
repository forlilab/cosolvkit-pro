"""Regression tests for hotspot_dashboard._load_hotspot_csvs.

The loader globs ``hotspot_sites_*.csv``, which also matches auxiliary
``hotspot_sites_geom_*.csv`` tables (per-site geometric descriptors) that
lack a ``cosolvent`` column. Concatenating those injected NaN into the
``cosolvent`` column, breaking the downstream ``sorted(...)`` call with
``TypeError: '<' not supported between instances of 'float' and 'str'``.
The loader must skip any CSV without a ``cosolvent`` column.
"""

import pandas as pd

from cosolvkit.analysis.hotspot_dashboard import _load_hotspot_csvs


def _write_sites_csv(path, cosolvent, ranks):
    pd.DataFrame({
        "rank": ranks,
        "site_id": ranks,
        "cosolvent": [cosolvent] * len(ranks),
        "n_voxels": [10] * len(ranks),
        "centroid_x": [0.0] * len(ranks),
        "centroid_y": [0.0] * len(ranks),
        "centroid_z": [0.0] * len(ranks),
        "agfe_min": [-1.0] * len(ranks),
    }).to_csv(path, index=False)


def _write_geom_csv(path, site_ids):
    # Mirrors hotspot_sites_geom_*.csv: site_id + geom_* columns, no cosolvent.
    pd.DataFrame({
        "site_id": site_ids,
        "geom_area": [1.0] * len(site_ids),
        "geom_solidity": [0.5] * len(site_ids),
    }).to_csv(path, index=False)


def test_loader_skips_geom_files_without_cosolvent(tmp_path):
    _write_sites_csv(tmp_path / "hotspot_sites_IMI.csv", "IMI", [1, 2])
    _write_sites_csv(tmp_path / "hotspot_sites_BEN.csv", "BEN", [1, 2, 3])
    # Auxiliary geom tables that share the prefix but must be excluded.
    _write_geom_csv(tmp_path / "hotspot_sites_geom_IMI.csv", [1, 2])
    _write_geom_csv(tmp_path / "hotspot_sites_geom_BEN.csv", [1, 2, 3])

    df = _load_hotspot_csvs(str(tmp_path))

    # Only the two real sites tables are loaded (2 + 3 rows), no geom pollution.
    assert len(df) == 5
    assert "cosolvent" in df.columns
    assert df["cosolvent"].isna().sum() == 0
    assert "geom_area" not in df.columns
    # The previously-crashing sort now succeeds.
    assert sorted(df["cosolvent"].unique().tolist()) == ["BEN", "IMI"]


def test_loader_returns_empty_when_no_sites_tables(tmp_path):
    # A directory with only geom tables yields an empty frame, not a crash.
    _write_geom_csv(tmp_path / "hotspot_sites_geom_IMI.csv", [1, 2])

    df = _load_hotspot_csvs(str(tmp_path))
    assert df.empty
