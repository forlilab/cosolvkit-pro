"""Non-finite region properties must serialize as None, not inf/nan.

``solidity`` is area/convex_area, and QHull returns a convex area of 0 for blobs that are too
small or degenerate to hull (~10 voxels), so the value comes back as inf. That inf reached the
site properties, the CSV, and the JSON — where it is not valid JSON and silently drops the
site's shape term downstream instead of being recorded as missing.
"""

import json
import math

import numpy as np

from cosolvkit.analysis.sites.properties import _serialize_regionprop_value


def test_positive_infinity_becomes_none():
    assert _serialize_regionprop_value(float("inf")) is None


def test_negative_infinity_becomes_none():
    assert _serialize_regionprop_value(float("-inf")) is None


def test_nan_becomes_none():
    assert _serialize_regionprop_value(float("nan")) is None


def test_numpy_non_finite_becomes_none():
    assert _serialize_regionprop_value(np.float64("inf")) is None
    assert _serialize_regionprop_value(np.float64("nan")) is None


def test_finite_floats_are_untouched():
    assert _serialize_regionprop_value(0.75) == 0.75
    assert _serialize_regionprop_value(np.float64(-2.5)) == -2.5
    assert _serialize_regionprop_value(0.0) == 0.0


def test_integers_stay_integers():
    v = _serialize_regionprop_value(np.int64(7))
    assert v == 7 and isinstance(v, int)


def test_non_finite_inside_an_array_becomes_none_elementwise():
    """Moment arrays can carry a single non-finite entry; it must not poison the whole list."""
    out = _serialize_regionprop_value(np.array([1.0, np.inf, 3.0]))
    assert out == [1.0, None, 3.0]


def test_serialized_output_is_json_dumpable():
    """The point of the fix: inf is not valid JSON, so a checkpoint carrying it is corrupt."""
    payload = {
        "solidity": _serialize_regionprop_value(float("inf")),
        "moments": _serialize_regionprop_value(np.array([1.0, np.nan])),
    }
    text = json.dumps(payload)                 # would raise ValueError on inf under allow_nan=False
    assert json.loads(text) == {"solidity": None, "moments": [1.0, None]}
    assert "Infinity" not in text and "NaN" not in text


def test_slice_handling_is_preserved():
    """Regression guard: the bbox/slice branch must keep working."""
    assert _serialize_regionprop_value(slice(1, 5, None)) == [1, 5, None]
    assert _serialize_regionprop_value((slice(0, 2, None), slice(3, 4, None))) == [
        [0, 2, None], [3, 4, None]
    ]


def test_math_isfinite_agrees_for_every_finite_case():
    for v in (0.0, 1.0, -1.0, 1e-12, 1e12):
        assert _serialize_regionprop_value(v) is not None
        assert math.isfinite(_serialize_regionprop_value(v))
