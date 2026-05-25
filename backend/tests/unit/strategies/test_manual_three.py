from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy
from app.strategies.base import l2_normalize


def _labels(*items: str) -> np.ndarray:
    return np.array(items, dtype=object)


def test_three_groups_full_yields_three_rows(synthetic_vectors):
    v = synthetic_vectors(n=9)  # 3 frontal + 3 left + 3 right
    labels = _labels(*(["frontal"] * 3 + ["left"] * 3 + ["right"] * 3))
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    assert out.shape == (3, 512)


def test_each_row_equals_group_mean_l2(synthetic_vectors):
    v = synthetic_vectors(n=6)
    labels = _labels("frontal", "frontal", "left", "left", "right", "right")
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    expected_frontal = l2_normalize(np.mean(v[:2], axis=0))
    expected_left = l2_normalize(np.mean(v[2:4], axis=0))
    expected_right = l2_normalize(np.mean(v[4:6], axis=0))
    # 行序固定为 frontal/left/right
    assert np.allclose(out[0], expected_frontal, atol=1e-6)
    assert np.allclose(out[1], expected_left, atol=1e-6)
    assert np.allclose(out[2], expected_right, atol=1e-6)


def test_missing_group_yields_fewer_rows(synthetic_vectors):
    v = synthetic_vectors(n=4)
    labels = _labels("frontal", "frontal", "left", "left")  # right 缺
    out = get_strategy("manual_three").build(
        v, rng=np.random.default_rng(0), group_labels=labels,
    )
    assert out.shape == (2, 512)


def test_all_groups_missing_raises(synthetic_vectors):
    v = synthetic_vectors(n=2)
    labels = _labels("unknown", "unknown")
    with pytest.raises(ValueError):
        get_strategy("manual_three").build(
            v, rng=np.random.default_rng(0), group_labels=labels,
        )


def test_missing_group_labels_raises(synthetic_vectors):
    v = synthetic_vectors(n=3)
    with pytest.raises(ValueError):
        get_strategy("manual_three").build(
            v, rng=np.random.default_rng(0), group_labels=None,
        )
