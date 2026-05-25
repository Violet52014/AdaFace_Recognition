from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_one_row(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    assert out.shape == (1, 512)


def test_equals_mean_then_l2(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    expected = np.mean(v, axis=0)
    expected = expected / (np.linalg.norm(expected) + 1e-8)
    assert np.allclose(out[0], expected, atol=1e-6)


def test_output_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("mean_all").build(v, rng=np.random.default_rng(0))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)
