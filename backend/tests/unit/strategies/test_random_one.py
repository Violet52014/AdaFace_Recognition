from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_one_row(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("random_one").build(v, rng=np.random.default_rng(42))
    assert out.shape == (1, 512)


def test_deterministic_given_seed(synthetic_vectors):
    v = synthetic_vectors(n=10)
    s = get_strategy("random_one")
    a = s.build(v, rng=np.random.default_rng(7))
    b = s.build(v, rng=np.random.default_rng(7))
    assert np.allclose(a, b)


def test_output_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("random_one").build(v, rng=np.random.default_rng(0))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)
