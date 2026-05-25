from __future__ import annotations

import numpy as np

from app.strategies import get_strategy


def test_shape_equals_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    out = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert out.shape == (7, 512)


def test_output_equals_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    out = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert np.array_equal(out, v)


def test_does_not_mutate_input(synthetic_vectors):
    v = synthetic_vectors(n=7)
    snapshot = v.copy()
    _ = get_strategy("all_vectors").build(v, rng=np.random.default_rng(0))
    assert np.array_equal(v, snapshot)
