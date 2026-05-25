from __future__ import annotations

import numpy as np
import pytest

from app.strategies import get_strategy


def test_shape_is_min_k_n(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (3, 512)


def test_n_smaller_than_k_degrades(synthetic_vectors):
    v = synthetic_vectors(n=2)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (2, 512)


def test_n_one_returns_single_template(synthetic_vectors):
    v = synthetic_vectors(n=1)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    assert out.shape == (1, 512)


def test_each_row_is_l2_normalized(synthetic_vectors):
    v = synthetic_vectors(n=10)
    out = get_strategy("kmeans_k3").build(v, rng=np.random.default_rng(42))
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic_given_seed(synthetic_vectors):
    v = synthetic_vectors(n=10)
    s = get_strategy("kmeans_k3")
    a = s.build(v, rng=np.random.default_rng(99))
    b = s.build(v, rng=np.random.default_rng(99))
    assert np.allclose(a, b)
