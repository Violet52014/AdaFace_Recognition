from __future__ import annotations

import pytest

from app.strategies import STRATEGIES, get_strategy


def test_registry_contains_all_five():
    expected = {"random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"}
    assert set(STRATEGIES.keys()) == expected


def test_get_strategy_returns_named():
    s = get_strategy("mean_all")
    assert s.name == "mean_all"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("not_a_real_strategy")
