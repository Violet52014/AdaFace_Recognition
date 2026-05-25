"""5 策略注册表。"""
from __future__ import annotations

from typing import Dict

from .all_vectors import AllVectorsStrategy
from .base import TemplateStrategy
from .kmeans_k3 import KmeansK3Strategy
from .manual_three import ManualThreeStrategy
from .mean_all import MeanAllStrategy
from .random_one import RandomOneStrategy


STRATEGIES: Dict[str, TemplateStrategy] = {
    "random_one": RandomOneStrategy(),
    "mean_all": MeanAllStrategy(),
    "manual_three": ManualThreeStrategy(),
    "kmeans_k3": KmeansK3Strategy(),
    "all_vectors": AllVectorsStrategy(),
}


def get_strategy(name: str) -> TemplateStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"未知策略: {name}，可选 {list(STRATEGIES)}")
    return STRATEGIES[name]


__all__ = ["STRATEGIES", "TemplateStrategy", "get_strategy"]
