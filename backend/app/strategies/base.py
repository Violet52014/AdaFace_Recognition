"""TemplateStrategy Protocol：5 策略的统一形状。"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


def l2_normalize(v: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """对每一行做 L2 归一化。输入 (M,D) 或 (D,) 都接受，输出形状不变。"""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + eps)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


class TemplateStrategy(Protocol):
    name: str

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...
