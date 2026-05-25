from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class MeanAllStrategy:
    name = "mean_all"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        m = np.mean(vectors, axis=0, keepdims=True)
        return l2_normalize(m)
