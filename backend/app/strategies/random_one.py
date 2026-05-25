from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class RandomOneStrategy:
    name = "random_one"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        idx = int(rng.integers(0, vectors.shape[0]))
        return l2_normalize(vectors[idx][None, :])
