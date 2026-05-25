from __future__ import annotations

from typing import Optional

import numpy as np


class AllVectorsStrategy:
    name = "all_vectors"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 输入约定已 L2 归一化；为避免污染调用方语义，返回拷贝
        return vectors.copy()
