from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


class KmeansK3Strategy:
    name = "kmeans_k3"
    k: int = 3

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = vectors.shape[0]
        if n == 0:
            raise ValueError("kmeans_k3 输入向量数为 0")
        k = min(self.k, n)
        if k == 1:
            return l2_normalize(np.mean(vectors, axis=0, keepdims=True))

        from sklearn.cluster import KMeans
        # 用 rng 派生一个稳定 int seed 给 sklearn
        seed = int(rng.integers(0, 2**31 - 1))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(vectors)
        return l2_normalize(km.cluster_centers_.astype(np.float32))
