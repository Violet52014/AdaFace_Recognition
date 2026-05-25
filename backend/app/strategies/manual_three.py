from __future__ import annotations

from typing import Optional

import numpy as np

from .base import l2_normalize


POSE_ORDER = ("frontal", "left", "right")


class ManualThreeStrategy:
    name = "manual_three"

    def build(
        self,
        vectors: np.ndarray,
        *,
        rng: np.random.Generator,
        group_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if group_labels is None:
            raise ValueError("manual_three 需要 group_labels（pose 标签）")
        if len(group_labels) != vectors.shape[0]:
            raise ValueError("group_labels 长度必须与 vectors 数量一致")

        rows = []
        for pose in POSE_ORDER:
            mask = np.array([str(g) == pose for g in group_labels])
            if not mask.any():
                continue
            rows.append(l2_normalize(np.mean(vectors[mask], axis=0)))
        if not rows:
            raise ValueError(f"无任一姿态组（{POSE_ORDER}）有样本")
        return np.stack(rows, axis=0)
