"""ROC / EER / TAR@FAR / AUC，纯 numpy。

实现要点：
- roc_curve 按 score 降序排序，逐阈值累计 TP/FP，输出从 (0,0) 到 (1,1) 的折线点。
- AUC 用梯形法（np.trapz）。
- EER = fpr 与 (1-tpr) 的最小交差；具体取 |fpr - fnr| 的 argmin。
- TAR@FAR(target) = 在 fpr <= target 区域的最大 tpr；若无可达点返回 0。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate(scores: np.ndarray, labels: np.ndarray) -> None:
    if scores.shape != labels.shape:
        raise ValueError("scores 与 labels 形状不一致")
    if np.isnan(scores).any():
        raise ValueError("scores 含 NaN")
    uniq = set(np.unique(labels).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError("labels 必须为 0/1")
    if uniq != {0, 1}:
        raise ValueError("labels 必须同时包含 0 和 1")


def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (fpr, tpr, thresholds)，三者长度一致，从 (0,0) 出发到 (1,1) 结束。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _validate(scores, labels)

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    n_pos = int(l_sorted.sum())
    n_neg = int(len(l_sorted) - n_pos)

    tp = np.cumsum(l_sorted == 1)
    fp = np.cumsum(l_sorted == 0)

    # 在每段 score 相等的边界处取点
    distinct = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s_sorted[distinct]

    # 在前面拼接 (0,0)，对应阈值 +inf
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    thr = np.r_[np.inf, thr]

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


def eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """EER = 让 fpr ≈ 1-tpr 的那个工作点；返回 (eer_value, threshold@eer)。"""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = int(np.argmin(diff))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """fpr <= target_far 区域的最大 tpr；不可达则 0.0。"""
    mask = fpr <= target_far
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())
