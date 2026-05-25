from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import auc, eer, roc_curve, tar_at_far


def test_perfect_separation():
    # 50 正样本得分=1，50 负样本得分=0
    scores = np.concatenate([np.ones(50), np.zeros(50)])
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
    fpr, tpr, thr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(1.0, abs=1e-6)
    eer_v, _ = eer(fpr, tpr, thr)
    assert eer_v == pytest.approx(0.0, abs=1e-6)


def test_random_uniform_auc_near_half():
    rng = np.random.default_rng(42)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    fpr, tpr, _ = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=0.05)


def test_tar_at_far_unreachable_returns_zero():
    # 全部 score 相同 → fpr 直接从 0 跳到 1，FAR=1e-3 不可达
    scores = np.full(100, 0.5)
    labels = np.array([0, 1] * 50)
    fpr, tpr, _ = roc_curve(scores, labels)
    assert tar_at_far(fpr, tpr, 1e-3) == 0.0


def test_constant_scores_degenerate():
    scores = np.full(100, 0.5)
    labels = np.array([0, 1] * 50)
    fpr, tpr, thr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=1e-6)
    eer_v, _ = eer(fpr, tpr, thr)
    assert eer_v == pytest.approx(0.5, abs=1e-6)


def test_rejects_nan():
    scores = np.array([0.1, np.nan, 0.5])
    labels = np.array([0, 1, 1])
    with pytest.raises(ValueError):
        roc_curve(scores, labels)


def test_rejects_single_class_labels():
    # 单类 labels 在 roc_curve 入口被拒绝；degenerate 已构造曲线交给 eer 不强制拒绝
    # （eer 是数值最近点，曲线本身不携带"是否单类"信息）
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        roc_curve(scores, labels)


def test_tpr_strictly_monotonic_perfect():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3])
    labels = np.array([1, 1, 1, 0, 0])
    fpr, tpr, _ = roc_curve(scores, labels)
    # tpr 单调不减
    assert np.all(np.diff(tpr) >= -1e-9)
    assert np.all(np.diff(fpr) >= -1e-9)
