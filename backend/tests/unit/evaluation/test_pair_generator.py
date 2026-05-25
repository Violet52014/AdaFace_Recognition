from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.pair_generator import Pair, make_pairs


def _l2(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-8)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def test_pair_counts_basic():
    rng = np.random.default_rng(0)
    # 3 人 × 2 probe；gallery 单模板；LFW 5 张
    probe = {
        "a": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
        "b": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
        "c": [_l2(rng.standard_normal(8).astype(np.float32)) for _ in range(2)],
    }
    gallery = {
        "a": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
        "b": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
        "c": _l2(rng.standard_normal((1, 8)).astype(np.float32)),
    }
    lfw = _l2(rng.standard_normal((5, 8)).astype(np.float32))
    pairs = make_pairs(probe, gallery, lfw)
    n_genuine = sum(1 for p in pairs if p.is_genuine)
    n_impostor = sum(1 for p in pairs if not p.is_genuine)
    assert n_genuine == 6              # 3 人 × 2 probe
    assert n_impostor == 6 * 2 + 5 * 3  # cross-person 12 + lfw 15


def test_max_cosine_with_multiple_templates():
    # gallery 含 3 模板，其中 1 个与 probe 完全一致 → score=1.0
    probe_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gallery = {
        "a": np.stack([other, probe_vec, other], axis=0),
    }
    probe = {"a": [probe_vec]}
    pairs = make_pairs(probe, gallery, np.zeros((0, 3), dtype=np.float32))
    assert len(pairs) == 1
    assert pairs[0].is_genuine
    assert pairs[0].score == pytest.approx(1.0, abs=1e-6)


def test_skips_persons_missing_in_gallery():
    """probe 中的人 d 在 gallery 中没有模板（manual_three 缺组场景），跳过该人 genuine。"""
    rng = np.random.default_rng(1)
    probe = {
        "a": [_l2(rng.standard_normal(4).astype(np.float32))],
        "d": [_l2(rng.standard_normal(4).astype(np.float32))],
    }
    gallery = {"a": _l2(rng.standard_normal((1, 4)).astype(np.float32))}
    pairs = make_pairs(probe, gallery, np.zeros((0, 4), dtype=np.float32))
    n_genuine = sum(1 for p in pairs if p.is_genuine)
    assert n_genuine == 1  # 仅 a 算


def test_score_is_in_unit_range():
    rng = np.random.default_rng(2)
    probe = {"a": [_l2(rng.standard_normal(8).astype(np.float32))]}
    gallery = {"a": _l2(rng.standard_normal((2, 8)).astype(np.float32))}
    pairs = make_pairs(probe, gallery, _l2(rng.standard_normal((3, 8)).astype(np.float32)))
    for p in pairs:
        assert -1.0 - 1e-6 <= p.score <= 1.0 + 1e-6
