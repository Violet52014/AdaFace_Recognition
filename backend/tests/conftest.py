"""测试公共 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 `from app.xxx import yyy` 在 pytest 中可用，与 scripts/build_face_gallery.py 同款
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_vectors():
    """构造 (N, 512) 已 L2 归一化的合成向量。"""
    def _make(n: int = 10, dim: int = 512, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, dim)).astype(np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return v
    return _make
