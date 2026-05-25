"""加载 LFW 子集作为 impostor 来源。

LFW 数据集本身不在仓库内：用户需要预先放置到 cache_dir，或自己解压官方
funneled tar.gz（http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz）。
本模块只做"抽样选择"，不实现下载（避免脚本调用网络的不确定性）。
若 cache_dir 不存在，run_ablation 会打印明确指引让用户手动准备。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Set

import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sample_lfw_paths(
    cache_dir: Path,
    *,
    n_images: int,
    seed: int,
    exclude_names: Set[str],
) -> List[Path]:
    """LFW 布局 cache_dir/<person>/<image>.jpg；不在 exclude_names 中的子目录被纳入。"""
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"LFW 缓存目录不存在: {cache_dir}")
    candidates: List[Path] = []
    for person_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        if person_dir.name in exclude_names:
            continue
        for img in sorted(person_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                candidates.append(img)
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    if n_images >= len(candidates):
        return candidates
    idx = rng.permutation(len(candidates))[:n_images]
    return [candidates[i] for i in sorted(idx)]
