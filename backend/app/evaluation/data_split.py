"""按人切分数据集为 train/probe，识别姿态子目录。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)

POSE_NAMES = frozenset({"frontal", "left", "right"})

# 评估时不可作为"人脸库人员"的特殊子目录名
RESERVED_SUBDIRS = frozenset({"lfw"})

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ImageEntry:
    path: Path
    pose: Optional[str]


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _collect_images_for_person(person_dir: Path) -> List[ImageEntry]:
    """递归发现 person_dir 下的所有图片，pose=直接父目录名（若在 POSE_NAMES 内）。"""
    entries: List[ImageEntry] = []
    # 直接子文件
    for p in sorted(person_dir.iterdir()):
        if _is_image(p):
            entries.append(ImageEntry(path=p, pose=None))
        elif p.is_dir() and p.name in POSE_NAMES:
            for q in sorted(p.iterdir()):
                if _is_image(q):
                    entries.append(ImageEntry(path=q, pose=p.name))
    return entries


def split_by_person(
    dataset_root: Path,
    *,
    train_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[ImageEntry], List[ImageEntry]]]:
    """每人切 train/probe；同 seed 同结果；N==1 时全部入 train。

    跳过 RESERVED_SUBDIRS（如 lfw）。
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Tuple[List[ImageEntry], List[ImageEntry]]] = {}
    for person_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if person_dir.name in RESERVED_SUBDIRS:
            continue
        entries = _collect_images_for_person(person_dir)
        if not entries:
            log.warning("人 %s 无可用图片，跳过", person_dir.name)
            continue
        n = len(entries)
        if n == 1:
            log.warning("人 %s 仅 1 张图，全部入 train，probe 为空", person_dir.name)
            out[person_dir.name] = (entries, [])
            continue
        idx = rng.permutation(n)
        n_train = max(1, int(round(n * train_ratio)))
        n_train = min(n_train, n - 1)  # 至少留 1 张做 probe
        train = [entries[i] for i in idx[:n_train]]
        probe = [entries[i] for i in idx[n_train:]]
        out[person_dir.name] = (train, probe)
    return out
