"""run_ablation 的执行核心：抽特征 → 5 策略 build → 评估，输出每策略一行结果。

设计：
- 一次性抽取所有 train / probe / lfw 特征（最贵的一步）
- 5 策略循环复用同一份特征
- 输出 dataclass AblationRow，由 CLI 落 CSV
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.config_loader import AppConfig
from app.evaluation.data_split import ImageEntry, split_by_person
from app.evaluation.lfw_loader import sample_lfw_paths
from app.evaluation.metrics import auc, eer, roc_curve, tar_at_far
from app.evaluation.pair_generator import make_pairs
from app.services.adaface_infer import extract_embedding_from_bgr
from app.services.image_utils import imread_bgr_unicode
from app.strategies import STRATEGIES, get_strategy


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationRow:
    strategy: str
    eer: float
    eer_threshold: float
    tar_at_far_1e_3: float
    auc: float
    n_pairs: int
    n_genuine: int
    n_impostor_internal: int
    n_impostor_lfw: int


def _extract_features(
    entries: List[ImageEntry],
) -> Tuple[np.ndarray, List[str]]:
    """对 entries 抽特征；返回 (features (N,512), pose_labels)。失败的图被跳过。"""
    feats: List[np.ndarray] = []
    poses: List[str] = []
    for e in entries:
        img = imread_bgr_unicode(e.path)
        if img is None:
            log.warning("无法读取 %s，跳过", e.path)
            continue
        emb, err, _ = extract_embedding_from_bgr(img)
        if err or emb is None:
            log.warning("特征提取失败 %s: %s", e.path, err)
            continue
        # 上游 emb 已 L2，但保险再做一次
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        feats.append(emb.astype(np.float32))
        poses.append(e.pose if e.pose else "")
    if not feats:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.stack(feats, axis=0), poses


def _evaluate_one_strategy(
    strategy_name: str,
    train_per_person: Dict[str, Tuple[np.ndarray, List[str]]],
    probe_per_person: Dict[str, List[np.ndarray]],
    lfw_features: np.ndarray,
    *,
    rng_seed: int,
    target_far: float,
) -> AblationRow:
    strategy = get_strategy(strategy_name)
    rng = np.random.default_rng(rng_seed)

    gallery: Dict[str, np.ndarray] = {}
    for name, (vecs, poses) in train_per_person.items():
        if vecs.shape[0] == 0:
            log.warning("策略 %s：人 %s 无可用 train 向量，剔除", strategy_name, name)
            continue
        try:
            if strategy_name == "manual_three":
                labels = np.array(poses, dtype=object)
                if not any(p in {"frontal", "left", "right"} for p in poses):
                    log.warning("策略 %s：人 %s 无姿态标签，剔除", strategy_name, name)
                    continue
                templates = strategy.build(vecs, rng=rng, group_labels=labels)
            else:
                templates = strategy.build(vecs, rng=rng)
            gallery[name] = templates
        except Exception as exc:  # 单人策略失败，剔除该人但不阻断整体
            log.warning("策略 %s：人 %s build 失败: %s", strategy_name, name, exc)

    pairs = make_pairs(probe_per_person, gallery, lfw_features)
    if not pairs:
        raise ValueError(f"策略 {strategy_name}：未生成任何 pair")

    scores = np.array([p.score for p in pairs], dtype=np.float64)
    labels = np.array([1 if p.is_genuine else 0 for p in pairs], dtype=np.int64)
    fpr, tpr, thr = roc_curve(scores, labels)
    eer_v, eer_thr = eer(fpr, tpr, thr)

    n_genuine = int(labels.sum())
    n_impostor = len(labels) - n_genuine
    # 拆分 cross-person vs lfw 数（按构造顺序：先 probe×内部，再 lfw×names；近似估算）
    names = list(gallery.keys())
    n_impostor_lfw = int(lfw_features.shape[0]) * len(names)
    n_impostor_internal = n_impostor - n_impostor_lfw

    return AblationRow(
        strategy=strategy_name,
        eer=float(eer_v),
        eer_threshold=float(eer_thr),
        tar_at_far_1e_3=float(tar_at_far(fpr, tpr, target_far)),
        auc=float(auc(fpr, tpr)),
        n_pairs=len(pairs),
        n_genuine=n_genuine,
        n_impostor_internal=n_impostor_internal,
        n_impostor_lfw=n_impostor_lfw,
    )


def run_ablation(
    dataset_root: Path,
    cfg: AppConfig,
) -> List[AblationRow]:
    """跑 5 策略消融。返回每策略一行。

    副作用：log warning；不写文件。文件落地在 CLI 层。
    """
    splits = split_by_person(
        dataset_root,
        train_ratio=cfg.evaluation.train_ratio,
        seed=cfg.evaluation.random_seed,
    )
    if len(splits) < 2:
        raise ValueError(f"评估至少需要 2 个人，当前 {len(splits)}")

    log.info("抽取 train/probe 特征中（%d 人）...", len(splits))
    train_per_person: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    probe_per_person: Dict[str, List[np.ndarray]] = {}
    min_n = cfg.evaluation.min_vectors_per_person
    for name, (train_entries, probe_entries) in splits.items():
        train_vecs, train_poses = _extract_features(train_entries)
        probe_vecs, _ = _extract_features(probe_entries)
        if train_vecs.shape[0] < min_n:
            log.warning("人 %s train=%d < %d，整人剔除", name, train_vecs.shape[0], min_n)
            continue
        train_per_person[name] = (train_vecs, train_poses)
        probe_per_person[name] = [probe_vecs[i] for i in range(probe_vecs.shape[0])]

    if len(train_per_person) < 2:
        raise ValueError(f"过滤后 train 人数不足，仅 {len(train_per_person)} 人")

    log.info("准备 LFW impostor 特征...")
    lfw_paths = sample_lfw_paths(
        Path(cfg.evaluation.lfw_cache_dir),
        n_images=cfg.evaluation.lfw_impostor_count,
        seed=cfg.evaluation.random_seed,
        exclude_names=set(train_per_person.keys()),
    )
    lfw_entries = [ImageEntry(path=p, pose=None) for p in lfw_paths]
    lfw_vecs, _ = _extract_features(lfw_entries)
    log.info("LFW impostor 实抽到 %d 张", lfw_vecs.shape[0])

    target_far = cfg.evaluation.far_targets[0]
    rows: List[AblationRow] = []
    for name in STRATEGIES.keys():
        log.info("评估策略: %s", name)
        rows.append(_evaluate_one_strategy(
            name,
            train_per_person,
            probe_per_person,
            lfw_vecs,
            rng_seed=cfg.evaluation.random_seed,
            target_far=target_far,
        ))
    return rows
