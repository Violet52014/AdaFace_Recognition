"""构造 genuine / impostor pair；score = 多模板 max-cosine。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pair:
    score: float
    is_genuine: bool


def _max_cosine(query: np.ndarray, templates: np.ndarray) -> float:
    """query (D,) 与 templates (M, D)，假设两端均 L2 归一化，余弦=点积。"""
    if templates.shape[0] == 0:
        return float("-inf")
    sims = templates @ query
    return float(sims.max())


def make_pairs(
    probe_features: Dict[str, List[np.ndarray]],
    gallery_templates: Dict[str, np.ndarray],
    lfw_features: np.ndarray,
) -> List[Pair]:
    """三类 pair：
        genuine:           probe_i  vs gallery[name_i]
        impostor_internal: probe_i  vs gallery[name_j]   for j != i
        impostor_lfw:      lfw_k    vs gallery[name_j]   for all j
    """
    pairs: List[Pair] = []
    names = list(gallery_templates.keys())

    for name, probes in probe_features.items():
        for q in probes:
            if name in gallery_templates:
                pairs.append(Pair(
                    score=_max_cosine(q, gallery_templates[name]),
                    is_genuine=True,
                ))
            else:
                log.warning("probe 人 %s 不在 gallery 中，跳过其 genuine pair", name)
            for other in names:
                if other == name:
                    continue
                pairs.append(Pair(
                    score=_max_cosine(q, gallery_templates[other]),
                    is_genuine=False,
                ))

    for k in range(lfw_features.shape[0]):
        q = lfw_features[k]
        for other in names:
            pairs.append(Pair(
                score=_max_cosine(q, gallery_templates[other]),
                is_genuine=False,
            ))

    return pairs
