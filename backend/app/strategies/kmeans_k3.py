from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class KmeansK3Strategy:
    name = "kmeans_k3"
    k: int = 3
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
