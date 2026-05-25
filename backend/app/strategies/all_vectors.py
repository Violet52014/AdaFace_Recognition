from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class AllVectorsStrategy:
    name = "all_vectors"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
