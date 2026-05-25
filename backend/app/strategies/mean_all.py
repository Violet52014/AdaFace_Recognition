from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class MeanAllStrategy:
    name = "mean_all"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
