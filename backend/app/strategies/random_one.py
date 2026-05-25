from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class RandomOneStrategy:
    name = "random_one"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
