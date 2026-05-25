from __future__ import annotations
import numpy as np
from .base import TemplateStrategy

class ManualThreeStrategy:
    name = "manual_three"
    def build(self, vectors, *, rng, group_labels=None):
        raise NotImplementedError
