"""集中读 config.yaml，pydantic 校验失败立即崩溃。

注意：上游 app/config.py 仍在使用（环境变量风格），本模块独立存在，
新代码统一引用 AppConfig，老代码不动。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


KNOWN_STRATEGIES = (
    "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors",
)


class ConfigError(Exception):
    pass


class _RecognitionCfg(BaseModel):
    match_threshold: float = Field(ge=0.0, le=1.0)
    production_strategy: Literal[
        "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"
    ]


class _EvaluationCfg(BaseModel):
    random_seed: int
    train_ratio: float
    min_vectors_per_person: int = Field(ge=1)
    lfw_cache_dir: str
    lfw_impostor_count: int = Field(ge=1)
    far_targets: List[float]
    output_dir: str

    @field_validator("train_ratio")
    @classmethod
    def _ratio_open_unit(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("train_ratio 必须在 (0, 1) 开区间内")
        return v


class _KmeansCfg(BaseModel):
    k: int = Field(ge=1)


class _ManualThreeCfg(BaseModel):
    pose_groups: List[str]


class _StrategiesCfg(BaseModel):
    kmeans: _KmeansCfg
    manual_three: _ManualThreeCfg


class AppConfig(BaseModel):
    recognition: _RecognitionCfg
    evaluation: _EvaluationCfg
    strategies: _StrategiesCfg

    @classmethod
    def load(cls, path: Path | str) -> "AppConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"配置文件不存在: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cls.model_validate(data)
        except (yaml.YAMLError, ValidationError) as e:
            raise ConfigError(f"配置文件解析失败 {p}: {e}") from e
