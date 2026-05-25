from __future__ import annotations

from pathlib import Path

import pytest

from app.config_loader import AppConfig, ConfigError


VALID_YAML = """
recognition:
  match_threshold: 0.35
  production_strategy: mean_all
evaluation:
  random_seed: 42
  train_ratio: 0.8
  min_vectors_per_person: 5
  lfw_cache_dir: backend/dataset/lfw
  lfw_impostor_count: 1000
  far_targets: [1.0e-3]
  output_dir: reports
"""


def test_loads_valid_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(VALID_YAML, encoding="utf-8")
    cfg = AppConfig.load(cfg_file)
    assert cfg.recognition.match_threshold == pytest.approx(0.35)
    assert cfg.recognition.production_strategy == "mean_all"
    assert cfg.evaluation.random_seed == 42
    assert cfg.evaluation.far_targets == [1e-3]


def test_threshold_must_be_in_range(tmp_path: Path):
    bad = VALID_YAML.replace("match_threshold: 0.35", "match_threshold: 1.5")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)


def test_production_strategy_must_be_known(tmp_path: Path):
    bad = VALID_YAML.replace("production_strategy: mean_all", "production_strategy: foo")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        AppConfig.load(tmp_path / "nope.yaml")


def test_train_ratio_must_be_open_unit_interval(tmp_path: Path):
    bad = VALID_YAML.replace("train_ratio: 0.8", "train_ratio: 1.0")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_file)
