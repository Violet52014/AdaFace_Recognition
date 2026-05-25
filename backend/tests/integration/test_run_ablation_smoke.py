from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from app.config_loader import AppConfig


SAMPLE_DIR = Path(__file__).resolve().parents[2] / "AdaFace" / "face_alignment" / "test_images"


@pytest.mark.slow
def test_ablation_runs_end_to_end_on_tiny_fixture(tmp_path, monkeypatch):
    samples = sorted(SAMPLE_DIR.glob("*.jpeg"))
    assert len(samples) >= 3, f"AdaFace sample 缺失: {SAMPLE_DIR}"

    # 构造 mini dataset：3 人 × 2 张同图（足够提取特征即可）
    dataset = tmp_path / "dataset"
    for i, src in enumerate(samples[:3]):
        for j in range(2):
            dst = dataset / f"person_{i}" / f"img_{j}.jpeg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    # 模拟 LFW 缓存：3 张 sample 各放在不同子目录
    lfw_dir = tmp_path / "lfw"
    for i, src in enumerate(samples[:3]):
        d = lfw_dir / f"L{i}"
        d.mkdir(parents=True)
        shutil.copy(src, d / src.name)

    # 写一份临时 config.yaml
    cfg_text = f"""
recognition:
  match_threshold: 0.35
  production_strategy: mean_all
evaluation:
  random_seed: 42
  train_ratio: 0.5
  min_vectors_per_person: 1
  lfw_cache_dir: {lfw_dir}
  lfw_impostor_count: 3
  far_targets: [1.0e-3]
  output_dir: {tmp_path / "reports"}
strategies:
  kmeans:
    k: 3
  manual_three:
    pose_groups: [frontal, left, right]
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    from app.evaluation.run_ablation import main
    code = main(["--dataset", str(dataset), "--config", str(cfg_file)])
    assert code == 0

    # 找输出目录
    runs = sorted((tmp_path / "reports").iterdir())
    assert len(runs) == 1
    csv_path = runs[0] / "ablation.csv"
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    strategies_found = {r["strategy"] for r in rows}
    assert strategies_found == {"random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"}
    for r in rows:
        # 数值字段必须可解析为浮点
        assert 0.0 <= float(r["eer"]) <= 1.0
        assert 0.0 <= float(r["auc"]) <= 1.0
        assert int(r["n_pairs"]) > 0
