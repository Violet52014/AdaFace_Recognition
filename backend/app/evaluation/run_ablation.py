"""CLI 入口：跑 5 策略消融，输出 reports/<ts>/ablation.csv + roc.png。

用法 (在 backend/ 下):
    python -m app.evaluation.run_ablation --dataset dataset/ --config ../config.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config_loader import AppConfig, ConfigError
from app.evaluation.runner import AblationRow, run_ablation
from app.services.adaface_infer import is_adaface_available


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5 策略消融评估")
    p.add_argument("--dataset", type=Path, required=True, help="数据集根目录")
    p.add_argument("--config", type=Path, required=True, help="config.yaml 路径")
    return p


def _write_csv(rows: list[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _plot_roc_placeholder(rows: list[AblationRow], path: Path) -> None:
    """占位：暂以纯文字摘要替代图。matplotlib 留给后续 PR。"""
    summary = "\n".join(
        f"{r.strategy}: EER={r.eer:.4f} TAR@1e-3={r.tar_at_far_1e_3:.4f} AUC={r.auc:.4f}"
        for r in rows
    )
    path.write_text(summary, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if not args.dataset.is_dir():
        print(f"dataset 不是目录: {args.dataset}", file=sys.stderr)
        return 2

    if not is_adaface_available():
        print("未配置 AdaFace 权重，请放置到 backend/models/ 或设 ADAFACE_MODEL_PATH", file=sys.stderr)
        return 3

    try:
        cfg = AppConfig.load(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 4

    rows = run_ablation(args.dataset, cfg)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.evaluation.output_dir) / ts
    _write_csv(rows, out_dir / "ablation.csv")
    _plot_roc_placeholder(rows, out_dir / "roc_summary.txt")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
