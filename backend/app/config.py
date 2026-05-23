from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)

ADAFACE_ROOT = BASE_DIR / "AdaFace"
MODELS_DIR = BASE_DIR / "models"


def _resolve_adaface_model_path() -> Optional[Path]:
    env = os.environ.get("ADAFACE_MODEL_PATH", "").strip()
    if env:
        p = Path(env)
        return p if p.is_file() else None
    if not MODELS_DIR.is_dir():
        return None
    for pattern in ("*.ckpt", "*.pth", "*.pt"):
        found = sorted(MODELS_DIR.glob(pattern))
        if found:
            return found[0]
    return None


class Config:
    SECRET_KEY = "dev-secret-key"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{INSTANCE_DIR / 'face_access.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AdaFace：权重放在 backend/models/，或通过环境变量 ADAFACE_MODEL_PATH 指定绝对路径
    ADAFACE_ROOT = ADAFACE_ROOT
    ADAFACE_MODEL_PATH = _resolve_adaface_model_path()
    ADAFACE_ARCH = os.environ.get("ADAFACE_ARCH", "ir_50")
    # 余弦相似度阈值（向量已 L2 归一化时，点积即余弦）
    ADAFACE_MATCH_THRESHOLD = float(os.environ.get("ADAFACE_MATCH_THRESHOLD", "0.35"))
    # 留空则自动选 cuda:0 / cpu；也可设为 cuda:0 或 cpu
    ADAFACE_DEVICE = os.environ.get("ADAFACE_DEVICE", "").strip() or None
