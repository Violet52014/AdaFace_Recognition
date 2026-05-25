"""从目录结构 dataset/<姓名>/{frontal,left,right}/*.jpg 或 dataset/<姓名>/*.jpg
批量提取 AdaFace 特征，运行 5 个 TemplateStrategy，写入 face_profiles + templates 两张表。

用法（在 backend 目录下）:
  python scripts/build_face_gallery.py --dataset dataset/ --config ../config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from app.config_loader import AppConfig  # noqa: E402
from app.evaluation.data_split import POSE_NAMES  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import FaceProfile, Template  # noqa: E402
from app.services.adaface_infer import extract_embedding_from_bgr  # noqa: E402
from app.services.image_utils import imread_bgr_unicode  # noqa: E402
from app.strategies import STRATEGIES  # noqa: E402


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_person_images(person_dir: Path):
    """遍历直接图片 + frontal/left/right 子目录图片，yield (path, pose_or_None)"""
    for p in sorted(person_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p, None
        elif p.is_dir() and p.name in POSE_NAMES:
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in IMG_EXTS:
                    yield q, p.name


def _extract_person(person_dir: Path):
    vectors = []
    poses = []
    for img_path, pose in _iter_person_images(person_dir):
        img = imread_bgr_unicode(img_path)
        if img is None:
            print(f"无法读取: {img_path}")
            continue
        emb, err, _ = extract_embedding_from_bgr(img)
        if err:
            print(f"跳过 {img_path}: {err}")
            continue
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        vectors.append(emb.astype(np.float32))
        poses.append(pose if pose else "")
    if not vectors:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.stack(vectors, axis=0), poses


def _upsert_profile(name: str, mean_vec: np.ndarray, class_name: str | None) -> FaceProfile:
    p = FaceProfile.query.filter_by(name=name).first()
    fv = json.dumps(mean_vec.astype(float).tolist())
    if p:
        p.feature_vector = fv
        if class_name:
            p.class_name = class_name
    else:
        p = FaceProfile(name=name, class_name=class_name or None, feature_vector=fv)
        db.session.add(p)
    db.session.flush()
    return p


def _upsert_template(profile_id: int, strategy: str, mat: np.ndarray, source_count: int) -> None:
    payload = mat[0].tolist() if mat.shape[0] == 1 else mat.tolist()
    vector_json = json.dumps(payload)
    existing = Template.query.filter_by(profile_id=profile_id, strategy=strategy).first()
    if existing:
        existing.vector_json = vector_json
        existing.source_count = source_count
    else:
        db.session.add(Template(
            profile_id=profile_id,
            strategy=strategy,
            vector_json=vector_json,
            source_count=source_count,
        ))


def main() -> None:
    ap = argparse.ArgumentParser(description="从文件夹数据集构建 5 策略人脸库")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config.yaml")
    ap.add_argument("--min-images", type=int, default=1)
    ap.add_argument("--class-name", default="")
    args = ap.parse_args()

    if not args.dataset.is_dir():
        print("dataset 不是目录:", args.dataset)
        sys.exit(2)

    cfg = AppConfig.load(args.config)
    rng_seed = cfg.evaluation.random_seed

    app = create_app()
    with app.app_context():
        for person_dir in sorted(p for p in args.dataset.iterdir() if p.is_dir()):
            name = person_dir.name
            if name == "lfw":
                continue
            vectors, poses = _extract_person(person_dir)
            if vectors.shape[0] < args.min_images:
                print(f"跳过 {name}: 仅 {vectors.shape[0]} 张可用")
                continue

            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
            profile = _upsert_profile(name, mean_vec, args.class_name)

            rng = np.random.default_rng(rng_seed)
            for strat_name, strat in STRATEGIES.items():
                try:
                    if strat_name == "manual_three":
                        labels = np.array(poses, dtype=object)
                        if not any(p in {"frontal", "left", "right"} for p in poses):
                            print(f"  跳过 {name} 的 manual_three（无姿态标签）")
                            continue
                        mat = strat.build(vectors, rng=rng, group_labels=labels)
                    else:
                        mat = strat.build(vectors, rng=rng)
                except Exception as exc:
                    print(f"  策略 {strat_name} 对 {name} 失败: {exc}")
                    continue
                _upsert_template(profile.id, strat_name, mat, source_count=vectors.shape[0])

            db.session.commit()
            print(f"已录入 {name}：{vectors.shape[0]} 张图，5 策略模板已更新")

    print("完成。")


if __name__ == "__main__":
    main()
