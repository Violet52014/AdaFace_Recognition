"""
从目录结构 dataset/<姓名>/*.jpg 批量提取 AdaFace 特征并写入数据库表 face_profiles。

用法（在 backend 目录下）:
  python scripts/build_face_gallery.py --dataset D:/data/face_dataset

依赖: 已安装 AdaFace 依赖，且 backend/models 下存在 .pth/.ckpt 权重。
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
from app.extensions import db  # noqa: E402
from app.models import FaceProfile  # noqa: E402
from app.services.adaface_infer import extract_embedding_from_bgr  # noqa: E402
from app.services.image_utils import imread_bgr_unicode  # noqa: E402


def iter_images(person_dir: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for pattern in exts:
        yield from sorted(person_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="从文件夹数据集构建人脸库特征")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="根目录，其下每个子文件夹名为人员姓名，内含多张人脸图",
    )
    parser.add_argument("--min-images", type=int, default=1, help="每人至少成功提取特征的图片数")
    parser.add_argument(
        "--class-name",
        default="",
        help="统一写入到 FaceProfile.class_name（可选）",
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print("dataset 不是有效目录:", args.dataset)
        sys.exit(1)

    app = create_app()
    with app.app_context():
        for person_dir in sorted(p for p in args.dataset.iterdir() if p.is_dir()):
            name = person_dir.name
            vectors = []
            for img_path in iter_images(person_dir):
                img = imread_bgr_unicode(img_path)
                if img is None:
                    print("无法读取:", img_path)
                    continue
                emb, err, _bbox = extract_embedding_from_bgr(img)
                if err:
                    print(f"跳过 {img_path}: {err}")
                    continue
                vectors.append(emb)

            if len(vectors) < args.min_images:
                print(f"跳过 {name}: 有效特征仅 {len(vectors)} 张，需要 >= {args.min_images}")
                continue

            stacked = np.stack(vectors, axis=0)
            mean_emb = np.mean(stacked, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

            vec_json = json.dumps(mean_emb.astype(float).tolist())
            profile = FaceProfile.query.filter_by(name=name).first()
            if profile:
                profile.feature_vector = vec_json
                if args.class_name:
                    profile.class_name = args.class_name
            else:
                profile = FaceProfile(
                    name=name,
                    class_name=args.class_name or None,
                    feature_vector=vec_json,
                )
                db.session.add(profile)
            db.session.commit()
            print(f"已录入: {name}, 使用 {len(vectors)} 张图, dim={mean_emb.shape[0]}")

    print("完成。")


if __name__ == "__main__":
    main()
