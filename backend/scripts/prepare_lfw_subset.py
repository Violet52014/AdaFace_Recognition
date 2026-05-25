"""从已解压的 LFW 数据集挑选 gallery 人 + impostor 人，复制到约定布局。

输入：lfw 原始解压目录（每人一个子目录，里面若干 .jpg）
输出：
  - dataset_root/<gallery_name>/img_xx.jpg     —— gallery 人（默认 35 人，N≥20 张）
  - dataset_root/lfw/<impostor_name>/...       —— impostor 人（默认 10 人）

用法（在 backend 下）:
    python scripts/prepare_lfw_subset.py \
        --lfw-src dataset/lfw_raw/lfw_funneled \
        --dataset-root dataset \
        --n-gallery 35 --min-images 20 --n-impostor 10 --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def _scan_persons(lfw_src: Path) -> List[Tuple[str, List[Path]]]:
    out: List[Tuple[str, List[Path]]] = []
    for d in sorted(p for p in lfw_src.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        if imgs:
            out.append((d.name, imgs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从 LFW 准备 gallery + impostor 子集")
    ap.add_argument("--lfw-src", type=Path, required=True,
                    help="LFW 解压根目录（如 dataset/lfw_raw/lfw_funneled）")
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="目标数据集根目录（如 dataset）")
    ap.add_argument("--n-gallery", type=int, default=35)
    ap.add_argument("--min-images", type=int, default=20,
                    help="gallery 候选人最小图片数")
    ap.add_argument("--n-impostor", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true",
                    help="用软链而不是复制（省空间，但跨设备/移动数据集可能失效）")
    args = ap.parse_args()

    if not args.lfw_src.is_dir():
        print(f"LFW 源目录不存在: {args.lfw_src}", file=sys.stderr)
        return 2

    persons = _scan_persons(args.lfw_src)
    if not persons:
        print(f"未在 {args.lfw_src} 找到任何人子目录", file=sys.stderr)
        return 2

    eligible = [(n, imgs) for n, imgs in persons if len(imgs) >= args.min_images]
    print(f"扫到 {len(persons)} 人，其中图片数 ≥ {args.min_images} 的 {len(eligible)} 人")
    if len(eligible) < args.n_gallery:
        print(f"合格人数不足 {args.n_gallery}，请降 --min-images 或减小 --n-gallery", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    eligible_sorted = sorted(eligible, key=lambda x: (-len(x[1]), x[0]))
    gallery = eligible_sorted[: args.n_gallery]
    gallery_names = {n for n, _ in gallery}

    impostor_pool = [(n, imgs) for n, imgs in persons if n not in gallery_names]
    rng.shuffle(impostor_pool)
    impostors = impostor_pool[: args.n_impostor]

    def _place(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy(src, dst)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    for name, imgs in gallery:
        for i, src in enumerate(imgs):
            _place(src, args.dataset_root / name / f"img_{i:03d}.jpg")
    print(f"gallery: {len(gallery)} 人已写入 {args.dataset_root}/<name>/")

    lfw_dir = args.dataset_root / "lfw"
    for name, imgs in impostors:
        for src in imgs:
            _place(src, lfw_dir / name / src.name)
    print(f"impostor: {len(impostors)} 人已写入 {lfw_dir}/<name>/")
    print(f"完成。{'软链' if args.symlink else '复制'}模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
