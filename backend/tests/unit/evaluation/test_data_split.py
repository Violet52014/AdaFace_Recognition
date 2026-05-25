from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.data_split import ImageEntry, split_by_person


def _make_dataset_old(root: Path, persons: dict[str, int]) -> None:
    """旧布局: root/<name>/img_xx.jpg"""
    for name, n in persons.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i:02d}.jpg").write_bytes(b"\x00")


def _make_dataset_pose(root: Path, persons: dict[str, dict[str, int]]) -> None:
    """新布局: root/<name>/<pose>/img_xx.jpg"""
    for name, poses in persons.items():
        for pose, n in poses.items():
            d = root / name / pose
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"img_{i:02d}.jpg").write_bytes(b"\x00")


def test_split_old_layout_returns_pose_none(tmp_path: Path):
    _make_dataset_old(tmp_path, {"alice": 10})
    out = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    train, probe = out["alice"]
    assert all(isinstance(e, ImageEntry) for e in train + probe)
    assert all(e.pose is None for e in train + probe)


def test_split_pose_layout_records_pose(tmp_path: Path):
    _make_dataset_pose(tmp_path, {"bob": {"frontal": 4, "left": 4, "right": 4}})
    out = split_by_person(tmp_path, train_ratio=0.5, seed=42)
    train, probe = out["bob"]
    poses = {e.pose for e in train + probe}
    assert poses == {"frontal", "left", "right"}


def test_split_ratio_approximate(tmp_path: Path):
    _make_dataset_old(tmp_path, {"x": 10})
    train, probe = split_by_person(tmp_path, train_ratio=0.8, seed=42)["x"]
    assert len(train) == 8
    assert len(probe) == 2


def test_split_deterministic_given_seed(tmp_path: Path):
    _make_dataset_old(tmp_path, {"x": 10})
    a = split_by_person(tmp_path, train_ratio=0.7, seed=42)["x"]
    b = split_by_person(tmp_path, train_ratio=0.7, seed=42)["x"]
    assert [str(e.path) for e in a[0]] == [str(e.path) for e in b[0]]
    assert [str(e.path) for e in a[1]] == [str(e.path) for e in b[1]]


def test_single_image_person_all_in_train(tmp_path: Path, caplog):
    _make_dataset_old(tmp_path, {"loner": 1})
    train, probe = split_by_person(tmp_path, train_ratio=0.8, seed=42)["loner"]
    assert len(train) == 1
    assert probe == []


def test_skips_lfw_subdir(tmp_path: Path):
    _make_dataset_old(tmp_path, {"alice": 4})
    # 模拟 dataset/lfw 缓存子目录，必须被跳过
    (tmp_path / "lfw" / "George").mkdir(parents=True)
    (tmp_path / "lfw" / "George" / "george_01.jpg").write_bytes(b"\x00")
    out = split_by_person(tmp_path, train_ratio=0.5, seed=42)
    assert "lfw" not in out
    assert "alice" in out
