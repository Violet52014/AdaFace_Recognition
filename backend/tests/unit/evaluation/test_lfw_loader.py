from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.lfw_loader import sample_lfw_paths


def _make_lfw_layout(root: Path, persons: dict[str, int]) -> None:
    for name, n in persons.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{name}_{i:04d}.jpg").write_bytes(b"\x00")


def test_samples_exact_n_when_enough(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"George_Bush": 50, "Tony_Blair": 30})
    paths = sample_lfw_paths(tmp_path, n_images=20, seed=42, exclude_names=set())
    assert len(paths) == 20


def test_excludes_overlapping_names(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"alice": 10, "bob": 10})
    paths = sample_lfw_paths(tmp_path, n_images=10, seed=42, exclude_names={"alice"})
    assert all("alice" not in str(p) for p in paths)


def test_deterministic_given_seed(tmp_path: Path):
    _make_lfw_layout(tmp_path, {f"p{i}": 5 for i in range(10)})
    a = sample_lfw_paths(tmp_path, n_images=10, seed=7, exclude_names=set())
    b = sample_lfw_paths(tmp_path, n_images=10, seed=7, exclude_names=set())
    assert [str(p) for p in a] == [str(p) for p in b]


def test_returns_all_when_n_exceeds_available(tmp_path: Path):
    _make_lfw_layout(tmp_path, {"x": 3})
    paths = sample_lfw_paths(tmp_path, n_images=100, seed=42, exclude_names=set())
    assert len(paths) == 3


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        sample_lfw_paths(tmp_path / "no_such", n_images=10, seed=42, exclude_names=set())
