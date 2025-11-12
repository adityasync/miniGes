"""Unit tests for recognition training utilities."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_recognition import (
    collect_calibration_samples,
    remove_excluded_labels,
    split_train_val,
)


class _DummyVideoDataset(Dataset):
    """Produce synthetic video tensors shaped (T, C, H, W)."""

    def __init__(self, num_samples: int, temporal_length: int = 8, num_channels: int = 3) -> None:
        self._frames = [
            torch.randn(temporal_length, num_channels, 8, 8, dtype=torch.float32)
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)

    def __getitem__(self, index: int):
        return self._frames[index], 0


def _make_dummy_paths(tmp_path: Path, label: str, count: int) -> list[Path]:
    class_dir = tmp_path / label
    class_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(count):
        video_path = class_dir / f"sample_{idx}.mp4"
        video_path.touch()
        paths.append(video_path)
    return paths


def test_remove_excluded_labels_filters_variants(tmp_path: Path) -> None:
    class_to_paths = {
        "01.HELLO": _make_dummy_paths(tmp_path, "01.HELLO", 1),
        "WORLD": _make_dummy_paths(tmp_path, "WORLD", 1),
        "GREET": _make_dummy_paths(tmp_path, "GREET", 1),
    }

    filtered = remove_excluded_labels(class_to_paths, ["HELLO", "2. WORLD"])

    assert "01.HELLO" not in filtered
    assert "WORLD" not in filtered
    assert "GREET" in filtered


def test_split_train_val_preserves_totals(tmp_path: Path) -> None:
    random.seed(42)
    class_to_paths = {
        "HELLO": _make_dummy_paths(tmp_path, "HELLO", 5),
        "WORLD": _make_dummy_paths(tmp_path, "WORLD", 5),
    }

    train_paths, val_paths = split_train_val(class_to_paths, val_split=0.4, max_samples_per_class=None)

    assert len(train_paths) + len(val_paths) == 10
    assert len(train_paths) > 0
    assert len(val_paths) > 0

    train_labels = {path.parent.name for path in train_paths}
    val_labels = {path.parent.name for path in val_paths}
    assert train_labels <= set(class_to_paths.keys())
    assert val_labels <= set(class_to_paths.keys())


@pytest.mark.parametrize("max_samples", [0, 3])
def test_collect_calibration_samples_shapes(max_samples: int) -> None:
    dataset = _DummyVideoDataset(num_samples=4)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    samples = collect_calibration_samples(loader, max_samples)

    if max_samples == 0:
        assert samples == []
    else:
        assert len(samples) <= max_samples
        batch = samples[0]
        assert batch.ndim == 5  # (B, C, T, H, W)
        assert batch.dtype == np.float32
