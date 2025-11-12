"""Unit tests for the keypoint training utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_keypoint_model import (
    KeypointSequenceDataset,
    KeypointTrainingConfig,
    build_dataloaders,
    split_samples,
)


def _create_landmark_file(path: Path, frames: int = 5, landmarks: int = 543) -> None:
    coords = np.random.rand(frames, landmarks, 3).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coordinates=coords)


def test_keypoint_sequence_dataset_resizes(tmp_path: Path) -> None:
    file_path = tmp_path / "class_a" / "sample_0_landmarks.npz"
    _create_landmark_file(file_path, frames=4)

    dataset = KeypointSequenceDataset([(file_path, 0)], temporal_length=6, input_size=1629)
    seq, label = dataset[0]

    assert label == 0
    assert seq.shape == torch.Size([6, 1629])
    assert torch.is_tensor(seq)


def test_split_samples_generates_val_split(tmp_path: Path) -> None:
    per_class = {}
    for class_idx in range(2):
        class_name = f"class_{class_idx}"
        files = []
        for sample_idx in range(3):
            file_path = tmp_path / class_name / f"sample_{sample_idx}_landmarks.npz"
            _create_landmark_file(file_path)
            files.append(file_path)
        per_class[class_name] = files

    class_to_idx = {name: idx for idx, name in enumerate(sorted(per_class))}
    train, val = split_samples(per_class, class_to_idx, val_split=0.3, max_samples_per_class=None, seed=123)

    assert train
    assert val
    all_paths = {path for path, _ in train} | {path for path, _ in val}
    expected = {path for paths in per_class.values() for path in paths}
    assert all_paths == expected


def test_build_dataloaders_returns_sampler(tmp_path: Path) -> None:
    samples = []
    for class_idx in range(2):
        class_name = f"class_{class_idx}"
        for sample_idx in range(2):
            file_path = tmp_path / class_name / f"sample_{sample_idx}_landmarks.npz"
            _create_landmark_file(file_path)
            samples.append((file_path, class_idx))

    cfg = KeypointTrainingConfig(
        input_size=1629,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=5,
        early_stopping_patience=2,
        gradient_clip_norm=1.0,
        mixed_precision=False,
        val_split=0.2,
        max_samples_per_class=None,
        sampler_type="weighted_random_sampler",
        compute_class_weights=True,
        scheduler="cosine",
        warmup_epochs=0,
        export_onnx=False,
        temporal_length=8,
        gradient_accumulation_steps=1,
        label_smoothing=0.0,
    )

    train_loader, val_loader, class_weights, _ = build_dataloaders(
        samples,
        samples[:1],
        cfg,
        num_workers=0,
        seed=42,
    )

    assert isinstance(train_loader, DataLoader)
    assert class_weights.size > 0
    assert val_loader is not None
    assert getattr(train_loader, "sampler", None) is not None
