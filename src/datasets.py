"""Dataset utilities for Indian Sign Language video classification."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional dependency guard
    A = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class SignVideoDataset(Dataset):
    """Dataset that loads ISL videos, samples frames, and returns tensors."""

    def __init__(
        self,
        video_paths: List[Path],
        class_to_idx: Dict[str, int],
        frame_size: Tuple[int, int],
        num_frames: int,
        cache: bool = True,
        train: bool = True,
        augment: bool = False,
        normalize: bool = False,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        temporal_jitter: bool = False,
        clip_stride: int = 1,
    ) -> None:
        self.samples: List[Tuple[Path, int]] = []
        for video_path in video_paths:
            label_name = video_path.parent.name
            if label_name not in class_to_idx:
                LOGGER.warning("Skipping %s because label %s is unknown", video_path, label_name)
                continue
            self.samples.append((video_path, class_to_idx[label_name]))

        if not self.samples:
            raise RuntimeError("No valid video samples found for training.")

        self.frame_size = frame_size
        self.num_frames = num_frames
        self.class_to_idx = class_to_idx
        self.cache_enabled = cache and not augment
        self._cache: Dict[Path, torch.Tensor] = {}
        self.train = train
        self.normalize = normalize
        self.mean = np.array(mean if mean else (0.0, 0.0, 0.0), dtype=np.float32)
        self.std = np.array(std if std else (1.0, 1.0, 1.0), dtype=np.float32)
        self.temporal_jitter = temporal_jitter
        self.clip_stride = max(1, int(clip_stride))

        self.augmentor = None
        if augment and train:
            if A is None:
                LOGGER.warning("Albumentations not installed; augmentation disabled.")
            else:
                self.augmentor = A.Compose(
                    [
                        A.HorizontalFlip(p=0.2),
                        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-10, 10), p=0.5),
                        A.RandomBrightnessContrast(p=0.3),
                        A.GaussNoise(var_limit=(1.0, 5.0), p=0.2),
                    ]
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label_idx = self.samples[index]

        if self.cache_enabled and video_path in self._cache:
            frames_tensor = self._cache[video_path]
        else:
            frames_tensor = self._load_video_frames(video_path)
            if self.cache_enabled:
                self._cache[video_path] = frames_tensor

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return frames_tensor, label_tensor

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        frames: List[np.ndarray] = []
        success, frame = capture.read()
        while success:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = capture.read()
        capture.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        frame_count = len(frames)
        frame_indices = self._select_frame_indices(frame_count)
        sampled = [frames[idx] for idx in frame_indices]

        processed_frames: List[np.ndarray] = []
        for frame in sampled:
            frame = frame.astype(np.float32) / 255.0
            if self.augmentor is not None:
                augmented = self.augmentor(image=frame)
                frame = augmented["image"]
            if self.normalize:
                frame = (frame - self.mean) / self.std
            processed_frames.append(frame)

        frames_np = np.stack(processed_frames)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
        return frames_tensor

    def _select_frame_indices(self, available_frames: int) -> List[int]:
        if available_frames <= 0:
            raise RuntimeError("Video contains no frames for sampling.")

        target = self.num_frames
        stride = self.clip_stride

        if stride > 1 and available_frames >= target * stride:
            max_start = available_frames - target * stride
            if self.train and self.temporal_jitter and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2 if max_start > 0 else 0
            indices = [start + i * stride for i in range(target)]
        else:
            bases = np.linspace(0, available_frames - 1, target, dtype=int)
            indices = bases.tolist()

        if len(indices) < target:
            indices.extend([indices[-1]] * (target - len(indices)))

        return indices

    @property
    def class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=self.class_to_idx.get)
