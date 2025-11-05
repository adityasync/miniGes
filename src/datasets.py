"""Dataset utilities for Indian Sign Language video classification."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        augmentations_config: Optional[Dict[str, Any]] = None,
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
                self.augmentor = self._build_augmentor(augmentations_config or {})

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

    def _build_augmentor(self, augmentations_config: Dict[str, Any]) -> Optional[Any]:
        if A is None:
            return None

        def _range(value, default):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return float(value[0]), float(value[1])
            return default

        def _offset_range(value, default):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return float(value[0]) - 1.0, float(value[1]) - 1.0
            return default

        transforms = []

        horizontal_flip = augmentations_config.get("horizontal_flip", True)
        flip_prob = augmentations_config.get("horizontal_flip_prob", 0.2)
        if isinstance(horizontal_flip, (int, float)):
            flip_prob = float(horizontal_flip)
            horizontal_flip = True
        if horizontal_flip:
            transforms.append(A.HorizontalFlip(p=float(flip_prob)))

        affine_prob = float(augmentations_config.get("affine_prob", 0.5))
        if affine_prob > 0:
            rotation_range = _range(augmentations_config.get("rotation"), (-10.0, 10.0))
            scale_range = _range(augmentations_config.get("scale"), (0.9, 1.1))
            translate_range = _range(augmentations_config.get("translate_percent"), (0.0, 0.05))
            transforms.append(
                A.Affine(
                    scale=scale_range,
                    translate_percent=translate_range,
                    rotate=rotation_range,
                    p=affine_prob,
                )
            )

        brightness_contrast_prob = float(augmentations_config.get("brightness_contrast_prob", 0.3))
        if brightness_contrast_prob > 0:
            brightness_limit = _offset_range(augmentations_config.get("brightness"), (-0.2, 0.2))
            contrast_limit = _offset_range(augmentations_config.get("contrast"), (-0.2, 0.2))
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=brightness_contrast_prob,
                )
            )

        gaussian_noise_prob = float(augmentations_config.get("gaussian_noise_prob", 0.2))
        if gaussian_noise_prob > 0:
            noise_range = _range(augmentations_config.get("gaussian_noise"), (1.0, 5.0))
            transforms.append(A.GaussNoise(var_limit=noise_range, p=gaussian_noise_prob))

        if not transforms:
            return None

        return A.Compose(transforms)

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
