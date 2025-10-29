"""Data preprocessing pipeline for the sinGes-mini project.

The INCLUDE dataset contains short video clips of 30 Indian Sign Language words.
This module coordinates frame extraction, landmark detection, augmentation, and
train/validation/test splitting. Implementations are intentionally scaffolded
with TODO markers to guide incremental development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional until installed
    mp = None  # type: ignore

from src.utils import load_config, resolve_path, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    dataset_root: Path
    raw_output_dir: Path
    processed_output_dir: Path
    splits_dir: Path
    mediapipe_output_dir: Path
    frame_rate: int
    frame_resize: tuple[int, int]
    max_frames_per_clip: int
    use_mediapipe: bool
    store_landmarks: bool

    @classmethod
    def from_dict(cls, config: dict) -> "PreprocessingConfig":
        preprocessing = config["preprocessing"]
        paths = config["paths"]
        return cls(
            dataset_root=resolve_path(paths["dataset_root"]),
            raw_output_dir=resolve_path(paths["raw_data_dir"]),
            processed_output_dir=resolve_path(paths["processed_data_dir"]),
            splits_dir=resolve_path(paths["splits_dir"]),
            mediapipe_output_dir=resolve_path(paths["mediapipe_output_dir"]),
            frame_rate=preprocessing["frame_rate"],
            frame_resize=tuple(preprocessing["frame_resize"]),
            max_frames_per_clip=preprocessing["max_frames_per_clip"],
            use_mediapipe=preprocessing["use_mediapipe"],
            store_landmarks=preprocessing["store_landmarks"],
        )


class DataPreprocessor:
    """Coordinate the preprocessing workflow."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.config.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.splits_dir.mkdir(parents=True, exist_ok=True)
        if self.config.use_mediapipe:
            self.config.mediapipe_output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self) -> None:
        """Entry point for processing the full dataset."""

        LOGGER.info("Starting dataset preprocessing from %s", self.config.dataset_root)
        video_files = list(self.config.dataset_root.rglob("*.mp4"))
        if not video_files:
            LOGGER.warning("No video files found in dataset root %s", self.config.dataset_root)
            return

        for video_path in video_files:
            frames = self._extract_frames(video_path)
            if self.config.use_mediapipe and mp is not None:
                landmarks = self._extract_landmarks(frames)
                if self.config.store_landmarks:
                    self._save_landmarks(video_path, landmarks)
            self._save_frames(video_path, frames)

        LOGGER.info("Completed preprocessing for %d videos", len(video_files))

    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Decode video into a list of resized frames."""

        capture = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        success, frame = capture.read()
        while success and len(frames) < self.config.max_frames_per_clip:
            frame = cv2.resize(frame, self.config.frame_resize)
            frames.append(frame)
            success, frame = capture.read()
        capture.release()
        return frames

    def _extract_landmarks(self, frames: Iterable[np.ndarray]) -> List[np.ndarray]:
        """Run MediaPipe Hands on frames to obtain keypoints."""

        if mp is None:
            LOGGER.error("MediaPipe is not installed. Skipping landmark extraction.")
            return []

        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
        landmarks: List[np.ndarray] = []
        for frame in frames:
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                    landmarks.append(coords)
            else:
                landmarks.append(np.zeros((21, 3)))
        hands.close()
        return landmarks

    def _save_frames(self, video_path: Path, frames: List[np.ndarray]) -> None:
        """Persist extracted frames to disk as NumPy arrays."""

        relative = video_path.relative_to(self.config.dataset_root)
        save_path = self.config.raw_output_dir / relative.with_suffix(".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, frames=np.stack(frames))
        LOGGER.debug("Saved %s", save_path)

    def _save_landmarks(self, video_path: Path, landmarks: List[np.ndarray]) -> None:
        """Persist landmark arrays."""

        if not landmarks:
            return

        relative = video_path.relative_to(self.config.dataset_root)
        save_path = self.config.mediapipe_output_dir / relative.with_suffix("_landmarks.npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, landmarks=np.stack(landmarks))
        LOGGER.debug("Saved landmarks %s", save_path)


def main() -> None:
    """CLI entry point for preprocessing."""

    setup_logging()
    config = load_config()
    seed_everything(config["project"]["seed"])
    preprocessing_config = PreprocessingConfig.from_dict(config)
    preprocessor = DataPreprocessor(preprocessing_config)
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
