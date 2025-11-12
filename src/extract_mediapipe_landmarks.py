#!/usr/bin/env python3
"""MediaPipe holistic keypoint extraction script.

This utility traverses the configured dataset directory, runs MediaPipe Holistic on
video clips, normalises the resulting keypoints, and persists them as compressed
NumPy archives. It is optimised for CPU execution on low-VRAM systems and offers
basic CLI overrides for paths and processing subsets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency at runtime
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, resolve_path, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    dataset_root: Path
    output_dir: Path
    frame_rate: int
    max_frames_per_clip: int
    holistic_enabled: bool
    model_complexity: int
    detection_confidence: float
    tracking_confidence: float
    max_hands: int
    static_mode: bool
    normalise_reference: str
    shoulder_width_epsilon: float
    center_hips: bool

    @classmethod
    def from_config(cls, cfg: dict) -> "ExtractionConfig":
        preprocessing = cfg["preprocessing"]
        mediapipe_cfg = preprocessing.get("mediapipe", {})
        norm_cfg = preprocessing.get("keypoint_normalization", {})
        paths = cfg["paths"]
        return cls(
            dataset_root=resolve_path(paths["dataset_root"]),
            output_dir=resolve_path(paths["mediapipe_output_dir"]),
            frame_rate=int(preprocessing.get("frame_rate", 25)),
            max_frames_per_clip=int(preprocessing.get("max_frames_per_clip", 32)),
            holistic_enabled=bool(mediapipe_cfg.get("holistic_enabled", True)),
            model_complexity=int(mediapipe_cfg.get("model_complexity", 0)),
            detection_confidence=float(mediapipe_cfg.get("detection_confidence", 0.6)),
            tracking_confidence=float(mediapipe_cfg.get("tracking_confidence", 0.5)),
            max_hands=int(mediapipe_cfg.get("max_hands", 2)),
            static_mode=bool(mediapipe_cfg.get("static_mode", False)),
            normalise_reference=str(norm_cfg.get("reference", "torso")),
            shoulder_width_epsilon=float(norm_cfg.get("shoulder_width_epsilon", 1e-6)),
            center_hips=bool(norm_cfg.get("center_hips", True)),
        )


def _discover_videos(dataset_root: Path, pattern: Optional[str] = None) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    patterns = [pattern] if pattern else ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.MKV"]
    videos: List[Path] = []
    for pat in patterns:
        videos.extend(sorted(dataset_root.rglob(pat)))
    unique = sorted({path.resolve() for path in videos if path.is_file()})
    LOGGER.info("Discovered %d video files under %s", len(unique), dataset_root)
    return unique


def _compute_stride(capture: cv2.VideoCapture, desired_fps: int) -> int:
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0:
        return 1
    stride = max(1, int(round(src_fps / max(1, desired_fps))))
    return stride


class HolisticExtractor:
    POSE_COUNT = 33
    LEFT_HAND_COUNT = 21
    RIGHT_HAND_COUNT = 21
    FACE_COUNT = 468
    TOTAL_LANDMARKS = POSE_COUNT + LEFT_HAND_COUNT + RIGHT_HAND_COUNT + FACE_COUNT

    def __init__(self, config: ExtractionConfig) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is required for keypoint extraction. Install it via `pip install mediapipe`."
            )
        if not config.holistic_enabled:
            raise RuntimeError("Holistic extraction is disabled in config.preprocessing.mediapipe")

        self.config = config
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=config.static_mode,
            model_complexity=config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

        self.pose_enum = mp.solutions.holistic.PoseLandmark  # type: ignore[attr-defined]
        self._pose_indices = {
            name: getattr(self.pose_enum, name).value for name in self.pose_enum.__members__.keys()
        }

    def close(self) -> None:
        self._holistic.close()

    @staticmethod
    def _zero_landmarks() -> Tuple[np.ndarray, np.ndarray]:
        coords = np.zeros((HolisticExtractor.TOTAL_LANDMARKS, 3), dtype=np.float32)
        visibility = np.zeros((HolisticExtractor.TOTAL_LANDMARKS,), dtype=np.float32)
        return coords, visibility

    def _pack_landmarks(self, results: mp.solutions.holistic.HolisticResults) -> Tuple[np.ndarray, np.ndarray]:
        coords, visibility = self._zero_landmarks()
        offset = 0

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                coords[offset + idx] = (landmark.x, landmark.y, landmark.z)
                visibility[offset + idx] = getattr(landmark, "visibility", 1.0)
        offset += self.POSE_COUNT

        def _store(part_landmarks: Optional[Iterable]) -> None:
            nonlocal offset
            if part_landmarks:
                for idx, landmark in enumerate(part_landmarks.landmark):
                    coords[offset + idx] = (landmark.x, landmark.y, landmark.z)
                    visibility[offset + idx] = 1.0
            offset += 21

        _store(results.left_hand_landmarks)
        _store(results.right_hand_landmarks)

        if results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
            limit = min(len(face_landmarks), self.FACE_COUNT)
            for idx in range(limit):
                landmark = face_landmarks[idx]
                coords[offset + idx] = (landmark.x, landmark.y, landmark.z)
                visibility[offset + idx] = 1.0
        return coords, visibility

    def _normalise(self, coords: np.ndarray) -> np.ndarray:
        coords = coords.copy()
        pose = coords[: self.POSE_COUNT]

        if self.config.center_hips:
            left_hip_idx = self._pose_indices.get("LEFT_HIP")
            right_hip_idx = self._pose_indices.get("RIGHT_HIP")
            if left_hip_idx is not None and right_hip_idx is not None:
                mid_hips = (pose[left_hip_idx] + pose[right_hip_idx]) / 2.0
                coords -= mid_hips

        if self.config.normalise_reference.lower() == "torso":
            left_shoulder_idx = self._pose_indices.get("LEFT_SHOULDER")
            right_shoulder_idx = self._pose_indices.get("RIGHT_SHOULDER")
            if left_shoulder_idx is not None and right_shoulder_idx is not None:
                shoulder_width = np.linalg.norm(
                    pose[left_shoulder_idx] - pose[right_shoulder_idx]
                )
                scale = max(shoulder_width, self.config.shoulder_width_epsilon)
                coords /= scale

        return coords

    def process_video(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")

        stride = _compute_stride(capture, self.config.frame_rate)
        frame_idx = 0
        coords_sequence: List[np.ndarray] = []
        visibility_sequence: List[np.ndarray] = []

        success, frame = capture.read()
        while success:
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._holistic.process(rgb)
                coords, visibility = self._pack_landmarks(results)
                coords = self._normalise(coords)
                coords_sequence.append(coords)
                visibility_sequence.append(visibility)
                if len(coords_sequence) >= self.config.max_frames_per_clip:
                    break
            frame_idx += 1
            success, frame = capture.read()

        capture.release()

        if not coords_sequence:
            coords, visibility = self._zero_landmarks()
            coords_sequence.append(coords)
            visibility_sequence.append(visibility)

        stacked_coords = np.stack(coords_sequence).astype(np.float32)
        stacked_visibility = np.stack(visibility_sequence).astype(np.float32)
        return stacked_coords, stacked_visibility


def _save_landmarks(base_dir: Path, dataset_root: Path, video_path: Path, coords: np.ndarray, visibility: np.ndarray) -> Path:
    relative = video_path.relative_to(dataset_root)
    output_path = base_dir / relative.parent / f"{relative.stem}_landmarks.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, coordinates=coords, visibility=visibility)
    return output_path


def _write_manifest(output_dir: Path, manifest: List[dict]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic keypoints")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store extracted landmarks",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional glob to filter videos (e.g. '*HELLO*.mp4')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos (useful for smoke tests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (defaults to project seed)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction if destination file already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    extraction_cfg = ExtractionConfig.from_config(cfg)

    if args.dataset_root:
        extraction_cfg.dataset_root = resolve_path(args.dataset_root)
    if args.output_dir:
        extraction_cfg.output_dir = resolve_path(args.output_dir)

    extraction_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(cfg["paths"].get("logs_dir", "logs")))
    seed_everything(args.seed or cfg.get("project", {}).get("seed", 42))

    videos = _discover_videos(extraction_cfg.dataset_root, args.pattern)
    if args.limit is not None:
        videos = videos[: args.limit]

    extractor = HolisticExtractor(extraction_cfg)
    manifest: List[dict] = []
    processed = 0

    try:
        for video_path in videos:
            dest_path = extraction_cfg.output_dir / video_path.relative_to(extraction_cfg.dataset_root).parent / (
                f"{video_path.stem}_landmarks.npz"
            )
            if args.skip_existing and dest_path.exists():
                LOGGER.info("Skipping %s (already exists)", dest_path)
                manifest.append(
                    {
                        "video": str(video_path),
                        "landmarks": str(dest_path),
                        "skipped": True,
                    }
                )
                continue

            try:
                coords, visibility = extractor.process_video(video_path)
                output_path = _save_landmarks(
                    extraction_cfg.output_dir,
                    extraction_cfg.dataset_root,
                    video_path,
                    coords,
                    visibility,
                )
                manifest.append(
                    {
                        "video": str(video_path),
                        "landmarks": str(output_path),
                        "frames": int(coords.shape[0]),
                    }
                )
                processed += 1
                LOGGER.info("[%d/%d] Saved landmarks to %s", processed, len(videos), output_path)
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.exception("Failed to process %s: %s", video_path, exc)
                manifest.append(
                    {
                        "video": str(video_path),
                        "error": str(exc),
                    }
                )
    finally:
        extractor.close()

    _write_manifest(extraction_cfg.output_dir, manifest)
    LOGGER.info("Extraction complete: %d/%d videos processed", processed, len(videos))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
