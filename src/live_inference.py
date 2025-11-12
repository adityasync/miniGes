#!/usr/bin/env python3
"""Live inference loop with sliding-window smoothing and ONNX Runtime support."""

from __future__ import annotations

import argparse
import collections
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import SignRecognizer
from src.utils import discover_class_labels, load_config, resolve_path, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    window_size: int
    stride: int
    min_confidence: float
    device: str
    smoothing_factor: float
    enable_keypoint_pipeline: bool
    realtime_batch_size: int
    max_queue: int

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "InferenceConfig":
        inference_cfg = cfg.get("inference", {})
        return cls(
            window_size=int(inference_cfg.get("window_size", 24)),
            stride=int(inference_cfg.get("stride", 6)),
            min_confidence=float(inference_cfg.get("min_confidence", 0.6)),
            device=str(inference_cfg.get("device", "cuda")),
            smoothing_factor=float(inference_cfg.get("smoothing_factor", 0.6)),
            enable_keypoint_pipeline=bool(inference_cfg.get("enable_keypoint_pipeline", True)),
            realtime_batch_size=int(inference_cfg.get("realtime_batch_size", 1)),
            max_queue=int(inference_cfg.get("max_queue", 4)),
        )


@dataclass
class SmoothingState:
    smoothing_factor: float
    current: Optional[np.ndarray] = None

    def update(self, logits: np.ndarray) -> np.ndarray:
        if self.current is None:
            self.current = logits
        else:
            self.current = (
                self.smoothing_factor * logits + (1.0 - self.smoothing_factor) * self.current
            )
        return self.current


class SlidingWindowBuffer:
    def __init__(self, window_size: int, stride: int) -> None:
        self.window_size = window_size
        self.stride = stride
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self.total_frames_seen = 0

    def add(self, frame: np.ndarray) -> None:
        self.buffer.append(frame)
        self.total_frames_seen += 1

    def ready(self) -> bool:
        return len(self.buffer) == self.window_size and (self.total_frames_seen % self.stride == 0)

    def get_clip(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("Sliding window not ready")
        clip = np.stack(self.buffer, axis=0)
        return clip


class OnnxRunner:
    def __init__(self, model_path: Path, device: str) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX inference")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        clip = clip.astype(np.float32)
        clip = np.expand_dims(clip, axis=0)
        inputs = {self.input_name: clip}
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0]


class TorchRunner:
    def __init__(self, recognizer: SignRecognizer, device: str) -> None:
        self.recognizer = recognizer
        self.device = device

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        frames_tensor = torch.from_numpy(clip)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).contiguous()
        frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).contiguous()
        frames_tensor = frames_tensor.to(self.recognizer.device)
        with torch.no_grad():
            logits = self.recognizer.model(frames_tensor)
            return logits.cpu().numpy()


class FrameGrabber(threading.Thread):
    def __init__(self, source: int | str, output_queue: "queue.Queue[np.ndarray]", stop_event: threading.Event, frame_size: Tuple[int, int]) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.frame_size = frame_size

    def run(self) -> None:  # pragma: no cover - relies on webcam
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            LOGGER.error("Unable to open video source %s", self.source)
            self.stop_event.set()
            return
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    LOGGER.warning("Frame grab failed; stopping")
                    break
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    self.output_queue.put(frame, timeout=0.1)
                except queue.Full:
                    LOGGER.debug("Frame queue full; dropping frame")
        finally:
            cap.release()
            self.stop_event.set()


class LiveInferenceLoop:
    def __init__(
        self,
        runner,
        class_names: List[str],
        config: InferenceConfig,
        frame_size: Tuple[int, int],
    ) -> None:
        self.runner = runner
        self.class_names = class_names
        self.config = config
        self.frame_size = frame_size
        self.buffer = SlidingWindowBuffer(config.window_size, config.stride)
        self.smoother = SmoothingState(config.smoothing_factor)
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.max_queue)
        self.stop_event = threading.Event()

    def start_capture(self, source: int | str) -> FrameGrabber:
        grabber = FrameGrabber(source, self.queue, self.stop_event, self.frame_size)
        grabber.start()
        return grabber

    def stop(self) -> None:
        self.stop_event.set()

    def run_loop(self, display: bool = True) -> None:  # pragma: no cover - runtime loop
        skip = False
        while not self.stop_event.is_set():
            try:
                frame = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if skip:
                skip = False
                continue

            self.buffer.add(frame)
            if self.buffer.ready():
                clip = self.buffer.get_clip()
                logits = self.runner(clip)
                logits = self.smoother.update(logits)
                probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
                top_idx = int(np.argmax(probs))
                top_label = self.class_names[top_idx]
                top_score = float(probs[top_idx])
                if top_score >= self.config.min_confidence:
                    LOGGER.info("Prediction: %s (%.2f)", top_label, top_score)
                    if display:
                        self._render(frame, top_label, top_score)
                else:
                    LOGGER.debug("Below confidence threshold: %.2f", top_score)

            if display:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.stop()
                    break

        if display:
            cv2.destroyAllWindows()

    def _render(self, frame: np.ndarray, label: str, score: float) -> None:
        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = f"{label}: {score:.2f}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("sinGes Live", annotated)


def load_onnx_runner(model_path: Path, device: str):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return OnnxRunner(model_path, device)


def load_torch_runner(cfg: Dict[str, object], device: str):
    recognition_ckpt = cfg["paths"].get("recognition_checkpoint_dir", "models/checkpoints/sign_recognition")
    ckpt_dir = resolve_path(recognition_ckpt)
    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {ckpt_dir}")
    latest_ckpt = checkpoints[-1]
    recognizer = SignRecognizer(
        checkpoint_path=latest_ckpt,
        config_path="config/config.yaml",
        model_config_path="config/model_config.yaml",
        device=device,
    )
    return TorchRunner(recognizer, device), recognizer.class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live sliding-window sign inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model for inference")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source", default=0, help="Camera index or video path")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window")
    parser.add_argument("--logs-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inference_cfg = InferenceConfig.from_config(cfg)
    if args.device:
        inference_cfg.device = args.device

    logs_dir = args.logs_dir or cfg["paths"].get("logs_dir", "logs")
    setup_logging(logs_dir)
    seed_everything(args.seed or cfg["project"].get("seed", 42))

    frame_size = tuple(cfg["preprocessing"].get("frame_resize", [112, 112]))

    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    class_names = discover_class_labels(dataset_root)

    if args.onnx:
        runner = load_onnx_runner(resolve_path(args.onnx), inference_cfg.device)
    else:
        runner, class_names = load_torch_runner(cfg, inference_cfg.device)

    if isinstance(args.source, str) and Path(args.source).exists():
        source = str(resolve_path(args.source))
    else:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    loop = LiveInferenceLoop(runner, class_names, inference_cfg, frame_size)
    grabber = loop.start_capture(source)

    try:
        loop.run_loop(display=not args.no_display)
    finally:
        loop.stop()
        grabber.join(timeout=2.0)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
