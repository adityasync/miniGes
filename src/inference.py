"""Inference utilities for the sinGes-mini project.

This module exposes helpers to load the trained sign-recognition backbone and
run predictions on uploaded video clips. It also provides a thin wrapper to
convert the predicted words into natural language sentences using a fine-tuned
transformer language model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from src.train_recognition import build_sign_recognition_model, discover_videos
from src.utils import (
    discover_class_labels,
    load_config,
    load_model_config,
    resolve_path,
    strip_label_prefix,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Container for prediction metadata."""

    label: str
    score: float

    @property
    def display_label(self) -> str:
        return strip_label_prefix(self.label)


def _load_video_frames(
    video_path: Path,
    frame_size: tuple[int, int],
    num_frames: int,
    normalize: bool,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Load frames from a video file and return a tensor shaped (T, C, H, W)."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frames: List[np.ndarray] = []
    success, frame = capture.read()
    while success:
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = capture.read()
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    frame_count = len(frames)
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        sampled = [frames[idx] for idx in indices]
    else:
        sampled = frames.copy()
        while len(sampled) < num_frames:
            sampled.append(sampled[-1])

    processed_frames: List[np.ndarray] = []
    for frame in sampled:
        frame = frame.astype(np.float32) / 255.0
        if normalize:
            frame = (frame - mean) / std
        processed_frames.append(frame)

    frames_np = np.stack(processed_frames)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
    return frames_tensor


class SignRecognizer:
    """Load a trained R3D-18 backbone and execute sign classification."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "config/config.yaml",
        model_config_path: str | Path = "config/model_config.yaml",
        device: Optional[str] = None,
    ) -> None:
        config = load_config(str(config_path))
        model_config = load_model_config(str(model_config_path))["sign_recognition"].copy()
        dataset_root = resolve_path(config["paths"]["dataset_root"])

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Recognition checkpoint not found at {checkpoint_path}. "
                "Train the model via src/train_recognition.py first."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        LOGGER.info("Loading sign recognizer on device: %s", self.device)

        self.frame_size = tuple(model_config.get("frame_size", (112, 112)))
        temporal_length = int(model_config.get("temporal_length", 32))
        self.num_frames = temporal_length
        train_cfg = config["training"]["recognition"]
        self.normalize = bool(train_cfg.get("normalize", False))
        preprocess_cfg = config["preprocessing"]
        self.mean = np.array(preprocess_cfg.get("normalize_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.array(preprocess_cfg.get("normalize_std", [1.0, 1.0, 1.0]), dtype=np.float32)

        LOGGER.debug(
            "Frame size: %s | num_frames: %d | normalize: %s",
            self.frame_size,
            self.num_frames,
            self.normalize,
        )

        class_to_paths = discover_videos(dataset_root)
        self.class_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(sorted(class_to_paths))}
        self.idx_to_class: Dict[int, str] = {idx: label for label, idx in self.class_to_idx.items()}
        num_classes = len(self.idx_to_class)
        LOGGER.info("Discovered %d classes for inference", num_classes)

        classifier_cfg = model_config.setdefault("classifier", {})
        classifier_cfg["num_classes"] = num_classes

        model = build_sign_recognition_model(model_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, video_path: str | Path, top_k: int = 5) -> List[Prediction]:
        video_path = resolve_path(video_path)
        frames_tensor = _load_video_frames(
            video_path,
            self.frame_size,
            self.num_frames,
            self.normalize,
            self.mean,
            self.std,
        )
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).contiguous()
        frames_tensor = frames_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(frames_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_k = min(top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=top_k)
        predictions = [
            Prediction(label=self.idx_to_class[idx.item()], score=score.item())
            for score, idx in zip(scores, indices)
        ]
        return predictions

    @property
    def class_names(self) -> List[str]:
        return [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]


class SentenceGenerator:
    """Wrapper around a fine-tuned causal language model for sentence generation."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        pretrained_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = resolve_path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Sentence generator checkpoint not found at {checkpoint_dir}. Run "
                "src/transformer_finetune.py to create it."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        LOGGER.info("Loading sentence generator on device: %s", self.device)

        model_source = str(checkpoint_dir if checkpoint_dir.exists() else pretrained_model)
        if model_source is None:
            raise ValueError("Either checkpoint_dir or pretrained_model must be provided.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        seed_words: List[str],
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        if not seed_words:
            raise ValueError("Provide at least one recognized word to generate a sentence.")

        prompt = " ".join(seed_words)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(generation[0], skip_special_tokens=True)
        return text.strip()


def load_display_labels(dataset_root: str | Path) -> List[str]:
    """Utility for UI layers to fetch cleaned label names."""

    labels = discover_class_labels(dataset_root)
    return [strip_label_prefix(label) for label in labels]
