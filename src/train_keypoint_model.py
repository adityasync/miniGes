#!/usr/bin/env python3
"""Train a BiLSTM classifier on MediaPipe keypoint sequences."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch import amp, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, resolve_path, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class KeypointTrainingConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: Optional[float]
    mixed_precision: bool
    val_split: float
    max_samples_per_class: Optional[int]
    sampler_type: str
    compute_class_weights: bool
    scheduler: Optional[str]
    warmup_epochs: int
    export_onnx: bool
    temporal_length: int
    gradient_accumulation_steps: int
    label_smoothing: float

    @classmethod
    def from_config(cls, cfg: Dict[str, Dict[str, object]]) -> "KeypointTrainingConfig":
        kc = cfg["keypoint_model"]
        def _opt_float(value):
            return None if value in (None, "null") else float(value)
        def _opt_int(value):
            return None if value in (None, "null") else int(value)
        return cls(
            input_size=int(kc.get("input_size", 1629)),
            hidden_size=int(kc.get("hidden_size", 192)),
            num_layers=int(kc.get("num_layers", 1)),
            dropout=float(kc.get("dropout", 0.3)),
            batch_size=int(kc.get("batch_size", 8)),
            learning_rate=float(kc.get("learning_rate", 5e-4)),
            weight_decay=float(kc.get("weight_decay", 1e-4)),
            num_epochs=int(kc.get("num_epochs", 40)),
            early_stopping_patience=int(kc.get("early_stopping_patience", 6)),
            gradient_clip_norm=_opt_float(kc.get("gradient_clip_norm", 5.0)),
            mixed_precision=bool(kc.get("mixed_precision", True)),
            val_split=float(kc.get("val_split", 0.2)),
            max_samples_per_class=_opt_int(kc.get("max_samples_per_class")),
            sampler_type=str(kc.get("sampler_type", "")),
            compute_class_weights=bool(kc.get("compute_class_weights", False)),
            scheduler=(str(kc.get("scheduler", "")) or None),
            warmup_epochs=int(kc.get("warmup_epochs", 0)),
            export_onnx=bool(kc.get("export_onnx", False)),
            temporal_length=int(kc.get("temporal_length", 32)),
            gradient_accumulation_steps=int(kc.get("gradient_accumulation_steps", 1)),
            label_smoothing=float(kc.get("label_smoothing", 0.0)),
        )


class KeypointSequenceDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], temporal_length: int, input_size: int) -> None:
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.temporal_length = temporal_length
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, seq: np.ndarray) -> np.ndarray:
        seq = seq.astype(np.float32)
        if seq.ndim == 3:
            seq = seq.reshape(seq.shape[0], -1)
        if seq.ndim != 2:
            raise ValueError(f"Unexpected sequence shape {seq.shape}")
        length = seq.shape[0]
        if length == self.temporal_length:
            return seq
        if length > self.temporal_length:
            idx = np.linspace(0, length - 1, self.temporal_length, dtype=int)
            return seq[idx]
        pad = np.repeat(seq[-1:], self.temporal_length - length, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with np.load(path, allow_pickle=False) as data:
            if "coordinates" in data:
                coords = data["coordinates"]
            elif "landmarks" in data:
                coords = data["landmarks"]
            else:
                raise KeyError(f"File {path} missing coordinates array")
        seq = self._resize(coords)
        return torch.from_numpy(seq), label


class KeypointBiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_landmark_files(dataset_root: Path, mediapipe_dir: Path) -> Dict[str, List[Path]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} not found")
    if not mediapipe_dir.exists():
        raise FileNotFoundError(
            f"MediaPipe directory {mediapipe_dir} missing. Run extract_mediapipe_landmarks.py first."
        )
    per_class: Dict[str, List[Path]] = {}
    for video in sorted(dataset_root.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        label = video.parent.name
        rel = video.relative_to(dataset_root)
        landmark_path = mediapipe_dir / rel.parent / f"{rel.stem}_landmarks.npz"
        if landmark_path.exists():
            per_class.setdefault(label, []).append(landmark_path)
    if not per_class:
        raise RuntimeError("No landmark archives discovered; verify extraction stage.")
    return per_class


def split_samples(
    per_class_files: Dict[str, List[Path]],
    class_to_idx: Dict[str, int],
    val_split: float,
    max_samples_per_class: Optional[int],
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []

    for label, files in per_class_files.items():
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)
        if max_samples_per_class is not None:
            files_sorted = files_sorted[:max_samples_per_class]
        if not files_sorted:
            continue

        label_idx = class_to_idx[label]

        if val_split <= 0 or len(files_sorted) < 2:
            train.extend((path, label_idx) for path in files_sorted)
            continue

        split_idx = int(round(len(files_sorted) * (1 - val_split)))
        split_idx = min(max(split_idx, 1), len(files_sorted) - 1)

        train.extend((path, label_idx) for path in files_sorted[:split_idx])
        val.extend((path, label_idx) for path in files_sorted[split_idx:])

    if not train:
        raise RuntimeError("No training samples assembled; adjust val_split or gather more data.")

    return train, val


def build_dataloaders(
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    cfg: KeypointTrainingConfig,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, Dataset]:
    train_dataset = KeypointSequenceDataset(train_samples, cfg.temporal_length, cfg.input_size)
    val_dataset = (
        KeypointSequenceDataset(val_samples, cfg.temporal_length, cfg.input_size) if val_samples else None
    )
    targets = np.array(train_dataset.targets, dtype=np.int64)
    num_classes = int(targets.max()) + 1
    class_weights = np.ones(num_classes, dtype=np.float32)
    if cfg.compute_class_weights:
        classes = np.arange(num_classes)
        class_weights = compute_class_weight("balanced", classes=classes, y=targets).astype(np.float32)
    sampler = None
    if cfg.sampler_type.lower() == "weighted_random_sampler":
        counts = np.bincount(targets, minlength=num_classes)
        weights = (1.0 / np.clip(counts, 1, None))[targets]
        sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), len(weights), replacement=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader, class_weights, train_dataset


def export_model(model: nn.Module, export_path: Path, cfg: KeypointTrainingConfig) -> None:
    dummy = torch.randn(1, cfg.temporal_length, cfg.input_size, dtype=torch.float32)
    ensure_dir(export_path.parent)
    torch.onnx.export(
        model.cpu(),
        dummy,
        export_path,
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    LOGGER.info("Exported ONNX model to %s", export_path)


def train_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model: nn.Module,
    cfg: KeypointTrainingConfig,
    class_weights: np.ndarray,
    checkpoint_dir: Path,
    device: torch.device,
) -> Tuple[Path, Dict[str, object]]:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler and cfg.scheduler.lower() == "cosine":
        total_epochs = max(1, cfg.num_epochs)
        warmup = max(0, min(cfg.warmup_epochs, total_epochs - 1))
        if warmup > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-2,
                total_iters=warmup,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs - warmup),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
            )
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device) if cfg.compute_class_weights else None,
        label_smoothing=cfg.label_smoothing,
    )
    use_amp = device.type == "cuda" and cfg.mixed_precision
    scaler = amp.GradScaler(enabled=use_amp)
    best_acc = 0.0
    epochs_no_improve = 0
    best_ckpt = checkpoint_dir / "keypoint_model_best.pt"
    grad_accum = max(1, cfg.gradient_accumulation_steps)
    metrics_history: List[Dict[str, float]] = []

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (seq, labels) in enumerate(train_loader, 1):
            seq = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            autocast_ctx = amp.autocast(device_type=device.type) if use_amp else nullcontext()
            with autocast_ctx:
                logits = model(seq)
                loss = criterion(logits, labels)
            loss_to_backprop = loss / grad_accum
            scaler.scale(loss_to_backprop).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if cfg.gradient_clip_norm:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc = float("nan")
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            v_loss = 0.0
            with torch.no_grad():
                for seq, labels in val_loader:
                    seq = seq.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    autocast_eval = amp.autocast(device_type=device.type) if use_amp else nullcontext()
                    with autocast_eval:
                        logits = model(seq)
                        loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
                    v_loss += loss.item() * labels.size(0)
            val_acc = v_correct / max(v_total, 1)
            val_loss = v_loss / max(v_total, 1)
            LOGGER.info(
                "Epoch %d | Train loss %.4f acc %.4f | Val loss %.4f acc %.4f",
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_ckpt)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            LOGGER.info("Epoch %d | Train loss %.4f acc %.4f", epoch + 1, train_loss, train_acc)
            torch.save(model.state_dict(), best_ckpt)
            best_acc = train_acc
        metrics_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if cfg.early_stopping_patience and epochs_no_improve >= cfg.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
            break
        if scheduler is not None:
            scheduler.step()
    return best_ckpt, {"best_acc": best_acc, "history": metrics_history}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, class_names: List[str]) -> Dict[str, object]:
    model.eval()
    preds_all: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device, non_blocking=True)
            logits = model(seq)
            preds_all.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
    acc = accuracy_score(labels_all, preds_all)
    report = classification_report(labels_all, preds_all, target_names=class_names, zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train keypoint BiLSTM model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    keypoint_cfg = KeypointTrainingConfig.from_config(config)
    logs_dir = args.logs_dir or config["paths"].get("logs_dir", "logs")
    setup_logging(logs_dir)
    seed_everything(args.seed or config["project"].get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Training keypoint model on %s", device)

    dataset_root = resolve_path(config["paths"]["dataset_root"])
    mediapipe_dir = resolve_path(config["paths"]["mediapipe_output_dir"])
    per_class_files = discover_landmark_files(dataset_root, mediapipe_dir)
    class_names = sorted(per_class_files)
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    train_samples, val_samples = split_samples(
        per_class_files,
        class_to_idx,
        val_split=keypoint_cfg.val_split,
        max_samples_per_class=keypoint_cfg.max_samples_per_class,
        seed=config["project"].get("seed", 42),
    )
    train_loader, val_loader, class_weights, train_dataset = build_dataloaders(
        train_samples,
        val_samples,
        keypoint_cfg,
        num_workers=args.num_workers,
        seed=config["project"].get("seed", 42),
    )
    model = KeypointBiLSTM(
        input_size=keypoint_cfg.input_size,
        hidden_size=keypoint_cfg.hidden_size,
        num_layers=keypoint_cfg.num_layers,
        dropout=keypoint_cfg.dropout,
        num_classes=len(class_names),
    ).to(device)

    checkpoint_dir = resolve_path(args.checkpoint_dir or config["paths"].get("keypoint_checkpoint_dir", "models/keypoint"))
    best_path, history = train_model(
        train_loader,
        val_loader,
        model,
        keypoint_cfg,
        class_weights,
        checkpoint_dir,
        device,
    )

    LOGGER.info("Best checkpoint saved to %s", best_path)

    eval_results = None
    if val_loader is not None and not args.skip_eval:
        model.load_state_dict(torch.load(best_path, map_location=device))
        eval_results = evaluate_model(model, val_loader, device, class_names)
        LOGGER.info("Validation accuracy: %.4f", eval_results["accuracy"])
        LOGGER.info("Classification report:%s%s", os.linesep, eval_results["report"])

    results_path = checkpoint_dir / "training_summary.json"
    payload = {
        "best_checkpoint": str(best_path),
        "class_names": class_names,
        "history": history,
        "eval": eval_results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved training summary to %s", results_path)

    logs_dir = resolve_path(config["paths"].get("logs_dir", "logs"))
    analysis_dir = logs_dir / "analysis"
    ensure_dir(analysis_dir)
    latest_summary = analysis_dir / "latest_keypoint_training_summary.json"
    latest_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Updated analytics summary at %s", latest_summary)

    if keypoint_cfg.export_onnx:
        onnx_path = checkpoint_dir / "keypoint_model.onnx"
        export_model(model.to("cpu"), onnx_path, keypoint_cfg)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
