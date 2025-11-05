"""Training script for the ISL sign recognition model.

This module provides a scaffold for loading preprocessed datasets, constructing a
baseline CNN-LSTM architecture, and handling the training/evaluation loop. The
heavy lifting is left as TODOs to be implemented during subsequent development
phases.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import amp, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import mvit_v2_s, r3d_18
from torchvision.models.video import MViT_V2_S_Weights, R3D_18_Weights

from src.utils import (
    load_config,
    load_model_config,
    seed_everything,
    setup_logging,
    strip_label_prefix,
)
from src.datasets import SignVideoDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LOGGER = logging.getLogger(__name__)


def discover_videos(dataset_root: Path) -> Dict[str, List[Path]]:
    """Traverse dataset directory and collect video paths per class."""

    supported_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    class_to_paths: Dict[str, List[Path]] = {}
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root {dataset_root} does not exist")

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            videos = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() in supported_suffixes
            )
            if not videos:
                LOGGER.warning("No videos found for class %s/%s", category_dir.name, class_dir.name)
                continue
            class_to_paths[class_dir.name] = videos

    if not class_to_paths:
        raise RuntimeError(f"No class folders with videos found under {dataset_root}")
    return class_to_paths


def split_train_val(
    class_to_paths: Dict[str, List[Path]],
    val_split: float,
    max_samples_per_class: int | None,
) -> Tuple[List[Path], List[Path]]:
    """Split dataset into train and validation path lists while preserving class balance."""

    train_paths: List[Path] = []
    val_paths: List[Path] = []

    for _, paths in sorted(class_to_paths.items()):
        label_paths = list(paths)
        if max_samples_per_class is not None:
            label_paths = label_paths[:max_samples_per_class]
        if not label_paths:
            continue

        random.shuffle(label_paths)

        if val_split <= 0 or len(label_paths) < 2:
            train_paths.extend(label_paths)
            continue

        split_idx = int(len(label_paths) * (1 - val_split))
        split_idx = min(max(split_idx, 1), len(label_paths) - 1)
        train_paths.extend(label_paths[:split_idx])
        val_paths.extend(label_paths[split_idx:])

    return train_paths, val_paths


class R3D18SignClassifier(nn.Module):
    """Wrapper around torchvision R3D-18 for sign classification."""

    def __init__(self, model_cfg: Dict[str, any]):
        super().__init__()
        weights_enum = model_cfg.get("pretrained_weights", "DEFAULT")
        weights = None
        if weights_enum and weights_enum not in {"none", "NONE", None}:
            weights = getattr(R3D_18_Weights, weights_enum, R3D_18_Weights.DEFAULT)
        self.backbone = r3d_18(weights=weights)

        in_features = self.backbone.fc.in_features
        dropout = model_cfg.get("classifier", {}).get("dropout", 0.3)
        num_classes = model_cfg.get("classifier", {}).get("num_classes", 30)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        if model_cfg.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            for layer_name in model_cfg.get("unfreeze_layers", []):
                layer = getattr(self.backbone, layer_name, None)
                if layer is None:
                    continue
                for param in layer.parameters():
                    param.requires_grad_(True)
            for param in self.backbone.fc.parameters():
                param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _maybe_freeze_parameters(model: nn.Module, model_cfg: Dict[str, any]) -> None:
    if not model_cfg.get("freeze_backbone", False):
        return

    trainable_keywords = set(model_cfg.get("classifier", {}).get("layer_keywords", []))
    unfreeze_layers = set(model_cfg.get("unfreeze_layers", []))

    for name, param in model.named_parameters():
        should_train = False
        if any(keyword in name for keyword in trainable_keywords):
            should_train = True
        if any(layer_name in name for layer_name in unfreeze_layers):
            should_train = True
        param.requires_grad_(should_train)


def build_sign_recognition_model(model_cfg: Dict[str, any]) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "r3d18")).lower()

    if architecture == "r3d18":
        model = R3D18SignClassifier(model_cfg)
    elif architecture == "mvit_v2_s":
        weights_enum = model_cfg.get("pretrained_weights", "DEFAULT")
        weights = None
        if weights_enum and weights_enum not in {"none", "NONE", None}:
            weights = getattr(MViT_V2_S_Weights, weights_enum, MViT_V2_S_Weights.DEFAULT)
        model = mvit_v2_s(weights=weights)
        if hasattr(model.head, "in_features"):
            embed_dim = model.head.in_features
        else:
            linear_layer = next((module for module in model.head.modules() if isinstance(module, nn.Linear)), None)
            if linear_layer is None:
                raise RuntimeError("Unable to infer classifier input features for MViT-V2 head")
            embed_dim = linear_layer.in_features
        classifier_cfg = model_cfg.get("classifier", {})
        dropout = float(classifier_cfg.get("dropout", 0.0))
        num_classes = int(classifier_cfg.get("num_classes", 30))
        model.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
    else:
        raise ValueError(f"Unsupported sign-recognition architecture: {architecture}")

    _maybe_freeze_parameters(model, model_cfg)
    return model


def collect_classifier_parameters(model: nn.Module, keywords: Iterable[str]) -> List[nn.Parameter]:
    if not keywords:
        return list(model.parameters())

    matched: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in keywords):
            matched.append(param)

    return matched or [param for param in model.parameters() if param.requires_grad]


def remove_excluded_labels(
    class_to_paths: Dict[str, List[Path]], exclude_labels: Iterable[str]
) -> Dict[str, List[Path]]:
    exclude_canon = {label.strip() for label in exclude_labels if label}
    exclude_canon |= {strip_label_prefix(label) for label in exclude_canon}

    if not exclude_canon:
        return class_to_paths

    filtered: Dict[str, List[Path]] = {}
    for label, paths in class_to_paths.items():
        label_clean = strip_label_prefix(label)
        if label in exclude_canon or label_clean in exclude_canon:
            LOGGER.info("Excluding label '%s' (%d samples) from training", label, len(paths))
            continue
        filtered[label] = paths

    if not filtered:
        raise RuntimeError("All labels were excluded; adjust exclude_labels in config.")

    return filtered


def train() -> None:
    """Main training routine scaffold."""

    config = load_config()
    model_config = load_model_config()["sign_recognition"]
    train_config = config["training"]["recognition"]
    seed_everything(config["project"]["seed"])
    setup_logging()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device not detected. The training pipeline requires a CUDA-enabled GPU "
            "per project configuration. Please ensure the NVIDIA drivers and CUDA toolkit "
            "are installed and that PyTorch is built with CUDA support."
        )

    device = torch.device("cuda")
    LOGGER.info("Using device: %s", device)

    dataset_root = Path(config["paths"]["dataset_root"])
    class_to_paths = discover_videos(dataset_root)
    exclude_labels = train_config.get("exclude_labels", [])
    class_to_paths = remove_excluded_labels(class_to_paths, exclude_labels)
    class_to_idx = {label: idx for idx, label in enumerate(sorted(class_to_paths))}
    model_config["classifier"]["num_classes"] = len(class_to_idx)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    val_split = float(train_config.get("val_split", 0.2))
    max_samples = train_config.get("max_samples_per_class")
    max_samples = int(max_samples) if max_samples not in (None, "", "null") else None

    train_paths, val_paths = split_train_val(class_to_paths, val_split, max_samples)

    if not train_paths:
        LOGGER.warning("No training samples discovered. Check dataset path configuration.")
        return

    frame_size = tuple(model_config.get("frame_size", [224, 224]))
    temporal_length = model_config.get("temporal_length", 64)
    normalize_enabled = bool(train_config.get("normalize", False))
    mean = tuple(config["preprocessing"].get("normalize_mean", [0.0, 0.0, 0.0]))
    std = tuple(config["preprocessing"].get("normalize_std", [1.0, 1.0, 1.0]))

    dataset_kwargs = dict(
        class_to_idx=class_to_idx,
        frame_size=frame_size,
        num_frames=temporal_length,
        cache=False,
        normalize=normalize_enabled,
        mean=mean,
        std=std,
        augmentations_config=config["preprocessing"].get("augmentations", {}),
        temporal_jitter=train_config.get("temporal_jitter", False),
        clip_stride=int(config["preprocessing"].get("clip_stride", 1)),
    )

    train_dataset = SignVideoDataset(
        video_paths=train_paths,
        train=True,
        augment=train_config.get("augment", False),
        **dataset_kwargs,
    )
    val_dataset = (
        SignVideoDataset(
            video_paths=val_paths,
            train=False,
            augment=False,
            **dataset_kwargs,
        )
        if val_paths
        else None
    )

    num_workers = int(train_config.get("num_workers", 2))
    batch_size = int(train_config.get("batch_size", 16))

    label_counts = Counter(class_to_idx[path.parent.name] for path in train_paths)
    class_weights = torch.tensor(
        [1.0 / label_counts.get(idx, 1) for idx in range(len(class_to_idx))],
        dtype=torch.float32,
        device=device,
    )

    sampler = None
    if train_config.get("balance_classes", False):
        sample_weights = torch.tensor(
            [1.0 / label_counts.get(label, 1) for _, label in train_dataset.samples], dtype=torch.double
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if "prefetch_factor" in train_config:
        loader_kwargs["prefetch_factor"] = int(train_config["prefetch_factor"])
    if "persistent_workers" in train_config:
        loader_kwargs["persistent_workers"] = bool(train_config["persistent_workers"])

    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
        if val_dataset
        else None
    )

    LOGGER.info(
        "Discovered %d classes | %d training videos | %d validation videos",
        len(class_to_idx),
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
    )

    model = build_sign_recognition_model(model_config).to(device)

    checkpoints_dir = Path(config["paths"]["recognition_checkpoint_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    architecture_name = model_config.get("architecture", "baseline")
    checkpoint_basename = f"{architecture_name}_sign_recognition.pt"
    checkpoint_path = checkpoints_dir / checkpoint_basename
    best_checkpoint_path = None
    backup_checkpoint_path = None
    if checkpoint_path.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        backup_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_backup_{timestamp}{checkpoint_path.suffix}"
        )
        try:
            shutil.copy2(checkpoint_path, backup_checkpoint_path)
            LOGGER.info("Existing checkpoint backed up to %s", backup_checkpoint_path)
        except OSError as exc:
            LOGGER.warning("Failed to back up existing checkpoint: %s", exc)

    resume_from_checkpoint = bool(train_config.get("resume_from_checkpoint", False))
    resume_lr_factor = float(train_config.get("resume_lr_factor", 1.0))
    resume_state_loaded = False
    if resume_from_checkpoint and checkpoint_path.exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            resume_state_loaded = True
            LOGGER.info("Loaded checkpoint weights from %s for fine-tuning", checkpoint_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to load checkpoint from %s: %s", checkpoint_path, exc)
    learning_rate = float(train_config.get("learning_rate", 5e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    backbone_lr_multiplier = float(train_config.get("backbone_lr_multiplier", 1.0))
    label_smoothing = float(train_config.get("label_smoothing", 0.0))

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    LOGGER.info("Total trainable parameters: %.2fM", trainable_params / 1e6)

    classifier_keywords = model_config.get("classifier", {}).get("layer_keywords", [])
    classifier_params = collect_classifier_parameters(model, classifier_keywords)
    classifier_param_ids = {id(param) for param in classifier_params}
    backbone_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in classifier_param_ids
    ]

    param_groups = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": learning_rate * backbone_lr_multiplier}
        )
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": learning_rate})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer setup.")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    LOGGER.info(
        "Initial learning rates -> backbone: %.2e | classifier: %.2e",
        param_groups[0]["lr"] if backbone_params else learning_rate,
        param_groups[-1]["lr"],
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
    )

    scheduler = None
    scheduler_name = str(train_config.get("scheduler", "")).lower()
    num_epochs = int(train_config.get("num_epochs", 1))
    warmup_epochs = int(train_config.get("warmup_epochs", 0))
    warmup_start_factor = float(train_config.get("warmup_start_factor", 0.1))
    if scheduler_name == "cosine":
        warmup_epochs = max(0, min(warmup_epochs, num_epochs))
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            )
            cosine_epochs = max(1, num_epochs - warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            LOGGER.info(
                "Using cosine scheduler with linear warmup (%d epochs, start factor %.2f)",
                warmup_epochs,
                warmup_start_factor,
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs),
            )
            LOGGER.info("Using cosine scheduler without warmup")

    scaler = amp.GradScaler("cuda", enabled=bool(train_config.get("mixed_precision", False)))
    gradient_clip = train_config.get("gradient_clip_norm", None)
    patience = train_config.get("early_stopping_patience", 5)
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if resume_state_loaded and resume_lr_factor not in (1.0,):
        for group in optimizer.param_groups:
            group["lr"] *= resume_lr_factor
        if scheduler is not None:
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [lr * resume_lr_factor for lr in scheduler.base_lrs]
            # SequentialLR stores schedulers attribute with their own base_lrs
            if hasattr(scheduler, "_schedulers"):
                for sub_scheduler in scheduler._schedulers:
                    if hasattr(sub_scheduler, "base_lrs"):
                        sub_scheduler.base_lrs = [
                            lr * resume_lr_factor for lr in sub_scheduler.base_lrs
                        ]
        LOGGER.info(
            "Scaled learning rates by factor %.3f for resumed fine-tuning",
            resume_lr_factor,
        )

    LOGGER.info("Starting training loop")
    grad_accum_steps = max(1, int(train_config.get("gradient_accumulation_steps", 1)))
    epoch_history: List[Dict[str, Any]] = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_correct = 0
        running_top5 = 0
        total_samples = 0

        for step, (frames, labels) in enumerate(train_loader, start=1):
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(frames)
                loss_value = criterion(logits, labels)

            loss = loss_value / grad_accum_steps
            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if gradient_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), gradient_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss_value.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            top5_preds = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
            running_correct += (preds == labels).sum().item()
            running_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / max(total_samples, 1)
        train_acc = running_correct / max(total_samples, 1)
        train_top5 = running_top5 / max(total_samples, 1)

        val_loss = float("nan")
        val_acc = float("nan")
        val_top5 = float("nan")
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_running_correct = 0
            val_top5_matches_total = 0
            val_samples = 0
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames = frames.permute(0, 2, 1, 3, 4).contiguous()
                    frames = frames.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(frames)
                    loss = criterion(logits, labels)
                    val_running_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(logits, dim=1)
                    top5_preds = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
                    val_running_correct += (preds == labels).sum().item()
                    val_top5_matches = (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                    val_top5_matches_total += val_top5_matches
                    val_samples += labels.size(0)

            val_loss = val_running_loss / max(val_samples, 1)
            val_acc = val_running_correct / max(val_samples, 1)
            val_top5 = val_top5_matches_total / max(val_samples, 1)
            lr_values = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
            LOGGER.info(
                "Epoch %d | LRs [%s] | Train loss: %.4f acc: %.4f top5: %.4f | Val loss: %.4f acc: %.4f top5: %.4f",
                epoch + 1,
                lr_values,
                train_loss,
                train_acc,
                train_top5,
                val_loss,
                val_acc,
                val_top5,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
                best_checkpoint_path = checkpoint_path
                LOGGER.info("Saved improved checkpoint to %s (val acc %.4f)", checkpoint_path, val_acc)
            else:
                epochs_without_improvement += 1
        else:
            lr_values = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
            LOGGER.info(
                "Epoch %d | LRs [%s] | Train loss: %.4f acc: %.4f",
                epoch + 1,
                lr_values,
                train_loss,
                train_acc,
            )
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_path = checkpoint_path

        epoch_record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rates": lr_values,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_top5": float(train_top5),
        }
        if val_loader:
            epoch_record.update(
                {
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_top5": float(val_top5),
                }
            )
        else:
            epoch_record.update({"val_loss": None, "val_acc": None, "val_top5": None})
        epoch_history.append(epoch_record)

        if scheduler:
            scheduler.step()

        if val_loader and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping triggered after %d epochs without improvement", patience)
            break

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
        best_checkpoint_path = checkpoint_path
        LOGGER.info("Checkpoint saved to %s", checkpoint_path)

    if best_checkpoint_path is None:
        best_checkpoint_path = checkpoint_path

    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    def evaluate_model(eval_loader):
        y_true: List[int] = []
        y_pred: List[int] = []
        top5_hits = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for frames, labels in eval_loader:
                frames = frames.permute(0, 2, 1, 3, 4).contiguous()
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(frames)
                preds = torch.argmax(logits, dim=1)
                top5_preds = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
                top5_hits += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        top5_acc = top5_hits / max(total, 1)
        return y_true, y_pred, top5_acc

    if val_loader:
        split_name = "validation"
        eval_loader = val_loader
    else:
        split_name = "training"
        eval_loader = train_loader

    y_true, y_pred, top5_final = evaluate_model(eval_loader)
    final_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    LOGGER.info("Final %s accuracy: %.4f | Top-5: %.4f", split_name, final_acc, top5_final)
    LOGGER.info("Classification report:%s%s", os.linesep, report)
    LOGGER.info("Confusion matrix:%s%s", os.linesep, cm)
    print(f"Final {split_name} accuracy: {final_acc:.4f} | Top-5: {top5_final:.4f}")

    results = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "architecture": model_config.get("architecture", "unknown"),
        "split": split_name,
        "accuracy": float(final_acc),
        "top5": float(top5_final),
        "checkpoint_path": str(best_checkpoint_path),
        "backed_up_checkpoint": str(backup_checkpoint_path) if backup_checkpoint_path else None,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "train_config": train_config,
        "model_config": model_config,
    }

    if epoch_history:
        def _best_score(record: Dict[str, Any]) -> float:
            val_acc = record.get("val_acc")
            if val_acc is not None:
                return float(val_acc)
            return float(record.get("train_acc", 0.0))

        best_epoch = max(epoch_history, key=_best_score, default=None)
    else:
        best_epoch = None

    results["epoch_metrics"] = epoch_history
    if best_epoch is not None:
        results["best_epoch"] = best_epoch

    logs_base = Path(config["paths"].get("logs_dir", "logs"))
    archive_dir = logs_base / "sign_recognition_runs"
    archived_run_path: Path | None = None
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        run_fname = f"{results['architecture']}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        archived_run_path = archive_dir / run_fname
        archived_run_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("Saved training summary to %s", archived_run_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to persist training summary: %s", exc)

    analysis_dir = logs_base / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "latest_training_summary.json"
    summary_payload = {
        "epochs": epoch_history,
        "best_epoch": best_epoch,
        "latest_archive": str(archived_run_path) if archived_run_path else None,
    }
    try:
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote latest training analytics summary to %s", summary_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to write training analytics summary: %s", exc)

    return results


if __name__ == "__main__":
    train()
