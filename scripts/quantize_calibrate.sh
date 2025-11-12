#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"

CONFIG_PATH=${1:-config/config.yaml}
MODEL_CONFIG_PATH=${2:-config/model_config.yaml}
CALIBRATION_LIMIT=${3:-0}

PYTHON_SCRIPT=$(cat <<'PY'
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import SignVideoDataset
from src.train_recognition import (
    build_sign_recognition_model,
    collect_calibration_samples,
    convert_onnx_to_fp16,
    discover_videos,
    quantize_onnx_int8,
    remove_excluded_labels,
    split_train_val,
    export_model_to_onnx,
)
from src.utils import load_config, load_model_config, resolve_path, seed_everything, setup_logging

config_path = Path(sys.argv[1])
model_config_path = Path(sys.argv[2])
calibration_override = int(sys.argv[3])

config = load_config(str(config_path))
model_config = load_model_config(str(model_config_path))["sign_recognition"]
train_config = config["training"]["recognition"]
optimization_config = config.get("optimization", {})

logs_dir = resolve_path(config["paths"].get("logs_dir", "logs"))
setup_logging(str(logs_dir))
seed_everything(config.get("project", {}).get("seed", 42))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Discover dataset and prepare loaders
dataset_root = resolve_path(config["paths"]["dataset_root"])
class_to_paths = discover_videos(dataset_root)
exclude_labels = train_config.get("exclude_labels", [])
class_to_paths = remove_excluded_labels(class_to_paths, exclude_labels)
class_to_idx = {label: idx for idx, label in enumerate(sorted(class_to_paths))}
model_config.setdefault("classifier", {})["num_classes"] = len(class_to_idx)

val_split = float(train_config.get("val_split", 0.0))
max_samples = train_config.get("max_samples_per_class")
max_samples = int(max_samples) if max_samples not in (None, "null", "None") else None
train_paths, val_paths = split_train_val(class_to_paths, val_split, max_samples)
calibration_paths = val_paths if val_paths else train_paths

if not calibration_paths:
    raise RuntimeError("No calibration videos available. Confirm dataset is populated.")

frame_size = tuple(model_config.get("frame_size", [112, 112]))
temporal_length = int(model_config.get("temporal_length", 32))
mean = tuple(config["preprocessing"].get("normalize_mean", [0.0, 0.0, 0.0]))
std = tuple(config["preprocessing"].get("normalize_std", [1.0, 1.0, 1.0]))

dataset_kwargs = dict(
    class_to_idx=class_to_idx,
    frame_size=frame_size,
    num_frames=temporal_length,
    cache=False,
    normalize=bool(train_config.get("normalize", False)),
    mean=mean,
    std=std,
    augmentations_config={},
    temporal_augmentations_config={},
    temporal_jitter=False,
    clip_stride=int(config["preprocessing"].get("clip_stride", 1)),
)

calibration_dataset = SignVideoDataset(
    video_paths=calibration_paths,
    train=False,
    augment=False,
    **dataset_kwargs,
)

batch_size = max(1, min(int(train_config.get("batch_size", 1)), 4))
calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

calibration_target = calibration_override or int(optimization_config.get("quantization_calibration_samples", 0))
calibration_samples = collect_calibration_samples(calibration_loader, calibration_target)

# Prepare model and export
checkpoint_dir = resolve_path(config["paths"]["recognition_checkpoint_dir"])
architecture = model_config.get("architecture", "sign_model")
checkpoint_path = checkpoint_dir / f"{architecture}_sign_recognition.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the recognition model first.")

model = build_sign_recognition_model(model_config).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

onnx_export_dir = resolve_path(config["paths"].get("onnx_export_dir", "models"))
onnx_path = onnx_export_dir / f"{architecture}.onnx"
fp16_path = onnx_export_dir / f"{architecture}.fp16.onnx"
int8_path = onnx_export_dir / f"{architecture}.int8.onnx"

onnx_export_dir.mkdir(parents=True, exist_ok=True)

dummy_input = torch.randn(
    1,
    model_config.get("input_channels", 3),
    temporal_length,
    frame_size[0],
    frame_size[1],
    device=device,
)

export_model_to_onnx(model, dummy_input, onnx_path)
fp16_success = convert_onnx_to_fp16(onnx_path, fp16_path)

int8_success = False
if calibration_samples:
    int8_success = quantize_onnx_int8(onnx_path, int8_path, calibration_samples)
else:
    print("[WARN] No calibration samples collected; skipping INT8 quantization.")

summary = {
    "onnx_path": str(onnx_path),
    "fp16_path": str(fp16_path) if fp16_success else None,
    "int8_path": str(int8_path) if int8_success else None,
    "calibration_samples": len(calibration_samples),
    "checkpoint": str(checkpoint_path),
}

print(json.dumps(summary, indent=2))
PY
)

python - <<PY "$CONFIG_PATH" "$MODEL_CONFIG_PATH" "$CALIBRATION_LIMIT"
$PYTHON_SCRIPT
PY
