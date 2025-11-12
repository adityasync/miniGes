#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"

CONFIG_PATH=${1:-config/config.yaml}
MODEL_CONFIG_PATH=${2:-config/model_config.yaml}
MODE=${3:-torch}
WARMUP=${4:-5}
ITERS=${5:-50}

PYTHON_SCRIPT=$(cat <<'PY'
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.train_recognition import build_sign_recognition_model
from src.utils import load_config, load_model_config, resolve_path, seed_everything

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

config_path = Path(sys.argv[1])
model_config_path = Path(sys.argv[2])
mode = sys.argv[3]
warmup = int(sys.argv[4])
iters = int(sys.argv[5])

config = load_config(str(config_path))
model_config = load_model_config(str(model_config_path))["sign_recognition"]
seed_everything(config.get("project", {}).get("seed", 42))

frame_size = tuple(model_config.get("frame_size", [112, 112]))
num_frames = int(model_config.get("temporal_length", 32))
channels = int(model_config.get("input_channels", 3))

input_tensor = torch.randn(1, channels, num_frames, frame_size[0], frame_size[1])

result = {
    "mode": mode,
    "warmup": warmup,
    "iterations": iters,
}

if mode == "torch":
    checkpoint_dir = resolve_path(config["paths"]["recognition_checkpoint_dir"])
    architecture = model_config.get("architecture", "sign_model")
    checkpoint_path = checkpoint_dir / f"{architecture}_sign_recognition.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sign_recognition_model(model_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    input_tensor = input_tensor.to(device)

    def _run():
        with torch.no_grad():
            output = model(input_tensor)
        return output

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize() if device.type == "cuda" else None

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _run()
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - start)

    result.update(
        {
            "latency_ms_mean": float(np.mean(times) * 1000),
            "latency_ms_p95": float(np.percentile(times, 95) * 1000),
            "device": str(device),
            "fps": float(1.0 / np.mean(times)),
        }
    )

elif mode in {"onnx", "onnx_fp16", "onnx_int8"}:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX benchmarking")
    export_dir = resolve_path(config["paths"].get("onnx_export_dir", "models"))
    architecture = model_config.get("architecture", "sign_model")
    suffix = {
        "onnx": ".onnx",
        "onnx_fp16": ".fp16.onnx",
        "onnx_int8": ".int8.onnx",
    }[mode]
    model_path = export_dir / f"{architecture}{suffix}"
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model {model_path} missing. Run quantization script first.")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)

    input_name = session.get_inputs()[0].name
    input_array = input_tensor.numpy().astype(np.float32)

    def _run():
        outputs = session.run(None, {input_name: input_array})
        return outputs

    for _ in range(warmup):
        _run()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _run()
        times.append(time.perf_counter() - start)

    result.update(
        {
            "latency_ms_mean": float(np.mean(times) * 1000),
            "latency_ms_p95": float(np.percentile(times, 95) * 1000),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "fps": float(1.0 / np.mean(times)),
        }
    )
else:
    raise ValueError(f"Unknown benchmarking mode: {mode}")

print(json.dumps(result, indent=2))
PY
)

python - <<PY "$CONFIG_PATH" "$MODEL_CONFIG_PATH" "$MODE" "$WARMUP" "$ITERS"
$PYTHON_SCRIPT
PY
