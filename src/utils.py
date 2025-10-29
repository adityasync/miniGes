"""Utility helpers for the sinGes-mini project.

This module centralises common helpers such as configuration loading, logging
initialisation, and reproducibility utilities.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load the main project configuration."""

    return load_yaml(Path(config_path))


def load_model_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load model-specific hyperparameters."""

    return load_yaml(Path(config_path))


def seed_everything(seed: int) -> None:
    """Seed major random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(logs_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure root logger to output both to file and stdout."""

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / "project.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path for a possibly relative input."""

    return Path(path).expanduser().resolve()


_LABEL_PREFIX_PATTERN = re.compile(r"^\s*\d+\.\s*")


def discover_class_labels(dataset_root: str | Path) -> List[str]:
    """Return a sorted list of label folder names under the dataset root."""

    dataset_path = resolve_path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at {dataset_path}. "
            "Confirm config.paths.dataset_root is correct."
        )

    labels: List[str] = []
    for category_dir in sorted(dataset_path.iterdir()):
        if not category_dir.is_dir():
            continue
        for label_dir in sorted(category_dir.iterdir()):
            if label_dir.is_dir():
                labels.append(label_dir.name)

    if not labels:
        raise RuntimeError(
            f"No label folders discovered beneath {dataset_path}. Populate the dataset first."
        )

    return sorted(labels)


def strip_label_prefix(label: str) -> str:
    """Remove numeric prefixes (e.g. '48. Hello' -> 'Hello') for display purposes."""

    cleaned = _LABEL_PREFIX_PATTERN.sub("", label).strip()
    return cleaned or label.strip()
