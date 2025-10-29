from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PREFS_PATH = Path(__file__).resolve().parent / "notebook_prefs.json"


def caption_block(text: str) -> str:
    return text.strip() + "\n"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            caption_block(
                """
# ISL Sign Recognition – Experiment Dossier

This notebook is crafted for presenting the Indian Sign Language recognition system to judges. It summarises the dataset, documents every conducted experiment, and highlights the best-performing checkpoint with reproducible configuration details.
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Quick start checklist

1. Activate the project environment: `!pyenv activate miniGes`
2. Confirm you are in the repository root (`sinGes(mini)/`).
3. Ensure the INCLUDE dataset is available under `dataset/` (see `Roadmap.md`).
4. Run each experiment cell sequentially; results are captured in the summary table automatically.
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Project overview

- **Objective:** 54-class Indian Sign Language recognition powered by a pretrained 3D ResNet (R3D-18) backbone.
- **Baseline accuracy:** ~35% Top-1 / ~65% Top-5 on validation split.
- **Success criteria:** Push Top-1 accuracy beyond the current baseline while maintaining robust Top-5 coverage.
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path('..').resolve()
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

prefs = {}
if NOTEBOOK_PREFS_PATH.exists():
    prefs = json.loads(NOTEBOOK_PREFS_PATH.read_text())
DATASET_ROOT = Path(prefs.get('dataset_root', PROJECT_ROOT / 'dataset'))
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset root: {DATASET_ROOT}")
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
from tmp.dataset_summary import gather_counts
import pandas as pd

try:
    counts = gather_counts(DATASET_ROOT)
except FileNotFoundError as exc:
    print(exc)
    counts = {}

if counts:
    df_counts = (
        pd.DataFrame(sorted(counts.items()), columns=['label', 'videos'])
        .sort_values('videos', ascending=False)
        .reset_index(drop=True)
    )
    display(df_counts.head(10))
    display(df_counts.tail(10))
    print(f"Total classes: {df_counts.shape[0]} | Total videos: {int(df_counts['videos'].sum())}")
else:
    print("Dataset summary unavailable. Please ensure the dataset is mounted and rerun the setup script:"
          " `PYENV_VERSION=miniGes pyenv exec python tmp/setup_notebook_env.py`.")
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Experiment registry

Each experiment is defined by a name, configuration overrides, and the resulting metrics. The registry below collects results so judges can quickly compare experiments and identify the best-performing model.
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
import yaml
from pprint import pprint
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


TRAIN_CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'
MODEL_CONFIG_PATH = PROJECT_ROOT / 'config' / 'model_config.yaml'


def load_configs():
    with TRAIN_CONFIG_PATH.open() as fp:
        train_cfg = yaml.safe_load(fp)
    with MODEL_CONFIG_PATH.open() as fp:
        model_cfg = yaml.safe_load(fp)
    return train_cfg, model_cfg


train_cfg_base, model_cfg_base = load_configs()

print('Recognition training config:')
pprint(train_cfg_base['training']['recognition'])
print('\nModel config:')
pprint(model_cfg_base['sign_recognition'])


@dataclass
class Experiment:
    name: str
    training_overrides: Dict[str, Any]
    model_overrides: Dict[str, Any]
    notes: str
    result: Dict[str, Any] | None = None


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="Baseline – Pretrained R3D-18",
        training_overrides={},
        model_overrides={},
        notes="Reference run using config defaults (layer4 unfrozen, 25 epochs).",
    ),
    Experiment(
        name="FT Layers 3&4 + Longer Training",
        training_overrides={
            'learning_rate': 2e-4,
            'batch_size': 4,
            'num_epochs': 30,
            'early_stopping_patience': 10,
        },
        model_overrides={
            'freeze_backbone': False,
            'unfreeze_layers': ['layer3', 'layer4'],
        },
        notes="Fine-tune deeper layers with extended training and lower LR.",
    ),
    Experiment(
        name="Label Smoothing + Patience",
        training_overrides={
            'learning_rate': 2.5e-4,
            'num_epochs': 30,
            'early_stopping_patience': 12,
        },
        model_overrides={
            'freeze_backbone': False,
            'unfreeze_layers': ['layer4'],
        },
        notes="Placeholder for label smoothing variant (implemented in code when ready).",
    ),
]
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
from contextlib import contextmanager
from copy import deepcopy


def _deep_update(mapping, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(mapping.get(key), dict):
            _deep_update(mapping[key], value)
        else:
            mapping[key] = value


@contextmanager
def apply_experiment_overrides(training_overrides=None, model_overrides=None):
    training_overrides = training_overrides or {}
    model_overrides = model_overrides or {}

    train_cfg, model_cfg = load_configs()

    updated_train_cfg = deepcopy(train_cfg)
    updated_model_cfg = deepcopy(model_cfg)

    if training_overrides:
        _deep_update(updated_train_cfg['training']['recognition'], training_overrides)
    if model_overrides:
        _deep_update(updated_model_cfg['sign_recognition'], model_overrides)

    with TRAIN_CONFIG_PATH.open('w') as fp:
        yaml.safe_dump(updated_train_cfg, fp, sort_keys=False)
    with MODEL_CONFIG_PATH.open('w') as fp:
        yaml.safe_dump(updated_model_cfg, fp, sort_keys=False)

    try:
        yield updated_train_cfg, updated_model_cfg
    finally:
        with TRAIN_CONFIG_PATH.open('w') as fp:
            yaml.safe_dump(train_cfg, fp, sort_keys=False)
        with MODEL_CONFIG_PATH.open('w') as fp:
            yaml.safe_dump(model_cfg, fp, sort_keys=False)
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block("from src.train_recognition import train")
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Run experiments

Execute each cell below to run the configured experiments. Results are stored in the `EXPERIMENTS` list and later summarised.
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
import math

experiment_outputs = []

for exp in EXPERIMENTS:
    print(f"\n=== Running: {exp.name} ===")
    print(exp.notes)
    with apply_experiment_overrides(exp.training_overrides, exp.model_overrides):
        result = train()
    exp.result = result
    experiment_outputs.append({**asdict(exp), 'result': result})
    print("Top-1 accuracy:", result['accuracy'])
    print("Top-5 accuracy:", result['top5'])
    print("Checkpoint:", result['checkpoint_path'])
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Experiment summary table

The table below aggregates the outcomes of every experiment, highlighting the best accuracy and the corresponding checkpoint path.
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
import pandas as pd


def summarise_experiments(experiments):
    rows = []
    for exp in experiments:
        if not exp.result:
            continue
        rows.append(
            {
                'Experiment': exp.name,
                'Top-1 Accuracy': exp.result['accuracy'],
                'Top-5 Accuracy': exp.result['top5'],
                'Checkpoint': exp.result['checkpoint_path'],
                'Notes': exp.notes,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values('Top-1 Accuracy', ascending=False).reset_index(drop=True)
    return df


summary_df = summarise_experiments(EXPERIMENTS)
if not summary_df.empty:
    display(summary_df)
    best_row = summary_df.iloc[0]
    print("\nBEST MODEL SUMMARY")
    print("Experiment:", best_row['Experiment'])
    print("Top-1 Accuracy:", best_row['Top-1 Accuracy'])
    print("Top-5 Accuracy:", best_row['Top-5 Accuracy'])
    print("Checkpoint Path:", best_row['Checkpoint'])
else:
    print("No experiments have been executed yet.")
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Inspect latest training logs

Review the training log tail to demonstrate convergence behaviour during the presentation.
"""
            )
        ),
        nbf.v4.new_code_cell(
            caption_block(
                """
log_dir = PROJECT_ROOT / 'logs'
if log_dir.exists():
    log_files = list(log_dir.glob('*.log'))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"Latest log file: {latest_log}")
        with latest_log.open() as fp:
            lines = fp.readlines()
        print(''.join(lines[-50:]))
    else:
        print('No log files found in the logs directory yet.')
else:
    print('Log directory not found.')
"""
            )
        ),
        nbf.v4.new_markdown_cell(
            caption_block(
                """
## Presentation talking points

- Highlight the steady improvement from the baseline to the fine-tuned models (Top-1 and Top-5 accuracy trends).
- Emphasise that checkpoints are stored in `models/checkpoints/sign_recognition/` and list the chosen best model path.
- Discuss next planned experiments (label smoothing, SWA, alternative backbones).
- Reiterate dataset challenges (54 classes, ~5 videos each) and how augmentation + transfer learning mitigated them.
"""
            )
        ),
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }

    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "model_experiments.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, notebook_path)
    print(f"Notebook written to {notebook_path}")


if __name__ == "__main__":
    main()
