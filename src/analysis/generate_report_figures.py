from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


def _ensure_figures_dir(config: Dict[str, object]) -> Path:
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    reports_root = paths_cfg.get("reports_dir", "reports")
    reports_dir = Path(reports_root).expanduser().resolve()
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_series(records: Sequence[Dict[str, object]], key: str) -> List[float | None]:
    series: List[float | None] = []
    for record in records:
        value = record.get(key)
        if value is None:
            series.append(None)
        else:
            series.append(float(value))
    return series


def _plot_curve(
    x: Sequence[int],
    series_map: Dict[str, Sequence[float | None]],
    title: str,
    ylabel: str,
    figures_dir: Path,
    filename: str,
    *,
    logy: bool = False,
) -> Path:
    plt.figure(figsize=(10, 6))
    plot_func = plt.semilogy if logy else plt.plot
    for label, series in series_map.items():
        if all(value is None for value in series):
            continue
        numeric = np.array([np.nan if value is None else value for value in series], dtype=float)
        plot_func(x, numeric, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _parse_learning_rates(records: Sequence[Dict[str, object]]) -> Dict[str, List[float | None]]:
    parsed: List[List[float | None]] = []
    max_len = 0
    for record in records:
        lr_str = str(record.get("learning_rates", ""))
        values: List[float | None] = []
        if lr_str:
            for token in lr_str.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(None)
        parsed.append(values)
        max_len = max(max_len, len(values))

    lr_series: Dict[str, List[float | None]] = {}
    for idx in range(max_len):
        column: List[float | None] = []
        for row in parsed:
            column.append(row[idx] if idx < len(row) else None)
        lr_series[f"group_{idx + 1}"] = column

    return {name: values for name, values in lr_series.items() if any(v is not None for v in values)}


def _plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    epochs: Sequence[int],
    title: str,
    xlabel: str,
    ylabel: str,
    figures_dir: Path,
    filename: str,
) -> Path:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=epochs, cmap="viridis", s=80, edgecolors="black", linewidths=0.5)
    for epoch, xi, yi in zip(epochs, x, y):
        plt.annotate(str(epoch), (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Epoch")
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _compute_class_metrics(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _plot_per_class_metrics(
    class_names: Iterable[str],
    metrics: Dict[str, np.ndarray],
    figures_dir: Path,
    filename: str = "per_class_metrics.png",
) -> Path:
    names = list(class_names)
    if not names:
        raise ValueError("Class names required for per-class metrics plot")

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    order = np.argsort(recall)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_precision = precision[order]
    sorted_recall = recall[order]
    sorted_f1 = f1[order]

    indices = np.arange(len(sorted_names))
    width = 0.25

    plt.figure(figsize=(max(12, len(sorted_names) * 0.6), 7))
    plt.bar(indices - width, sorted_precision, width, label="Precision")
    plt.bar(indices, sorted_recall, width, label="Recall")
    plt.bar(indices + width, sorted_f1, width, label="F1-score")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall / F1")
    plt.xticks(indices, sorted_names, rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _plot_support_distribution(
    class_names: Iterable[str],
    supports: np.ndarray,
    figures_dir: Path,
    filename: str = "class_support_distribution.png",
) -> Path:
    names = list(class_names)
    values = supports.astype(float)
    order = np.argsort(values)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_values = values[order]

    plt.figure(figsize=(max(12, len(sorted_names) * 0.45), 6))
    plt.bar(sorted_names, sorted_values, color="#4c72b0")
    plt.ylabel("Samples")
    plt.title("Class Support Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _plot_confusion_matrix(
    matrix: Sequence[Sequence[int | float]],
    class_names: Iterable[str],
    figures_dir: Path,
    filename: str = "confusion_matrix.png",
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    array = np.array(matrix, dtype=float)
    if normalize:
        row_sums = array.sum(axis=1, keepdims=True)
        array = np.divide(array, row_sums, out=np.zeros_like(array), where=row_sums > 0)

    plt.figure(figsize=(max(8, array.shape[0] * 0.4), max(6, array.shape[1] * 0.4)))
    sns.heatmap(
        array,
        annot=array.shape[0] <= 30,
        fmt=".2f" if normalize else ".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={"label": "Normalized value" if normalize else "Predicted frequency"},
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = figures_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_report_figures(
    summary_path: str | Path = "logs/analysis/latest_training_summary.json",
    config_path: str | Path = "config/config.yaml",
) -> List[Path]:
    config = load_config(str(config_path))
    figures_dir = _ensure_figures_dir(config)

    summary = _load_json(Path(summary_path))
    epochs = summary.get("epochs", [])
    if not epochs:
        raise ValueError("No epoch metrics found in training summary.")

    epoch_numbers = [int(record.get("epoch", idx + 1)) for idx, record in enumerate(epochs)]
    train_loss = _safe_series(epochs, "train_loss")
    val_loss = _safe_series(epochs, "val_loss")
    train_acc = _safe_series(epochs, "train_acc")
    val_acc = _safe_series(epochs, "val_acc")
    train_top5 = _safe_series(epochs, "train_top5")
    val_top5 = _safe_series(epochs, "val_top5")

    generated_paths: List[Path] = []
    generated_paths.append(
        _plot_curve(
            epoch_numbers,
            {"Train": train_loss, "Validation": val_loss},
            title="Loss per Epoch",
            ylabel="Cross-Entropy Loss",
            figures_dir=figures_dir,
            filename="loss_curve.png",
        )
    )
    generated_paths.append(
        _plot_curve(
            epoch_numbers,
            {"Train": train_acc, "Validation": val_acc},
            title="Top-1 Accuracy per Epoch",
            ylabel="Accuracy",
            figures_dir=figures_dir,
            filename="accuracy_curve.png",
        )
    )
    generated_paths.append(
        _plot_curve(
            epoch_numbers,
            {"Train": train_top5, "Validation": val_top5},
            title="Top-5 Accuracy per Epoch",
            ylabel="Accuracy",
            figures_dir=figures_dir,
            filename="top5_accuracy_curve.png",
        )
    )

    lr_series = _parse_learning_rates(epochs)
    if lr_series:
        generated_paths.append(
            _plot_curve(
                epoch_numbers,
                lr_series,
                title="Learning Rate Schedule",
                ylabel="Learning Rate",
                figures_dir=figures_dir,
                filename="learning_rate_schedule.png",
                logy=True,
            )
        )

    val_loss_numeric = np.array([np.nan if value is None else value for value in val_loss], dtype=float)
    val_acc_numeric = np.array([np.nan if value is None else value for value in val_acc], dtype=float)
    valid_mask = ~np.isnan(val_loss_numeric) & ~np.isnan(val_acc_numeric)
    if np.any(valid_mask):
        generated_paths.append(
            _plot_scatter(
                val_loss_numeric[valid_mask].tolist(),
                val_acc_numeric[valid_mask].tolist(),
                np.array(epoch_numbers)[valid_mask].tolist(),
                title="Validation Loss vs Accuracy",
                xlabel="Validation Loss",
                ylabel="Validation Accuracy",
                figures_dir=figures_dir,
                filename="val_loss_vs_accuracy.png",
            )
        )

    confusion_matrix_path = summary.get("latest_archive")
    if confusion_matrix_path:
        archive_path = Path(confusion_matrix_path)
        if archive_path.exists():
            archive_data = _load_json(archive_path)
            matrix = archive_data.get("confusion_matrix")
            class_names = archive_data.get("class_names")
            if matrix and class_names:
                class_names_list = list(class_names)
                matrix_array = np.array(matrix, dtype=float)
                metrics = _compute_class_metrics(matrix_array)
                generated_paths.append(
                    _plot_confusion_matrix(
                        matrix_array,
                        class_names_list,
                        figures_dir,
                    )
                )
                generated_paths.append(
                    _plot_confusion_matrix(
                        matrix_array,
                        class_names_list,
                        figures_dir,
                        filename="confusion_matrix_normalized.png",
                        title="Normalized Confusion Matrix",
                        normalize=True,
                    )
                )
                generated_paths.append(
                    _plot_per_class_metrics(
                        class_names_list,
                        metrics,
                        figures_dir,
                    )
                )
                generated_paths.append(
                    _plot_support_distribution(
                        class_names_list,
                        metrics["support"],
                        figures_dir,
                    )
                )

    return generated_paths


if __name__ == "__main__":
    paths = generate_report_figures()
    for path in paths:
        print(f"Saved figure to {path}")
