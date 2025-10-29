"""List all label words present in the dataset directory."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


LABEL_PREFIX_PATTERN = re.compile(r"^\s*\d+\.\s*")


def strip_label_prefix(label: str) -> str:
    cleaned = LABEL_PREFIX_PATTERN.sub("", label).strip()
    return cleaned or label.strip()


def discover_class_labels(dataset_root: Path):
    labels = []
    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for label_dir in sorted(category_dir.iterdir()):
            if label_dir.is_dir():
                labels.append(label_dir.name)
    if not labels:
        raise RuntimeError(f"No label folders discovered in {dataset_root}")
    return labels


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_root = (PROJECT_ROOT / config["paths"]["dataset_root"]).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found at {dataset_root}")

    labels = discover_class_labels(dataset_root)
    rows = []
    category_counts = defaultdict(int)

    for label in labels:
        cleaned = strip_label_prefix(label)
        category = label.split(".")[0] if "." in label else "unknown"
        category_counts[category] += 1
        rows.append((category, label, cleaned))

    full_csv = dataset_root / "ISL_words_full.csv"
    with full_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["category", "raw_label", "word"])
        writer.writerows(rows)

    vocab_csv = dataset_root / "ISL_30_words_complete.csv"
    with vocab_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word"])
        for _, _, cleaned in rows:
            writer.writerow([cleaned])

    print(f"Discovered {len(rows)} labels. Wrote summaries to:\n - {full_csv}\n - {vocab_csv}")


if __name__ == "__main__":
    main()
