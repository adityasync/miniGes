"""Utility script to inspect dataset class distribution."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"


def gather_counts(dataset_root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at {dataset_root}. "
            "Update config.paths.dataset_root or mount the dataset before running experiments."
        )
    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            videos = [
                file
                for file in class_dir.iterdir()
                if file.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
            ]
            if videos:
                counts[class_dir.name] += len(videos)
    return counts


def main() -> None:
    counts = gather_counts(DATASET_ROOT)
    total_videos = sum(counts.values())
    print(f"Total classes: {len(counts)}")
    print(f"Total videos: {total_videos}")

    print("\nTop 15 classes by sample count:")
    for label, count in counts.most_common(15):
        print(f"  {label:<20} {count}")

    print("\nBottom 15 classes by sample count:")
    for label, count in counts.most_common()[-15:]:
        print(f"  {label:<20} {count}")

    rare = [label for label, count in counts.items() if count < 3]
    print(f"\nClasses with fewer than 3 videos: {len(rare)}")
    if rare:
        print(", ".join(sorted(rare)))


if __name__ == "__main__":
    main()
