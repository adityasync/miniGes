"""Generate synthetic vocabulary CSV for sinGes-mini if missing."""

from __future__ import annotations

import csv
from pathlib import Path

VOCAB_WORDS = [
    "Hello",
    "Thank you",
    "Please",
    "Yes",
    "No",
    "Good",
    "Bad",
    "Morning",
    "Night",
    "Friend",
    "Help",
    "Where",
    "Why",
    "Food",
    "Water",
    "Love",
    "Happy",
    "Sorry",
    "Time",
    "Day",
    "Family",
    "School",
    "Work",
    "Home",
    "Go",
    "Come",
    "Stop",
    "Wait",
    "Play",
    "Learn",
]


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    vocab_path = dataset_dir / "ISL_30_words_complete.csv"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word"])
        for word in VOCAB_WORDS:
            writer.writerow([word])
    print(f"Created vocabulary file at {vocab_path}")


if __name__ == "__main__":
    main()
