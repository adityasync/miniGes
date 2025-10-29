import re
from pathlib import Path
from typing import Dict, List

THRESHOLD = 0.10
LOG_PATH = Path("logs/project.log")


def extract_classification_report(text: str) -> List[Dict[str, float]]:
    marker = "Classification report:"
    last_index = text.rfind(marker)
    if last_index == -1:
        raise RuntimeError("No classification report found in the log file.")

    block = text[last_index + len(marker) :]
    lines = block.strip().splitlines()

    pattern = re.compile(
        r"^(?P<label>.+?)\s+"
        r"(?P<precision>\d+\.\d+)\s+"
        r"(?P<recall>\d+\.\d+)\s+"
        r"(?P<f1>\d+\.\d+)\s+"
        r"(?P<support>\d+)\s*$"
    )

    entries: List[Dict[str, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.lower().startswith("accuracy"):
            break
        match = pattern.match(stripped)
        if not match:
            continue
        data = match.groupdict()
        entries.append(
            {
                "label": data["label"],
                "precision": float(data["precision"]),
                "recall": float(data["recall"]),
                "f1": float(data["f1"]),
                "support": int(data["support"]),
            }
        )

    return entries


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found at {LOG_PATH}")

    content = LOG_PATH.read_text(encoding="utf-8")
    metrics = extract_classification_report(content)

    low_recall = [m for m in metrics if m["recall"] <= THRESHOLD]
    low_recall.sort(key=lambda m: (m["recall"], m["precision"]))

    print(f"Total classes parsed: {len(metrics)}")
    print(f"Classes with recall <= {THRESHOLD:.2f}: {len(low_recall)}\n")
    for entry in low_recall:
        print(
            f"{entry['label']}: precision={entry['precision']:.2f}, "
            f"recall={entry['recall']:.2f}, f1={entry['f1']:.2f}, support={entry['support']}"
        )

    if low_recall:
        exclude_labels = ", ".join(f"'{entry['label']}'" for entry in low_recall)
        print("\nSuggested exclude_labels list:")
        print(f"[{exclude_labels}]")


if __name__ == "__main__":
    main()
