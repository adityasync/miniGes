from pathlib import Path
import re
import json

LOG_PATH = Path("logs/project.log")
ARCHIVE_DIR = Path("logs/sign_recognition_runs")
OUT_PATH = Path("logs/analysis/latest_training_summary.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not LOG_PATH.exists():
    raise FileNotFoundError(LOG_PATH)

content = LOG_PATH.read_text(encoding="utf-8")
pattern = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+LRs \[(?P<lrs>[^\]]+)\] \|\s+Train loss: (?P<train_loss>\d+\.\d+) acc: (?P<train_acc>\d+\.\d+) top5: (?P<train_top5>\d+\.\d+) \| Val loss: (?P<val_loss>\d+\.\d+) acc: (?P<val_acc>\d+\.\d+) top5: (?P<val_top5>\d+\.\d+)"
)

results = []
for match in pattern.finditer(content):
    results.append({
        "epoch": int(match.group("epoch")),
        "learning_rates": match.group("lrs"),
        "train_loss": float(match.group("train_loss")),
        "train_acc": float(match.group("train_acc")),
        "train_top5": float(match.group("train_top5")),
        "val_loss": float(match.group("val_loss")),
        "val_acc": float(match.group("val_acc")),
        "val_top5": float(match.group("val_top5")),
    })

summary = {
    "epochs": results,
    "best_epoch": max(results, key=lambda r: r["val_acc"], default=None),
    "latest_archive": None,
}

if ARCHIVE_DIR.exists():
    candidates = sorted(ARCHIVE_DIR.glob("*.json"))
    if candidates:
        summary["latest_archive"] = str(candidates[-1])

OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved summary to {OUT_PATH}")
