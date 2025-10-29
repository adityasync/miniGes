from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
NOTEBOOK_PREFS = Path(__file__).resolve().parents[1] / "tmp" / "notebook_prefs.json"


def load_dataset_root() -> str:
    import yaml

    with CONFIG_PATH.open() as fp:
        cfg = yaml.safe_load(fp)
    return str((Path(__file__).resolve().parents[1] / cfg["paths"]["dataset_root"]).resolve())


def main() -> None:
    prefs = {"dataset_root": load_dataset_root()}
    NOTEBOOK_PREFS.write_text(json.dumps(prefs, indent=2))
    print(f"Wrote notebook preferences to {NOTEBOOK_PREFS} -> {prefs['dataset_root']}")


if __name__ == "__main__":
    main()
