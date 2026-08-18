from __future__ import annotations

import json
from pathlib import Path

from utils import load_config
from utils import classification_metrics
from models.baseline import build_baseline_model

def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = cfg["paths"]

    _, _, x_test, y_test = paths
    model = build_baseline_model()
    y_pred = model.predict(x_test)
    metrics = classification_metrics(y_test, y_pred)

    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "evaluation_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()

