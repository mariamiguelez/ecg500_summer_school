from __future__ import annotations

import json
from pathlib import Path

import joblib

from utils import load_config
from models.baseline import build_baseline_model
from utils import classification_metrics


def main() -> None:
    cfg = load_config("configs/default.yaml")
    paths = cfg["paths"]
    training = cfg["training"]

    x_train, y_train, x_test, y_test = paths
    model = build_baseline_model()
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    train_metrics = classification_metrics(y_train, y_pred_train)
    test_metrics = classification_metrics(y_test, y_pred_test)

    models_dir = Path(paths["models_dir"])
    reports_dir = Path(paths["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "baseline.joblib")

    with (reports_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    with (reports_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Training complete.")
    print(f"Saved model to: {models_dir / 'baseline.joblib'}")
    print(f"Saved reports to: {reports_dir}")


if __name__ == "__main__":
    main()

