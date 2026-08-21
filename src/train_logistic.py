from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing.preprocess import main as preprocess_main
from models.logistic_baseline import fit_logistic_model
from utils import load_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")
    paths = config["paths"]

    processed_dir = project_root / paths["processed_data_dir"]
    models_dir = project_root / paths["models_dir"]
    reports_dir = project_root / paths["reports_dir"]
    train_path = processed_dir / "train_structured.csv"
    test_path = processed_dir / "test_structured.csv"

    if not train_path.exists() or not test_path.exists():
        print("Structured data not found. Running preprocessing first...")
        preprocess_main()

    train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

    x_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int)

    model = fit_logistic_model(x_train=x_train, y_train=y_train)
    #logreg = model.named_steps['classifier']
    #print(logreg.coef_.shape)
    #print(logreg.intercept_.shape)
    #print(model.coef_.size + model.intercept_.size)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "baseline_logistic_regression.joblib")
    pd.DataFrame([metrics]).to_csv(
        reports_dir / "logistic_regression_metrics.csv", index=False
    )
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        reports_dir / "logistic_regression_predictions.csv", index=False
    )

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test f1_macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()