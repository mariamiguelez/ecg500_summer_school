from __future__ import annotations

from pathlib import Path
import sys

import joblib
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
    target_column = config.get("training", {}).get("target_column", "target")

    processed_dir = project_root / paths["processed_data_dir"]
    models_dir = project_root / paths["models_dir"]
    reports_dir = project_root / paths["reports_dir"]
    train_path = processed_dir / "train_structured.csv"
    test_path = processed_dir / "test_structured.csv"

    if not train_path.exists() or not test_path.exists():
        print("Structured data not found. Running preprocessing first...")
        preprocess_main()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if target_column == "target_binary_gt1_to0" and target_column not in train_df.columns:
        train_df[target_column] = (train_df["target"].astype(int) <= 1).astype(int)
        test_df[target_column] = (test_df["target"].astype(int) <= 1).astype(int)

    feature_columns = [column for column in train_df.columns if column.startswith("x_")]
    x_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=int)
    x_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df[target_column].to_numpy(dtype=int)

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