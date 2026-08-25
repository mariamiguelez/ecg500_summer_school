import importlib
from pathlib import Path
import sys
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from utils import load_config
from data_processing.preprocess import main as preprocess_main

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _load_structured_data(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load the preprocessed data in and proceed to te train test split
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
    return x_train, y_train, x_test, y_test


def _split_validation_from_test(
    x_test: np.ndarray,
    y_test: np.ndarray,
    validation_fraction: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract the validation data from the test set based on the fraccion defined
    x_val, x_test_holdout, y_val, y_test_holdout = train_test_split(
        x_test,
        y_test,
        test_size=1.0 - validation_fraction,
        stratify=y_test,
        random_state=random_state,
    )
    return x_val, y_val, x_test_holdout, y_test_holdout


def _fit_model_from_config(
    model_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    # Select the model desired and run
    module = importlib.import_module(model_cfg["module"])
    fit_function = getattr(module, model_cfg["fit_function"])
    fit_params = model_cfg.get("fit_params", {})
    return fit_function(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=model_cfg.get("random_state", 42),
        **fit_params,
    )


def _scale_test_data(
    x_test: np.ndarray,
    model: Any,
    model_cfg: dict[str, Any],
) -> np.ndarray:
    fit_params = model_cfg.get("fit_params", {})
    use_scaling = fit_params.get("use_scaling", False)
    if not use_scaling:
        print("No scaling applied on test data")
        return x_test
    print("Scaling test data")
    scaler = getattr(model, "scaler", None)
    if scaler is None:
        raise ValueError(
            "Config sets use_scaling=true, but trained model has no scaler."
        )

    if x_test.ndim == 2 and getattr(scaler, "n_features_in_", None) == 1:
        return scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)

    return scaler.transform(x_test)


def main() -> None:
    # Load paths
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "default.yaml")
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    models_cfg = cfg["models"]

    processed_dir = project_root / paths["processed_data_dir"]
    models_dir = project_root / paths["models_dir"]
    reports_dir = project_root / paths["reports_dir"]

    selected_model = training_cfg["selected_model"]
    if selected_model not in models_cfg:
        raise KeyError(f"Model '{selected_model}' not found under 'models' in config.")
    x_train, y_train, x_test, y_test = _load_structured_data(processed_dir=processed_dir)
    validation_fraction = training_cfg.get("validation_from_test_fraction",0.5)

    x_val, y_val, x_test, y_test = _split_validation_from_test(
        x_test=x_test,
        y_test=y_test,
        validation_fraction=validation_fraction,
        random_state=42,
    )

    print(
        f"Train shape: {x_train.shape}, Validation shape: {x_val.shape}, "
        f"Test shape: {x_test.shape}"
    )
    print(f"Performing training on: {selected_model}")
    print('-----------------------------------------')

    model_cfg = models_cfg[selected_model]
    model = _fit_model_from_config(
        model_cfg=model_cfg,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    print('-----------------------------------------')
    print('Saving predictions')
    print('-----------------------------------------')
    print('Evaluating')

    # sacale only if indicated
    x_test_s= _scale_test_data(x_test=x_test, model=model, model_cfg=model_cfg)
    y_pred = model.predict(x_test_s)

    # Evaluation
    metrics = {
        "model": selected_model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred,  average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_cfg["artifact_name"]
    metrics_path = reports_dir / f"train_metrics_{selected_model}.csv"
    predictions_path = reports_dir / f"test_predictions_{selected_model}.csv"

    joblib.dump(model, model_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    ).to_csv(predictions_path, index=False)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"Test f1_macro: {metrics['f1_macro']:.4f}")
    print(f"Saved model to: {model_path.resolve()}")
    print(f"Saved metrics to: {metrics_path.resolve()}")
    print(f"Saved predictions to: {predictions_path.resolve()}")


if __name__ == "__main__":
    main()
