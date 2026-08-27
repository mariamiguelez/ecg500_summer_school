import argparse
import gc
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from train import (
    _filter_allowed_classes,
    _load_structured_data,
)
from utils import load_config

RESULTS_DIR = Path(__file__).resolve().parent / "models" / "optuna_results"
MODELS_SUPPORTED = {"transformer", "xlstm"}


def _resolve_model_name(raw_model: str | None, default_model: str) -> str:
    selected = (raw_model or default_model).lower()
    if selected == "encoder":
        selected = "transformer"
    if selected not in MODELS_SUPPORTED:
        raise ValueError(
            f"Unsupported model '{selected}'. Supported models: {sorted(MODELS_SUPPORTED)}."
        )
    return selected

def _build_tunable_fit_params(
    trial: optuna.Trial,
    fit_params: dict[str, Any],
) -> dict[str, Any]:
    tuned: dict[str, Any] = {}

    """ Define the bounds for parameter tuning """
    for key, value in fit_params.items():
        if key == "loss_weights":
            if not isinstance(value, list) or not value:
                raise ValueError("fit_params.loss_weights must be a non-empty list.")
            tuned_weights = []
            for idx, base in enumerate(value):
                base_weight = float(base) if float(base) > 0 else 1.0
                low = max(1e-3, base_weight * 0.2)
                high = max(low * 1.1, base_weight * 5.0)
                tuned_weights.append(
                    trial.suggest_float(f"loss_weight_{idx}", low, high, log=True)
                )
            tuned[key] = tuned_weights
            continue

        if isinstance(value, bool):
            tuned[key] = trial.suggest_categorical(key, [True, False])
            continue

        if isinstance(value, int):
            if key == "input_size":
                # Input feature size is constrained by dataset shape.
                tuned[key] = value
                continue
            if key == "d_model":
                low = max(2, int(round(value * 0.5)))
                high = max(low + 2, int(round(value * 2.0)))
                if low % 2 != 0:
                    low += 1
                if high % 2 != 0:
                    high -= 1
                if low > high:
                    low, high = 2, max(4, 2 * value)
                tuned[key] = trial.suggest_int(key, low, high, step=2)
                continue
            if key == "batch_size":
                # Keep memory bounded during repeated fold/trial trainings.
                low = max(16, int(round(value * 0.25)))
                high = max(low + 1, min(int(round(value * 1.25)), 1024))
                tuned[key] = trial.suggest_int(key, low, high)
                continue
            low = max(1, int(round(value * 0.5)))
            high = max(low + 1, int(round(value * 2.0)))
            if key == "n_blocks":
                low = max(low, 2)
            tuned[key] = trial.suggest_int(key, low, high)
            continue

        if isinstance(value, float):
            low = max(1e-5, value / 10.0)
            high = max(low * 1.1, value * 10.0)
            tuned[key] = trial.suggest_float(key, low, high, log=True)
            continue

        tuned[key] = value

    return tuned


def _validate_trial_params(model_name: str, fit_params: dict[str, Any]) -> None:
    d_model = fit_params.get("d_model")
    n_heads = fit_params.get("n_heads")
    qkv_proj_blocksize = fit_params.get("qkv_proj_blocksize")

    if d_model is not None and int(d_model) % 2 != 0:
        raise optuna.exceptions.TrialPruned("d_model must be even for positional encoding.")

    if model_name == "transformer" and d_model is not None and n_heads is not None:
        if int(d_model) % int(n_heads) != 0:
            raise optuna.exceptions.TrialPruned(
                "d_model must be divisible by n_heads for transformer encoder."
            )

    if model_name == "xlstm" and d_model is not None and qkv_proj_blocksize is not None:
        if int(d_model) % int(qkv_proj_blocksize) != 0:
            raise optuna.exceptions.TrialPruned(
                "d_model must be divisible by qkv_proj_blocksize for xLSTM."
            )


def _fit_model(
    model_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    fit_params: dict[str, Any],
):
    module = importlib.import_module(model_cfg["module"])
    fit_function = getattr(module, model_cfg["fit_function"])
    call_fit_params = dict(fit_params)
    signature = inspect.signature(fit_function)
    if "plot_losses" in signature.parameters and "plot_losses" not in call_fit_params:
        # Avoid matplotlib accumulation while running many CV folds.
        call_fit_params["plot_losses"] = False
    return fit_function(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=model_cfg.get("random_state", 42),
        **call_fit_params,
    )


def _scale_validation(
    x_val: np.ndarray,
    adapter: Any,
    use_scaling: bool,
) -> np.ndarray:
    if not use_scaling:
        return x_val
    scaler = getattr(adapter, "scaler", None)
    if scaler is None:
        raise ValueError(
            "use_scaling=True but trained model adapter does not expose a scaler."
        )
    if x_val.ndim == 2 and getattr(scaler, "n_features_in_", None) == 1:
        return scaler.transform(x_val.reshape(-1, 1)).reshape(x_val.shape)
    return scaler.transform(x_val)


def _crossentropy_on_validation(
    adapter: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    loss_weights: list[float],
    batch_size: int,
) -> float:
    model = adapter.model
    device = adapter.device
    classes = np.asarray(adapter.classes)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    """CrossEntropy loss of `model` over an entire `loader`."""

    try:
        y_indices = np.array([class_to_idx[int(label)] for label in y_val], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            "Validation labels contain classes not present in the trained model."
        ) from exc

    if len(loss_weights) != len(classes):
        raise ValueError(
            "loss_weights length must match the number of model classes. "
            f"Got {len(loss_weights)} weights for {len(classes)} classes."
        )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weights, dtype=torch.float32, device=device)
    )
    dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_indices, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            if x_batch.dim() == 2:
                x_batch = x_batch.unsqueeze(-1)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            batch_loss = criterion(logits, y_batch)
            batch_size_actual = y_batch.shape[0]
            total_loss += batch_loss.item() * batch_size_actual
            total_samples += batch_size_actual

    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for encoder/transformer and xLSTM models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to optimize: transformer, encoder (alias), or xlstm. Defaults to training.selected_model.",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold folds over training set (>=2 recommended).",
    )
    args = parser.parse_args()

    # Extract parameters and arguments from the config
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")
    paths_cfg = config["paths"]
    training_cfg = config["training"]
    models_cfg = config["models"]
    selected_model = _resolve_model_name(
        raw_model=args.model,
        default_model=training_cfg.get("selected_model", "transformer"),
    )
    model_cfg = models_cfg[selected_model]
    base_fit_params = dict(model_cfg.get("fit_params", {}))
    if "loss_weights" not in base_fit_params:
        raise KeyError(
            f"Model '{selected_model}' must define fit_params.loss_weights for weighted CrossEntropy optimization."
        )

    processed_dir = project_root / paths_cfg["processed_data_dir"]
    target_column = training_cfg.get("target_column", "target")
    allowed_classes = training_cfg.get("allowed_classes")

    x_train, y_train, x_test, y_test = _load_structured_data(
        processed_dir=processed_dir,
        target_column=target_column,
    )
    x_train, y_train, x_test, y_test = _filter_allowed_classes(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        allowed_classes=allowed_classes,
    )
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2 for leakage-safe cross-validation.")
    if args.cv_folds > len(y_train):
        raise ValueError(
            f"--cv-folds ({args.cv_folds}) cannot exceed training size ({len(y_train)})."
        )

    torch.manual_seed(int(model_cfg.get("random_state", 42)))
    np.random.seed(int(model_cfg.get("random_state", 42)))

    # Apply optimization per trial
    def objective(trial: optuna.Trial) -> float:
        """Optuna calls this once per trial; it returns the validation Crossentropy to minimize."""

        fit_params = _build_tunable_fit_params(trial=trial, fit_params=base_fit_params)
        _validate_trial_params(model_name=selected_model, fit_params=fit_params)
        fold_losses: list[float] = []
        splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=int(model_cfg.get("random_state", 42)),
        )
        for train_idx, val_idx in splitter.split(x_train, y_train):
            x_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
            x_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]
            adapter = _fit_model(
                model_cfg=model_cfg,
                x_train=x_fold_train,
                y_train=y_fold_train,
                x_val=x_fold_val,
                y_val=y_fold_val,
                fit_params=fit_params,
            )
            x_val_eval = _scale_validation(
                x_val=x_fold_val,
                adapter=adapter,
                use_scaling=bool(fit_params.get("use_scaling", False)),
            )
            fold_loss = _crossentropy_on_validation(
                adapter=adapter,
                x_val=x_val_eval,
                y_val=y_fold_val,
                loss_weights=fit_params["loss_weights"],
                batch_size=int(fit_params.get("batch_size", 64)),
            )
            fold_losses.append(float(fold_loss))
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = float(np.mean(fold_losses))
        trial.set_user_attr("model", selected_model)
        trial.set_user_attr("cv_folds", int(args.cv_folds))
        return float(val_loss)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(model_cfg.get("random_state", 42))),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best_fit_params = dict(base_fit_params)
    for key, value in study.best_params.items():
        if key.startswith("loss_weight_"):
            continue
        best_fit_params[key] = value
    n_weights = len(base_fit_params["loss_weights"])
    best_fit_params["loss_weights"] = [
        float(study.best_params[f"loss_weight_{idx}"]) for idx in range(n_weights)
    ]

    # Retrain with the best hyperparameters using a small train-internal
    # validation split (test set remains untouched until final evaluation).
    x_final_train, x_final_val, y_final_train, y_final_val = train_test_split(
        x_train,
        y_train,
        stratify=y_train,
        test_size=0.1,
        random_state=int(model_cfg.get("random_state", 42)),
    )
    best_adapter = _fit_model(
        model_cfg=model_cfg,
        x_train=x_final_train,
        y_train=y_final_train,
        x_val=x_final_val,
        y_val=y_final_val,
        fit_params=best_fit_params,
    )
    x_test_eval = _scale_validation(
        x_val=x_test,
        adapter=best_adapter,
        use_scaling=bool(best_fit_params.get("use_scaling", False)),
    )
    test_crossentropy = _crossentropy_on_validation(
        adapter=best_adapter,
        x_val=x_test_eval,
        y_val=y_test,
        loss_weights=best_fit_params["loss_weights"],
        batch_size=int(best_fit_params.get("batch_size", 64)),
    )
    del best_adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"best_{selected_model}_hpo.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": selected_model,
                "target_column": target_column,
                "allowed_classes": allowed_classes,
                "cv_folds": int(args.cv_folds),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "best_cv_validation_crossentropy": float(study.best_value),
                "best_params": study.best_params,
                "best_fit_params": best_fit_params,
                "test_crossentropy_final_eval": float(test_crossentropy),
                "n_trials": len(study.trials),
                "n_pruned": int(
                    sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
                ),
            },
            f,
            indent=2,
        )

    print(f"Model optimized: {selected_model}")
    print(f"Best mean CV CrossEntropy: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Final test CrossEntropy (single holdout eval): {test_crossentropy:.6f}")
    print(f"Saved HPO summary to: {results_path}")


if __name__ == "__main__":
    main()
