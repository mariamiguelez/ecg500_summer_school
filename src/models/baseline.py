from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_baseline_model(random_state: int = 42) -> RandomForestClassifier:
    """Random Forest baseline classifier"""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_baseline_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train and return the Random Forest baseline."""
    model = build_baseline_model(random_state=random_state)
    model.fit(x_train, y_train)
    return model
