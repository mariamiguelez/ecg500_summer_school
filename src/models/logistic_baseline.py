from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_model(random_state: int = 42) -> Pipeline:
    """Build a scaled, class-balanced Logistic Regression classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_logistic_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Pipeline:
    """Train and return the Logistic Regression baseline."""
    model = build_logistic_model(random_state=random_state)
    model.fit(x_train, y_train)
    return model