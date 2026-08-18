import numpy as np
from pathlib import Path
from typing import Any
import yaml


def classification_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    score = 0
    return score

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
