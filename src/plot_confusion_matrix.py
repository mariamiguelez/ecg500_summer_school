from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _plot_matrix(
    matrix: np.ndarray,
    classes: np.ndarray,
    axis: plt.Axes,
    title: str,
    value_format: str,
) -> None:
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], value_format),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the ECG5000 test confusion matrix.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/test_predictions.csv"),
        help="CSV containing y_true and y_pred columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/confusion_matrix_random_forest.png"),
        help="Path for the generated plot.",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required_columns = {"y_true", "y_pred"}
    if not required_columns.issubset(predictions.columns):
        raise ValueError(f"{args.predictions} must contain y_true and y_pred columns.")

    classes = np.sort(predictions["y_true"].unique())
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=classes)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_matrix(matrix, classes, axes[0], "Confusion matrix", "d")
    _plot_matrix(normalized_matrix, classes, axes[1], "Normalized by true class", ".2f")
    figure.suptitle("ECG5000 test predictions", y=1.02)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrix to {args.output}")


if __name__ == "__main__":
    main()