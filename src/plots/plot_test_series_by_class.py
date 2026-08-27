from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_page(
    series: np.ndarray,
    target_class: int,
    page_number: int,
    total_pages: int,
    first_index: int,
    output_path: Path,
) -> None:
    n_series = len(series)
    n_columns = 4
    n_rows = int(np.ceil(n_series / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, max(3, n_rows * 2.5)),
        squeeze=False,
    )
    axes = axes.ravel()

    for offset, values in enumerate(series):
        axes[offset].plot(values, linewidth=0.8)
        axes[offset].set_title(f"Series {first_index + offset + 1}")
        axes[offset].set_xlabel("Time point")
        axes[offset].set_ylabel("Value")
        axes[offset].grid(alpha=0.25)

    for axis in axes[n_series:]:
        axis.remove()

    figure.suptitle(
        f"Test time series: actual class {target_class} "
        f"(page {page_number}/{total_pages})",
        y=1.0,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all ECG5000 test time series grouped by original class."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/test_structured.csv"),
        help="Processed test data containing target and x_ feature columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/test_series_by_class"),
        help="Directory for the generated class-specific figures.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=20,
        help="Number of individual time series per image.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Original target classes to plot. Defaults to all classes.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=100,
        help="Maximum number of series to plot for each selected class.",
    )
    args = parser.parse_args()

    if args.per_page < 1:
        raise ValueError("--per-page must be at least 1.")
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError("--max-per-class must be at least 1.")

    data = pd.read_csv(args.data)
    if "target" not in data.columns:
        raise ValueError(f"{args.data} must contain the original 'target' column.")
    feature_columns = [column for column in data.columns if column.startswith("x_")]
    if not feature_columns:
        raise ValueError(f"{args.data} contains no x_ feature columns.")

    series = data[feature_columns].to_numpy(dtype=float)
    targets = data["target"].to_numpy(dtype=int)

    available_classes = np.sort(np.unique(targets))
    target_classes = available_classes if args.classes is None else np.array(args.classes)
    unknown_classes = np.setdiff1d(target_classes, available_classes)
    if len(unknown_classes) > 0:
        raise ValueError(f"Requested classes not found in data: {unknown_classes.tolist()}")

    for target_class in target_classes:
        class_series = series[targets == target_class]
        if args.max_per_class is not None:
            class_series = class_series[: args.max_per_class]
        total_pages = int(np.ceil(len(class_series) / args.per_page))
        for page_number, start in enumerate(
            range(0, len(class_series), args.per_page), start=1
        ):
            page_series = class_series[start : start + args.per_page]
            output_path = args.output_dir / (
                f"class_{target_class}_page_{page_number:03d}.png"
            )
            _plot_page(
                series=page_series,
                target_class=int(target_class),
                page_number=page_number,
                total_pages=total_pages,
                first_index=start,
                output_path=output_path,
            )
        print(
            f"Saved {len(class_series)} series for class {target_class} "
            f"in {total_pages} page(s)"
        )


if __name__ == "__main__":
    main()
