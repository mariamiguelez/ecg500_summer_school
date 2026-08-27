from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    if "target" not in data.columns:
        raise ValueError(f"{path} must contain the original 'target' column.")
    feature_columns = [column for column in data.columns if column.startswith("x_")]
    if not feature_columns:
        raise ValueError(f"{path} contains no x_ feature columns.")
    return data["target"].to_numpy(dtype=int), data[feature_columns].to_numpy(dtype=float)


def _plot_mean_std(labels: np.ndarray, values: np.ndarray, output_path: Path) -> None:
    classes = np.sort(np.unique(labels))
    figure, axes = plt.subplots(
        len(classes), 1, figsize=(12, max(4, len(classes) * 2.8)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for axis, target_class in zip(axes, classes):
        class_values = values[labels == target_class]
        mean = class_values.mean(axis=0)
        std = class_values.std(axis=0)
        time_points = np.arange(values.shape[1])
        axis.plot(time_points, mean, color="tab:blue", linewidth=1.8, label="Mean")
        axis.fill_between(
            time_points,
            mean - std,
            mean + std,
            color="tab:blue",
            alpha=0.22,
            label="Mean +/- 1 standard deviation",
        )
        axis.set_ylabel(f"Class {target_class}")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time point")
    figure.suptitle("Full ECG5000 dataset: mean and standard deviation by class", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _calculate_characteristics(labels: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    time_points = np.arange(values.shape[1])
    rows = []
    for target_class in np.sort(np.unique(labels)):
        class_values = values[labels == target_class]
        peak_indices = np.argmax(class_values, axis=1)
        valley_indices = np.argmin(class_values, axis=1)
        series_means = class_values.mean(axis=1)
        series_stds = class_values.std(axis=1)
        first_25_sum = class_values[:, :25].sum(axis=1)
        last_25_sum = class_values[:, -25:].sum(axis=1)
        sum_ratio = np.divide(
            first_25_sum,
            last_25_sum,
            out=np.full_like(first_25_sum, np.nan),
            where=last_25_sum != 0,
        )
        centered_values = class_values - series_means[:, None]
        autocorrelation = {}
        for lag in (1, 5, 10):
            if lag >= class_values.shape[1]:
                autocorrelation[f"autocorrelation_lag_{lag}"] = np.full(
                    len(class_values), np.nan
                )
                continue
            numerator = np.sum(
                centered_values[:, :-lag] * centered_values[:, lag:], axis=1
            )
            denominator = np.sum(centered_values**2, axis=1)
            autocorrelation[f"autocorrelation_lag_{lag}"] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=float),
                where=denominator != 0,
            )
        rows.append(
            pd.DataFrame(
                {
                    "target": target_class,
                    #"series_mean": series_means,
                    #"series_std": series_stds,
                    #"coefficient_of_variation": np.divide(
                    #    series_stds,
                    #   series_means,
                    #    out=np.full_like(series_stds, np.nan),
                    #    where=series_means != 0,
                    #),
                    #"minimum": class_values.min(axis=1),
                    #"maximum": class_values.max(axis=1),
                    "amplitude_range": class_values.ptp(axis=1),
                    #"peak_value": class_values.max(axis=1),
                    "peak_time_point": time_points[peak_indices],
                    "valley_time_point": time_points[valley_indices],
                    "first_25_sum": first_25_sum,
                    "last_25_sum": last_25_sum,
                    "sum_ratio": sum_ratio,
                    "area_under_curve": np.trapz(class_values, axis=1)
                    #**autocorrelation,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _plot_autocorrelation(
    labels: np.ndarray, values: np.ndarray, output_path: Path
) -> None:
    classes = np.sort(np.unique(labels))
    max_lag = min(50, values.shape[1] - 1)
    lags = np.arange(max_lag + 1)
    figure, axis = plt.subplots(figsize=(12, 6))

    for target_class in classes:
        class_values = values[labels == target_class]
        centered_values = class_values - class_values.mean(axis=1, keepdims=True)
        denominator = np.sum(centered_values**2, axis=1)
        correlations = np.full((len(class_values), len(lags)), np.nan)
        correlations[:, 0] = 1.0
        for lag in lags[1:]:
            numerator = np.sum(
                centered_values[:, :-lag] * centered_values[:, lag:], axis=1
            )
            correlations[:, lag] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=float),
                where=denominator != 0,
            )
        mean = np.nanmean(correlations, axis=0)
        std = np.nanstd(correlations, axis=0)
        axis.plot(lags, mean, linewidth=1.8, label=f"Class {target_class}")
        axis.fill_between(lags, mean - std, mean + std, alpha=0.12)

    axis.set(
        title="Full ECG5000 dataset: autocorrelation by class",
        xlabel="Lag",
        ylabel="Autocorrelation",
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_characteristics(characteristics: pd.DataFrame, output_path: Path) -> None:
    feature_columns = [
        "peak_time_point",
        "valley_time_point",
        "amplitude_range",
        "area_under_curve",
        "first_25_sum",
        "last_25_sum",
        "sum_ratio"
    ]
    classes = np.sort(characteristics["target"].unique())
    figure, axes = plt.subplots(1, len(feature_columns), figsize=(18, 5))

    for axis, feature in zip(axes, feature_columns):
        distributions = [
            characteristics.loc[
                characteristics["target"] == target_class, feature
            ].dropna()
            for target_class in classes
        ]
        axis.boxplot(distributions, showfliers=False)
        axis.set_xticks(np.arange(1, len(classes) + 1))
        axis.set_xticklabels(classes)
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Class")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Full ECG5000 dataset: time-series characteristics by class", y=1.02)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot full ECG5000 time-series characteristics by class."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train_structured.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/test_structured.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/exploration"),
    )
    args = parser.parse_args()

    train_labels, train_values = _load_data(args.train)
    test_labels, test_values = _load_data(args.test)
    labels = np.concatenate([train_labels, test_labels])
    values = np.concatenate([train_values, test_values])

    characteristics = _calculate_characteristics(labels, values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_mean_std(labels, values, args.output_dir / "full_dataset_mean_std_by_class.png")
    _plot_characteristics(
        characteristics,
        args.output_dir / "full_dataset_characteristics_by_class.png",
    )
    _plot_autocorrelation(
        labels,
        values,
        args.output_dir / "full_dataset_autocorrelation_by_class.png",
    )
    characteristics.to_csv(
        args.output_dir / "full_dataset_characteristics.csv", index=False
    )
    print(f"Analyzed {len(values)} time series across classes {sorted(np.unique(labels))}")
    print(f"Saved plots and characteristics to {args.output_dir}")


if __name__ == "__main__":
    main()
