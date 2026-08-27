from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


def _load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    if "target" not in data.columns:
        raise ValueError(f"{path} must contain the original 'target' column.")
    feature_columns = [column for column in data.columns if column.startswith("x_")]
    if not feature_columns:
        raise ValueError(f"{path} contains no x_ feature columns.")
    return data["target"].to_numpy(dtype=int), data[feature_columns].to_numpy(dtype=float)


def _transform(values: np.ndarray, scales: np.ndarray, wavelet: str) -> np.ndarray:
    coefficients, _ = pywt.cwt(values, scales, wavelet)
    return coefficients


def _plot_mean_scalograms(
    labels: np.ndarray,
    values: np.ndarray,
    scales: np.ndarray,
    wavelet: str,
    output_path: Path,
) -> None:
    classes = np.sort(np.unique(labels))
    figure, axes = plt.subplots(
        len(classes), 1, figsize=(12, max(5, len(classes) * 3.2)), squeeze=False
    )

    image = None
    for axis, target_class in zip(axes[:, 0], classes):
        class_mean = values[labels == target_class].mean(axis=0)
        coefficients = _transform(class_mean, scales, wavelet)
        image = axis.imshow(
            np.abs(coefficients),
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, values.shape[1] - 1, scales[0], scales[-1]],
        )
        axis.set_ylabel(f"Class {target_class}\nScale")
        axis.set_title("Wavelet magnitude of class-mean signal")

    axes[-1, 0].set_xlabel("Time point")
    figure.subplots_adjust(right=0.86, top=0.94, hspace=0.35)
    figure.colorbar(
        image,
        ax=axes[:, 0].tolist(),
        label="Coefficient magnitude",
        fraction=0.03,
        pad=0.04,
    )
    figure.suptitle(f"CWT of full ECG5000 dataset class means ({wavelet})", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_representatives(
    labels: np.ndarray,
    values: np.ndarray,
    scales: np.ndarray,
    wavelet: str,
    representatives_per_class: int,
    output_path: Path,
) -> None:
    classes = np.sort(np.unique(labels))
    representative_groups = [
        values[labels == target_class][:representatives_per_class]
        for target_class in classes
    ]
    n_representatives = sum(len(group) for group in representative_groups)
    figure, axes = plt.subplots(
        n_representatives,
        2,
        figsize=(14, max(5, n_representatives * 2.5)),
        squeeze=False,
    )

    image = None
    row = 0
    for target_class, group in zip(classes, representative_groups):
        for representative_number, representative in enumerate(group, start=1):
            coefficients = _transform(representative, scales, wavelet)
            axes[row, 0].plot(representative, linewidth=0.9)
            axes[row, 0].set_title(
                f"Class {target_class}: representative {representative_number}"
            )
            axes[row, 0].set_ylabel("Value")
            axes[row, 0].grid(alpha=0.25)
            image = axes[row, 1].imshow(
                np.abs(coefficients),
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[0, values.shape[1] - 1, scales[0], scales[-1]],
            )
            axes[row, 1].set_title(f"Class {target_class}: wavelet magnitude")
            axes[row, 1].set_ylabel("Scale")
            axes[row, 1].grid(alpha=0.15)
            row += 1

    axes[-1, 0].set_xlabel("Time point")
    axes[-1, 1].set_xlabel("Time point")
    figure.colorbar(image, ax=axes[:, 1].tolist(), label="Coefficient magnitude")
    figure.suptitle(f"Representative ECG5000 wavelet transforms ({wavelet})", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_energy_summary(
    labels: np.ndarray,
    values: np.ndarray,
    scales: np.ndarray,
    wavelet: str,
    output_path: Path,
) -> None:
    rows = []
    for target_class in np.sort(np.unique(labels)):
        class_values = values[labels == target_class]
        energies = []
        for series in class_values:
            coefficients = _transform(series, scales, wavelet)
            energies.append(np.mean(coefficients**2, axis=1))
        rows.append(
            pd.DataFrame(
                {
                    "target": target_class,
                    "scale": scales,
                    "mean_wavelet_energy": np.mean(energies, axis=0),
                    "std_wavelet_energy": np.std(energies, axis=0),
                }
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot continuous wavelet transforms for ECG5000."
    )
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/train_structured.csv")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("data/processed/test_structured.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/exploration")
    )
    parser.add_argument("--wavelet", default="morl", help="PyWavelets wavelet name.")
    parser.add_argument("--max-scale", type=int, default=64)
    parser.add_argument(
        "--representatives-per-class",
        type=int,
        default=5,
        help="Number of representative series to plot for each class.",
    )
    args = parser.parse_args()

    if args.max_scale < 2:
        raise ValueError("--max-scale must be at least 2.")
    if args.representatives_per_class < 1:
        raise ValueError("--representatives-per-class must be at least 1.")

    train_labels, train_values = _load_data(args.train)
    test_labels, test_values = _load_data(args.test)
    labels = np.concatenate([train_labels, test_labels])
    values = np.concatenate([train_values, test_values])
    scales = np.arange(1, args.max_scale + 1)

    _plot_mean_scalograms(
        labels, values, scales, args.wavelet,
        args.output_dir / "full_dataset_wavelet_class_means.png",
    )
    _plot_representatives(
        labels, values, scales, args.wavelet,
        args.representatives_per_class,
        args.output_dir / "full_dataset_wavelet_representatives.png",
    )
    _save_energy_summary(
        labels, values, scales, args.wavelet,
        args.output_dir / "full_dataset_wavelet_energy.csv",
    )
    print(f"Analyzed {len(values)} time series across classes {sorted(np.unique(labels))}")
    print(f"Saved wavelet outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
