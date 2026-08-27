from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config


def _load_processed_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
	data = np.loadtxt(path, delimiter=",", skiprows=1)
	if data.ndim != 2 or data.shape[1] < 2:
		raise ValueError(f"Expected labels and time-series values in {path}.")
	return data[:, 0].astype(int), data[:, 1:]


def _plot_class_distribution(
	train_labels: np.ndarray,
	test_labels: np.ndarray,
	output_path: Path,
) -> None:
	classes = np.unique(np.concatenate([train_labels, test_labels]))
	train_counts = np.array([(train_labels == label).sum() for label in classes])
	test_counts = np.array([(test_labels == label).sum() for label in classes])

	train_ditributions = train_counts / train_counts.sum()
	test_ditributions = test_counts / test_counts.sum()
	print(train_ditributions)
	print(test_ditributions)

	positions = np.arange(len(classes))
	width = 0.38
	figure, axis = plt.subplots(figsize=(8, 5))
	axis.bar(positions - width / 2, train_counts, width, label="Training")
	axis.bar(positions + width / 2, test_counts, width, label="Test")
	axis.set(
		title="ECG5000 class distribution",
		xlabel="Class",
		ylabel="Number of series",
		xticks=positions,
		xticklabels=classes,
	)
	axis.legend()
	figure.tight_layout()
	figure.savefig(output_path, dpi=160)
	plt.close(figure)


def _plot_time_series(
	train_labels: np.ndarray,
	train_values: np.ndarray,
	test_labels: np.ndarray,
	test_values: np.ndarray,
	output_path: Path,
) -> None:
	classes = np.unique(np.concatenate([train_labels, test_labels]))
	figure, axes = plt.subplots(2, len(classes), figsize=(16, 7), sharex=True, sharey=True)
	axes = np.atleast_2d(axes)

	for row, (split, labels, values) in enumerate(
		(("Training", train_labels, train_values), ("Test", test_labels, test_values))
	):
		for column, label in enumerate(classes):
			class_values = values[labels == label]
			axis = axes[row, column]
			axis.plot(class_values.T, color="tab:blue", alpha=0.04, linewidth=0.7)
			axis.plot(class_values.mean(axis=0), color="black", linewidth=1.8)
			axis.set_title(f"Class {label}")
			axis.grid(alpha=0.2)
			if column == 0:
				axis.set_ylabel(f"{split}\nvalue")
			if row == 1:
				axis.set_xlabel("Time point")

	figure.suptitle("ECG5000 time series by class", y=1.02)
	figure.tight_layout()
	figure.savefig(output_path, dpi=160, bbox_inches="tight")
	plt.close(figure)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot the processed ECG5000 datasets.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for generated plots (defaults to reports/exploration).",
	)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[1]
	config = load_config(project_root / "configs" / "default.yaml")
	processed_dir = project_root / config["paths"]["processed_data_dir"]
	output_dir = args.output_dir or project_root / "reports" / "exploration"
	output_dir.mkdir(parents=True, exist_ok=True)

	train_labels, train_values = _load_processed_data(processed_dir / "train_structured.csv")
	test_labels, test_values = _load_processed_data(processed_dir / "test_structured.csv")

	_plot_class_distribution(train_labels, test_labels, output_dir / "class_distribution.png")
	_plot_time_series(
		train_labels,
		train_values,
		test_labels,
		test_values,
		output_dir / "time_series_by_class.png",
	)
	print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
	main()
