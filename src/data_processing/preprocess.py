from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config


def _resolve_raw_files(
    project_root: Path,
    raw_data_dir: str,
    raw_train_file: str,
    raw_test_file: str,
) -> tuple[Path, Path]:
    base_dir = project_root / raw_data_dir
    train_path = base_dir / raw_train_file
    test_path = base_dir / raw_test_file

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Could not find configured raw files. "
            f"Expected train={train_path} and test={test_path}."
        )

    return train_path, test_path


def _to_structured_array(raw_array: np.ndarray) -> tuple[np.ndarray, str]:
    if raw_array.ndim != 2 or raw_array.shape[1] < 2:
        raise ValueError("Expected a 2D array with at least one label column and one feature column.")

    labels = raw_array[:, 0].astype(int).reshape(-1, 1)
    features = raw_array[:, 1:]
    structured = np.hstack([labels, features])

    n_features = features.shape[1]
    header = ",".join(["target", *[f"x_{i}" for i in range(n_features)]])
    return structured, header


def _save_outputs(data: np.ndarray, header: str, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    txt_path = output_prefix.with_suffix(".txt")

    n_columns = data.shape[1]
    fmt = ["%d", *["%.8f" for _ in range(n_columns - 1)]]

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=fmt,
    )
    np.savetxt(
        txt_path,
        data,
        delimiter="\t",
        header=header.replace(",", "\t"),
        comments="",
        fmt=fmt,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_config(project_root / "configs" / "default.yaml")
    try:
        paths = cfg["paths"]
        raw_data_dir = paths["raw_data_dir"]
        raw_train_file = paths["raw_train_file"]
        raw_test_file = paths["raw_test_file"]
        processed_data_dir = paths["processed_data_dir"]
    except KeyError as exc:
        raise KeyError(
            "Missing required config key in configs/default.yaml under 'paths': "
            "raw_data_dir, raw_train_file, raw_test_file, processed_data_dir"
        ) from exc

    train_raw_path, test_raw_path = _resolve_raw_files(
        project_root=project_root,
        raw_data_dir=raw_data_dir,
        raw_train_file=raw_train_file,
        raw_test_file=raw_test_file,
    )
    processed_dir = project_root / processed_data_dir

    train_raw = np.loadtxt(train_raw_path)
    test_raw = np.loadtxt(test_raw_path)

    train_structured, header = _to_structured_array(train_raw)
    test_structured, _ = _to_structured_array(test_raw)

    _save_outputs(train_structured, header, processed_dir / "train_structured")
    _save_outputs(test_structured, header, processed_dir / "test_structured")

    print(f"Loaded raw train data from: {train_raw_path}")
    print(f"Loaded raw test data from: {test_raw_path}")
    print(f"Saved processed files to: {processed_dir}")


if __name__ == "__main__":
    main()
