# ECG 5000

## UV setup

Requirements:
- Python 3.11+
- `uv`

Create the environment and install base dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --extra dev
```

Run commands inside the environment:

```bash
uv run <command>
```

Preprocess ECG5000 raw files into structured train/test files:

```bash
uv run python src/data_processing/preprocess.py
```

## Suggested storage locations

- Place original input files in `data/raw/`.
- Write cleaned/derived datasets to `data/processed/`.
- Keep reusable code in `src/`.

## Configuration guide

Project configuration lives in `configs/default.yaml`.
This file controls data paths, model selection, training behavior, and per-model hyperparameters.

### `paths`

- `raw_data_dir`: folder containing original ECG5000 text files.
- `raw_train_file` / `raw_test_file`: raw dataset filenames.
- `processed_data_dir`: destination for preprocessed structured datasets.
- `models_dir`: where trained model artifacts are saved.
- `reports_dir`: where metrics, predictions, and plots are saved.

### `models`

Each model entry defines:

- `module`: Python module path used for dynamic import.
- `fit_function`: training function called from that module.
- `artifact_name`: output filename for the trained artifact.
- `random_state`: seed used inside model fitting.
- `fit_params`: model-specific training/HPO parameters.

Current model keys include:

- `random_forest`
- `logistic_regression`
- `lstm`
- `transformer` (encoder-based model)
- `xlstm`

### `fit_params` (per model)

`fit_params` are passed directly to each model's `fit_function`, so keys must match that function signature.
For deep models (`lstm`, `transformer`, `xlstm`) typical keys are:

- `epochs`
- `batch_size`
- `learning_rate`
- `use_scaling`
- architecture parameters (`hidden_size`, `n_layers`, `d_model`, `n_heads`, `n_blocks`, etc.)
- `loss_weights`: class weights for `CrossEntropyLoss`

Notes on `loss_weights`:

- Length must match the number of classes used in the run.
- Order must match the internal class order of the model (sorted class labels).
- These weights are also tuned during HPO.

### `training`

- `selected_model`: active model key under `models` used by `src/train.py`.
- `target_column`: label column from processed data (`target` or `target_binary`).
  - target : Class 1 (Healthy), Classes 2-5 (disease)
  - target_binary: Class 1 (Healthy), Class 0 (disese)
- `allowed_classes`: optional class filter (for reduced-class experiments).
  - _e.g. 1,2,3 only
- `validation_from_test_fraction`: fraction of test set used as validation in `train.py`.
  - Set to `0.0` to keep full test set for testing.
- `split_random_state`: split reproducibility seed.
- `plot_latent_space`: enables latent-space PCA plotting when supported.

## How configuration is used

- `src/data_processing/preprocess.py` uses `paths` to locate raw input and write structured output.
- `src/train.py` uses:
  - `training.selected_model` to choose the model config.
  - `training.target_column` and `training.allowed_classes` to shape labels/classes.
  - `models.<selected>.fit_params` as direct fit arguments.
- `src/HPO.py` (for `transformer`/`xlstm`) uses model `fit_params` as the tuning space and optimizes weighted validation cross-entropy.

## Typical workflow

1. Set paths and model/training options in `configs/default.yaml`.
2. Preprocess:
   ```bash
   uv run python src/data_processing/preprocess.py
   ```
3. Train:
   ```bash
   uv run python src/train.py
   ```
4. Optional HPO:
   ```bash
   uv run python src/HPO.py --model transformer --n-trials 20 --cv-folds 5
   ```
__Note: The HPO is only fitted to the transformer and the xLSTM__

## Other
Additional plots can be found in `scr/plots`.
These scripts are used for 
- confusion matrix creation
- plot data characteristics
- plot special data transformations
- plot samples grouped by class
