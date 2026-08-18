

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

## Suggested storage locations

- Place original input files in `data/raw/`.
- Write cleaned/derived datasets to `data/processed/`.
- Keep reusable code in `src/`.
- Keep test files in `tests/`.


