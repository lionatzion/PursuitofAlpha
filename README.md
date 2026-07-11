# Pursuit of Alpha

AI-driven quantitative trading framework for model training and strategy backtesting.

## Quick Start

```bash
python -m pip install -r requirements-dev.txt

# Train the model, run a backtest, or execute both stages.
make train
make backtest
make full

# Run the same runtime contract used by CI.
make verify
```

The pipeline targets use Python module execution, so imports resolve from the project root
without modifying `PYTHONPATH` or `sys.path`.

## Create a New Project

Generate a runnable skeleton whose configuration only references modules created by the
scaffold:

```bash
make scaffold SCAFFOLD_ROOT=/path/to/new-project
cd /path/to/new-project
python -m pipelines.full_pipeline
```

## Verification Contract

`make verify` checks all documented pipeline entrypoints, executes the generated scaffold
contract, and runs the complete non-slow test suite. GitHub Actions invokes this same target.

## Project Structure

- `modules/`: data ingestion, feature engineering, model training, and backtesting logic
- `pipelines/`: executable train, backtest, and full-pipeline modules
- `tests/`: unit, smoke, scaffold, and documentation contract tests

## Disclaimer

This project is intended for research and education. Past performance does not indicate
future results. Validate strategies with paper trading before risking capital.
