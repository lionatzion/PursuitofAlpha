# Quant Backtest ML

Medallion-style quant project.

## Structure
- data/: raw, processed, altdata
- models/: trained, hf (FinBERT)
- modules/: Python code
- pipelines/: scripts
- notebooks/: Colab & Jupyter
- logs/: run outputs

## Getting Started

Run the scaffold script to create the standard project structure and placeholder files:

```bash
python scaffold.py
```

Automation is available via the included Makefile:

```bash
make setup     # create folders and placeholder files
make train     # run the training pipeline
make backtest  # run the backtest pipeline
make full      # run train and backtest sequentially
```
