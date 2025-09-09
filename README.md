# Quant Backtest ML

Medallion-style quant project.

## Structure
- data/: raw, processed, altdata
- models/: trained, hf (FinBERT)
- modules/: Python code
- pipelines/: scripts
- notebooks/: Colab & Jupyter
- logs/: run outputs

## Tooling
- `project_scaffold.py` generates the directory tree and boilerplate files for a new project.
- `pipelines/full_pipeline.py` reads `config.yaml` and executes the listed pipeline scripts sequentially.
