# Repository Guidelines

## Project Structure & Module Organization
- `modules/` — core logic: data ingestion, feature engineering, model training,
  evaluation helpers, and the Backtrader strategy wrapper.
- `pipelines/` — runnable scripts (`train_pipeline.py`, `backtest_pipeline.py`,
  `evaluate_model.py`, `full_pipeline.py`).
- `tests/` — pytest suite (fast unit tests by default, `slow` mark for
  backtesting smoke test).
- `notebooks/` — exploratory work; keep saved outputs trimmed.
- `models/` — persisted artifacts (`models/gradient_boost_model.joblib`, HF
  placeholders).
- `data/` — raw/processed/altdata directories referenced in `config.yaml`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Train (defaults: AAPL, ~2y intraday window): `python pipelines/train_pipeline.py`
- Backtest recent data: `python pipelines/backtest_pipeline.py`
- Evaluate multiple tickers / export metrics: `python pipelines/evaluate_model.py`
- Docker (optional): `docker build -t pursuit-alpha .` then `docker compose up`

## Coding Style & Naming Conventions
- Python 3.10+, PEP8, 4‑space indents, max line length ~100.
- `snake_case` for modules/functions/vars; `CapWords` for classes.
- Keep functions pure; isolate I/O in pipeline/CLI layers.
- Docstrings for public functions; type hints encouraged.
- Linters/formatters are not enforced; `ruff .` and `black .` are welcome.

## Testing Guidelines
- Framework: `pytest` (add to dev env if needed).
- Place tests under `tests/`; name `test_<module>.py` and focus on deterministic units (feature calcs, label creation, strategy thresholds).
- Run: `pytest -q`; add a smoke run snippet in PRs (e.g., `python pipelines/backtest_pipeline.py` output line).

## Commit & Pull Request Guidelines
- History shows imperative messages (e.g., “Add”, “Fix”). Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `exp:` (experiments), `data:` (artifacts).
- Scope examples: `feat(training): add ROC AUC report`, `fix(backtest): handle non‑DatetimeIndex`.
- PRs must include: purpose, approach, commands run, sample outputs (paths like
  `models/...`, `metrics/...`), and risks.

## Security & Configuration Tips
- Do not commit secrets or large datasets; use `.env` (see `.env_template`).
- Update `config.yaml` or introduce environment-specific overrides when moving
  between Colab, local dev, and Docker (avoid hard-coding `/content/...`).
- Pin new deps in `requirements.txt`; note GPU needs for `torch/transformers` if applicable.
- Network calls (e.g., `yfinance`, HF models) may fail in CI; guard with retries
  or flags. Intraday Yahoo data is capped at ~730 days, so handle empty frames
  defensively.
