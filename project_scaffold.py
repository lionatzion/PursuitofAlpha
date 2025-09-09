import argparse
import os
import textwrap
import yaml

# Default folder structure for a new quant project
FOLDERS = [
    "data/raw",
    "data/processed",
    "data/altdata",
    "models/trained",
    "models/hf/finbert",
    "models/hf/finbert-tone",
    "modules",
    "pipelines",
    "notebooks",
    "logs",
]

# Minimal Python requirements to get started
REQUIREMENTS = [
    "pandas",
    "numpy",
    "yfinance",
    "scikit-learn",
    "backtrader",
    "ta",
    "transformers",
    "torch",
    "joblib",
    "pyyaml",
]

ENV_TEMPLATE = textwrap.dedent(
    """\
    # Copy to .env and fill your keys
    HUGGINGFACE_TOKEN=
    TIINGO_API_KEY=
    POLYGON_API_KEY=
    NEWSAPI_KEY=
    """
)

DOCKERFILE_TEMPLATE = textwrap.dedent(
    """\
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    CMD ["python", "main.py"]
    """
)

README_TEMPLATE = textwrap.dedent(
    """\
    # Quant Backtest ML

    This folder was generated with ``project_scaffold.py``.

    ## Structure
    - `data/`: raw, processed, altdata
    - `models/`: trained models and HuggingFace assets
    - `modules/`: Python modules
    - `pipelines/`: end-to-end scripts
    - `notebooks/`: Jupyter or Colab notebooks
    - `logs/`: run logs
    """
)


def create_structure(root: str) -> None:
    """Create the directory tree and placeholder files."""
    for folder in FOLDERS:
        path = os.path.join(root, folder)
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(README_TEMPLATE)

    with open(os.path.join(root, "requirements.txt"), "w") as f:
        f.write("\n".join(REQUIREMENTS) + "\n")

    with open(os.path.join(root, ".env_template"), "w") as f:
        f.write(ENV_TEMPLATE)

    config = {
        "project_root": root,
        "data": {"raw": "data/raw", "processed": "data/processed", "alt": "data/altdata"},
        "models": {
            "hf": {"finbert": "models/hf/finbert", "tone": "models/hf/finbert-tone"},
            "trained": "models/trained",
        },
        "tickers": ["AAPL", "SPY"],
        "backtest": {
            "start_date": "2019-01-01",
            "end_date": "2024-12-31",
            "interval": "1h",
        },
        "pipelines": [
            "pipelines/train_pipeline.py",
            "pipelines/backtest_pipeline.py",
        ],
    }
    with open(os.path.join(root, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(os.path.join(root, "Dockerfile"), "w") as f:
        f.write(DOCKERFILE_TEMPLATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quant backtest project scaffold")
    parser.add_argument("--root", default=".", help="Directory where the scaffold will be created")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    create_structure(root)
    print(f"Scaffold created at {root}")


if __name__ == "__main__":
    main()
