from pathlib import Path

ROOT = Path(__file__).parent.resolve()

DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "data" / "altdata",
    ROOT / "models" / "trained",
    ROOT / "models" / "hf" / "finbert",
    ROOT / "models" / "hf" / "finbert-tone",
    ROOT / "modules",
    ROOT / "pipelines",
    ROOT / "notebooks",
    ROOT / "logs",
]

FILES = {
    ROOT / "README.md": "# Quant Backtest ML\n\nMedallion-style quant project.\n",
    ROOT / "requirements.txt": "\n".join([
        "pandas",
        "numpy",
        "yfinance",
        "scikit-learn",
        "backtrader",
        "ta",
        "transformers",
        "torch",
        "joblib",
    ]) + "\n",
    ROOT / ".env_template": """# Copy to .env and fill your keys
HUGGINGFACE_TOKEN=
TIINGO_API_KEY=
POLYGON_API_KEY=
NEWSAPI_KEY=
""",
    ROOT / "config.yaml": (
        "project_root: \"{root}\"\n"
        "data:\n"
        "  raw: \"data/raw\"\n"
        "  processed: \"data/processed\"\n"
        "  alt: \"data/altdata\"\n"
        "models:\n"
        "  hf:\n"
        "    finbert: \"models/hf/finbert\"\n"
        "    tone: \"models/hf/finbert-tone\"\n"
        "  trained: \"models/trained\"\n"
    ),
    ROOT / "Dockerfile": """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [\"python\", \"main.py\"]
""",
}


def write_file(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content.format(root=ROOT.as_posix()))
        print(f"Created {path.relative_to(ROOT)}")
    else:
        print(f"Exists  {path.relative_to(ROOT)}")


def main() -> None:
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Ensured {d.relative_to(ROOT)}")
    for p, content in FILES.items():
        write_file(p, content)


if __name__ == "__main__":
    main()
