import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_MODULES = ("pipelines.train_pipeline", "pipelines.backtest_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest the equity model")
    parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    for module in PIPELINE_MODULES:
        subprocess.run([sys.executable, "-m", module], cwd=project_root, check=True)


if __name__ == "__main__":
    main()
