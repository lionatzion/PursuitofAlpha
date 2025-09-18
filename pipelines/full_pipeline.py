import subprocess
import sys


def main() -> int:
    commands = (
        [sys.executable, "pipelines/train_pipeline.py"],
        [sys.executable, "pipelines/backtest_pipeline.py"],
    )
    for cmd in commands:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
