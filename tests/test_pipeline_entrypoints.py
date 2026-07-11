import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "pipelines.train_pipeline",
        "pipelines.backtest_pipeline",
        "pipelines.full_pipeline",
    ],
)
def test_pipeline_module_entrypoint_help(module):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
