import re
import subprocess
from pathlib import Path


def test_documented_make_targets_exist():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text()
    documented_targets = set(re.findall(r"\bmake ([a-z][a-z0-9-]*)", readme))

    assert documented_targets >= {"backtest", "full", "scaffold", "train", "verify"}
    for target in sorted(documented_targets):
        result = subprocess.run(
            ["make", "--dry-run", target],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_pipeline_commands_use_module_execution():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text()
    makefile = (project_root / "Makefile").read_text()
    full_pipeline = (project_root / "pipelines" / "full_pipeline.py").read_text()

    file_path_invocation = re.compile(r"(?:python|\$\(PYTHON\))\s+pipelines/[^\s]+\.py")
    assert not file_path_invocation.search(readme)
    assert not file_path_invocation.search(makefile)
    assert '"pipelines/train_pipeline.py"' not in full_pipeline
    assert '"pipelines/backtest_pipeline.py"' not in full_pipeline
