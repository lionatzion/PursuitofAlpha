import re
from pathlib import Path


def test_documented_make_targets_exist():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text()
    makefile = (project_root / "Makefile").read_text()
    documented_targets = set(re.findall(r"\bmake ([a-z][a-z0-9-]*)", readme))
    available_targets = set(re.findall(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE))

    assert documented_targets >= {"backtest", "full", "scaffold", "train", "verify"}
    assert documented_targets <= available_targets


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
