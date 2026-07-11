import subprocess
import sys
from pathlib import Path

import yaml


def test_scaffold_requires_explicit_empty_root(tmp_path):
    source_root = Path(__file__).resolve().parents[1]

    missing_root = subprocess.run(
        [sys.executable, str(source_root / "project_scaffold.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert missing_root.returncode != 0
    assert "--root" in missing_root.stderr

    occupied_root = tmp_path / "occupied"
    occupied_root.mkdir()
    sentinel = occupied_root / "keep.txt"
    sentinel.write_text("preserve me")
    occupied = subprocess.run(
        [sys.executable, str(source_root / "project_scaffold.py"), "--root", str(occupied_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert occupied.returncode != 0
    assert sentinel.read_text() == "preserve me"


def test_generated_scaffold_pipeline_contract(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "generated-project"

    scaffold = subprocess.run(
        [sys.executable, str(source_root / "project_scaffold.py"), "--root", str(project_root)],
        cwd=source_root,
        capture_output=True,
        text=True,
    )
    assert scaffold.returncode == 0, scaffold.stderr + "\n" + scaffold.stdout

    with (project_root / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    assert config["pipelines"], "generated config must list at least one pipeline"
    for module in config["pipelines"]:
        module_path = project_root.joinpath(*module.split(".")).with_suffix(".py")
        assert module_path.exists(), f"configured pipeline does not exist: {module}"

        command = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert command.returncode == 0, command.stderr + "\n" + command.stdout

    full_pipeline = subprocess.run(
        [sys.executable, "-m", "pipelines.full_pipeline"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert full_pipeline.returncode == 0, full_pipeline.stderr + "\n" + full_pipeline.stdout
    assert "Training pipeline placeholder" in full_pipeline.stdout
    assert "Backtest pipeline placeholder" in full_pipeline.stdout

    dockerfile = (project_root / "Dockerfile").read_text()
    assert 'CMD ["python", "-m", "pipelines.full_pipeline"]' in dockerfile
