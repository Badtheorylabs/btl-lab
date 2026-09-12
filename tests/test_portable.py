import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from btl_lab.bootstrap import initialize
from btl_lab.cli import main
from btl_lab.doctor import interpreter_path, worker_environment
from btl_lab.store import Store
from btl_lab.workspace import Workspace


def test_initialize_once_preserves_existing_run_history(tmp_path, capsys):
    root = tmp_path / "my research workspace"
    assert main(["--json", "init", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["created"]
    args = ["--workspace", str(root), "--json"]
    assert main(args + ["run", "registry-audit"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "passed"
    assert main(args + ["runs", run["id"], "--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]
    state = root / ".btl/state/lab.sqlite3"
    before = state.read_bytes()
    assert not initialize(root)["created"]
    assert state.read_bytes() == before


def test_init_refuses_existing_files_and_does_not_modify_them(tmp_path):
    existing = tmp_path / "user-work.txt"
    existing.write_text("keep me")
    with pytest.raises(ValueError, match="existing files"):
        initialize(tmp_path)
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text() == "keep me"


def test_control_cli_loads_without_numpy_or_gpu_packages(tmp_path):
    env = worker_environment("btl_lab", "btl_rl", "btl_train", "btl_measure", "btl_kernels")
    command = [sys.executable, "-S", "-c",
               "from btl_lab.cli import main; import sys; "
               "assert not any(n in sys.modules for n in ('numpy','torch','mlx.core','unsloth')); "
               "main(['--help'])"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "init" in result.stdout and "doctor" in result.stdout


def test_init_and_measure_work_outside_source_tree(tmp_path):
    env = worker_environment("btl_lab", "btl_rl", "btl_train", "btl_measure", "btl_kernels")
    env.pop("BTL_WORKSPACE", None)
    root = tmp_path / "portable"
    launcher = [sys.executable, "-S", "-m", "btl_lab.cli"]
    created = subprocess.run(launcher + ["init", str(root)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    compared = subprocess.run(launcher + ["--json", "measure", "compare", "examples/baseline.json", "examples/candidate.json"],
                              cwd=root, env=env, capture_output=True, text=True)
    assert compared.returncode == 0, compared.stderr
    assert json.loads(compared.stdout)["result"]["paired_records"] == 2


def test_doctor_reports_missing_runtime_and_does_not_create_state(tmp_path, capsys):
    root = tmp_path / "empty"
    assert main(["--workspace", str(root), "--json", "doctor"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert "btl init" in report["blockers"][0]
    assert not root.exists()


def test_python_interpreter_keeps_virtual_environment_symlink(tmp_path):
    python = tmp_path / "venv-python"
    python.symlink_to(sys.executable)
    assert interpreter_path(python) == python


def test_advance_preflight_failure_is_recorded_without_spawning_model(tmp_path, monkeypatch):
    from btl_lab.advance import execute_advance
    root = tmp_path / "new"
    initialize(root)
    workspace = Workspace(root)
    model = root / "model"
    model.mkdir()
    monkeypatch.setattr("btl_lab.advance.probe_advance", lambda *a: {
        "ready_for_worker_start": False, "blockers": ["numpy is missing"]})
    monkeypatch.setattr("btl_lab.advance.run_process", lambda *a, **kw: pytest.fail("model must not launch"))
    store = Store(workspace.state / "lab.sqlite3")
    try:
        args = SimpleNamespace(action="local", model=model, python=Path(sys.executable), steps=3,
                               group_size=8, max_seconds=60, resume=None, out=None, experiment=None)
        run = execute_advance(args, workspace, store)
        assert run["status"] == "blocked" and run["artifacts"]
        assert [event["status"] for event in run["events"]] == ["planned", "blocked"]
    finally:
        store.db.close()
