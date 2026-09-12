"""Probe selected worker environments without loading frameworks into Lab."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .workspace import Workspace, read_json


PACKAGES = ("btl_lab", "btl_train", "btl_rl", "btl_kernels", "btl_measure")


def worker_environment(*packages: str) -> dict:
    roots = [str(Path(importlib.import_module(package).__file__).resolve().parent.parent)
             for package in packages]
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(PYTHONPATH=os.pathsep.join(dict.fromkeys(roots)), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
               UNSLOTH_DISABLE_STATISTICS="1", TOKENIZERS_PARALLELISM="false")
    return env


def interpreter_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        found = shutil.which(str(value))
        if found:
            path = Path(found)
    # Keep the venv entry point. Resolving its symlink would select base Python.
    path = path.absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python interpreter is missing or not executable: {value}")
    return path


def probe_advance(python: str | Path, model: Path | None = None) -> dict:
    command = [str(interpreter_path(python)), "-m", "btl_rl.runtime"]
    if model is not None:
        command += ["--model", str(model.expanduser().absolute())]
    try:
        proc = subprocess.run(command, env=worker_environment("btl_rl"), capture_output=True,
                              text=True, timeout=30)
        report = json.loads(proc.stdout)
        if proc.returncode not in {0, 2} or not isinstance(report.get("ready_for_worker_start"), bool):
            raise ValueError("Runtime probe returned an invalid result")
        report["python"] = command[0]
        return report
    except (subprocess.TimeoutExpired, ValueError) as error:
        return {"ready_for_worker_start": False, "blockers": [f"Runtime probe failed: {error}"],
                "python": command[0]}


def diagnose(args) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package.replace("_", "-")] = importlib.metadata.version(package.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            versions[package.replace("_", "-")] = "source-checkout"
    result = {"engine": args.engine, "python": sys.executable, "packages": versions,
              "ready": True, "blockers": [], "external_spend_usd": 0}
    if args.engine == "control":
        try:
            workspace = Workspace(args.workspace)
            workspace.projects()
            workspace.recipes()
            result["workspace"] = str(workspace.root)
        except (ValueError, OSError, KeyError) as error:
            result.update(ready=False, blockers=[str(error)])
    elif args.engine == "advance":
        runtime = probe_advance(args.python, args.model)
        result.update(ready=runtime["ready_for_worker_start"], blockers=runtime["blockers"], runtime=runtime)
    elif args.engine == "adapt":
        if not args.recipe:
            result.update(ready=False, blockers=["Pass --recipe with the pinned Adapt JSON configuration"])
        else:
            from btl_train.finetune_data import validate_config
            config = read_json(args.recipe.expanduser().absolute())
            validate_config(config)
            proc = subprocess.run([str(interpreter_path(args.python)), "-m", "btl_lab.runtime_probe"],
                                  input=json.dumps(config), env=worker_environment("btl_lab", "btl_train"),
                                  text=True, capture_output=True, timeout=30)
            try:
                runtime = json.loads(proc.stdout)
            except ValueError:
                raise ValueError("Adapt runtime probe failed: " + proc.stderr[-1000:]) from None
            result.update(ready=proc.returncode == 0 and runtime["ready_for_worker_start"],
                          blockers=runtime["blockers"], runtime=runtime)
    elif args.engine == "prime-rl":
        from .planner import plan
        workspace = Workspace(args.workspace)
        resolved = plan(workspace, "prime-rl-reverse-text", args.checkout)
        result.update(ready=False, blockers=resolved["blockers"], plan=resolved)
    result["scope"] = "Read-only setup diagnosis; no training or capability qualification"
    return result
