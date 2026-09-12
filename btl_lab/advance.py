from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import btl_rl.advance_local as worker
from btl_rl.artifacts import latest_checkpoint, model_identity, verify_checkpoint
from btl_rl.process import run_process

from .checks import artifact_record, verify_artifacts
from .doctor import interpreter_path, probe_advance, worker_environment
from .store import Store
from .workspace import Workspace, inside


def add_advance(sub):
    cli = sub.add_parser("advance", help="BTL Advance: policy improvement through reinforcement learning")
    actions = cli.add_subparsers(dest="action", required=True)
    local = actions.add_parser("local", help="Run the bounded MLX recipe and compare evaluations")
    local.add_argument("--model", type=Path, required=True, help="Existing local Qwen3.5 model directory")
    local.add_argument("--python", type=Path, help="Interpreter of the MLX environment")
    local.add_argument("--steps", type=int, default=worker.DEFAULT_STEPS)
    local.add_argument("--total-steps", type=int, help="Final target shared by a split run and resume")
    local.add_argument("--group-size", type=int, default=worker.DEFAULT_GROUP_SIZE)
    local.add_argument("--max-seconds", type=int, default=worker.DEFAULT_MAX_SECONDS)
    local.add_argument("--out", type=Path)
    local.add_argument("--resume", type=Path)
    local.add_argument("--experiment")
    resume = actions.add_parser("resume", help="Continue a run from its latest complete checkpoint")
    resume.add_argument("id", help="Advance run ID")
    resume.add_argument("--python", type=Path)
    resume.add_argument("--steps", type=int)
    resume.add_argument("--max-seconds", type=int, default=worker.DEFAULT_MAX_SECONDS)


def run_directory(workspace: Workspace, run: dict) -> Path:
    output = run["plan"].get("output_override")
    return inside(workspace.root, output) if output else workspace.state / "advance" / run["id"]


def resume_advance(args, workspace, store):
    source = store.run(args.id)
    if source["kind"] != "model-operation-advance-local" or source["status"] not in {"passed", "failed", "interrupted"}:
        raise ValueError("Resume requires a completed or stopped Advance run")
    saved = source["plan"]
    if not saved.get("model_assets"):
        raise ValueError("This older run has no model manifest; review its model before explicitly resuming a checkpoint")
    if not verify_artifacts(workspace, source)["passed"]:
        raise ValueError("Recorded source artifacts changed; review them before resuming")
    checkpoint = latest_checkpoint(run_directory(workspace, source) / "worker")
    local = argparse.Namespace(action="local", model=Path(saved["model"]),
        python=args.python or Path(saved["python"]), steps=args.steps or saved["total_steps"],
        total_steps=saved["total_steps"], group_size=saved["group_size"], max_seconds=args.max_seconds,
        out=None, resume=checkpoint, experiment=saved.get("experiment_id"), source_run_id=source["id"],
        expected_model_identity=saved["model_assets"]["sha256"], expected_implementation=saved["implementation_sha256"])
    return execute_advance(local, workspace, store)


def execute_advance(args, workspace: Workspace, store: Store) -> dict:
    if args.action == "resume":
        return resume_advance(args, workspace, store)
    if args.action != "local":
        raise ValueError("Unknown BTL Advance action")
    model = args.model.expanduser().absolute()
    if not model.is_dir():
        raise ValueError("--model must be an existing local model directory")
    if args.steps <= 0 or args.group_size < 2 or args.max_seconds <= 0:
        raise ValueError("steps and max-seconds must be positive; group-size must be at least two")
    total_steps = getattr(args, "total_steps", None)
    total_steps = args.steps if total_steps is None else total_steps
    if total_steps <= 0 or total_steps < args.steps:
        raise ValueError("total-steps must be at least steps")
    resume = args.resume.expanduser().absolute() if args.resume else None
    if resume is not None and verify_checkpoint(resume)["step"] >= args.steps:
        raise ValueError("Checkpoint already reaches the requested final step")
    project = "workspace/btl-rl"
    if args.experiment:
        from .research import Research
        experiment = Research(workspace, store).get(args.experiment)
        if experiment["project_id"] not in {project, "workspace/btl-train"} or experiment["status"] == "decided":
            raise ValueError("Advance run must link to an open RL experiment")
        project = experiment["project_id"]
    configured = args.python or os.environ.get("BTL_MLX_PYTHON")
    fallback = workspace.root / ".mlxvenv/bin/python"
    configured = configured or (fallback if fallback.is_file() else sys.executable)
    try:
        interpreter = interpreter_path(configured)
    except ValueError as error:
        raise ValueError(f"MLX interpreter is unavailable: {error}") from error
    output = inside(workspace.root, str(args.out), exists=False) if args.out else None
    if output is not None and output.exists():
        raise ValueError("--out already exists; choose a new run directory")
    implementation = worker.sha256(Path(worker.__file__))
    if getattr(args, "expected_implementation", implementation) != implementation:
        raise ValueError("Advance implementation changed since the source run")
    runtime = probe_advance(interpreter, model)
    assets = model_identity(model) if runtime["ready_for_worker_start"] else None
    if assets and getattr(args, "expected_model_identity", assets["sha256"]) != assets["sha256"]:
        raise ValueError("Model assets changed since the source run")
    plan = {"schema_version": 1, "project_id": project, "recipe_id": "advance:mlx-local-contextual-bandit-v1",
            "backend": "mlx-local", "engine": "BTL Advance", "model": str(model), "model_assets": assets,
            "steps": args.steps, "total_steps": total_steps, "group_size": args.group_size,
            "temperature": worker.TEMPERATURE, "objective": "group-relative-policy-gradient", "seed": worker.SEED,
            "python": str(interpreter), "max_seconds": args.max_seconds, "runtime": runtime,
            "experiment_id": args.experiment, "parent_run_id": getattr(args, "source_run_id", None),
            "output_override": str(output.relative_to(workspace.root)) if output else None,
            "resume": str(resume) if resume else None, "external_spend_usd": 0,
            "scope": "Local contextual-bandit mechanics; no agentic-RL or Tinfield capability qualification",
            "implementation_sha256": implementation}
    run_id = store.create(plan, kind="model-operation-advance-local")
    output = output or workspace.state / "advance" / run_id
    try:
        output.mkdir(parents=True, exist_ok=False)
        runtime_path = output / "runtime.json"
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
        store.attach(run_id, artifact_record(workspace, str(runtime_path.relative_to(workspace.root)), "runtime", "preflight"))
        if not runtime["ready_for_worker_start"]:
            store.transition(run_id, "blocked", {"passed": False, "blockers": runtime["blockers"], "runtime": runtime})
            return store.run(run_id)
        worker_output = output / "worker"
        command = [str(interpreter), "-m", "btl_rl.advance_local", "--model", str(model),
                   "--out", str(worker_output), "--steps", str(args.steps), "--total-steps", str(total_steps),
                   "--group-size", str(args.group_size), "--max-seconds", str(args.max_seconds)]
        if resume:
            command += ["--resume", str(resume)]
        store.transition(run_id, "running")
        process = run_process(command, workspace.root, output / "worker.log", args.max_seconds,
                              worker_environment("btl_rl"), grace_seconds=5)
        result_file = worker_output / "result.json"
        result = {"process": process, "worker": None, "external_spend_usd": 0,
                  "scope": plan["scope"], "evidence_class": "optimization", "passed": False}
        if result_file.is_file():
            result["worker"] = json.loads(result_file.read_text())
            gates = result["worker"].get("gates", {})
            result["passed"] = process["exit_code"] == 0 and not process["timed_out"] and all(
                gates.get(key) is True for key in ("optimizer_update", "finite_gradients", "adapter_changed"))
            if result["passed"]:
                verify_checkpoint(worker_output / f"checkpoint-{args.steps}")
        (output / "lab-result.json").write_text(json.dumps(result, indent=2) + "\n")
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != runtime_path:
                store.attach(run_id, artifact_record(workspace, str(file.relative_to(workspace.root)),
                                                     str(file.relative_to(output)), "local-optimization-only"))
        store.transition(run_id, "passed" if result["passed"] else "failed", result)
    except (Exception, KeyboardInterrupt) as error:
        current = store.run(run_id)["status"]
        if current in {"planned", "running"}:
            status = "blocked" if current == "planned" else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            store.transition(run_id, status, {"error": str(error), "type": type(error).__name__})
        raise
    finished = store.run(run_id)
    if finished["status"] == "passed":
        from .measure import measure_advance
        evaluation = measure_advance(workspace, store, run_id)
        finished["evaluation_run_id"] = evaluation["id"]
        finished["evaluation_status"] = evaluation["status"]
    return finished
