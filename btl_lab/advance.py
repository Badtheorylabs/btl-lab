from __future__ import annotations

import json
import os
from pathlib import Path

from btl_train.advance_local import DEFAULT_GROUP_SIZE, DEFAULT_MAX_SECONDS, DEFAULT_STEPS, SEED, sha256
from btl_train.process import run_process

from .checks import artifact_record
from .store import Store
from .workspace import Workspace


def add_advance(sub):
    cli = sub.add_parser("advance", help="BTL Advance: policy improvement through reinforcement learning")
    actions = cli.add_subparsers(dest="action", required=True)
    local = actions.add_parser("local", help="Run the no-spend MLX contextual-bandit qualification")
    local.add_argument("--model", type=Path, required=True, help="Absolute local MLX model directory")
    local.add_argument("--python", type=Path, help="MLX environment interpreter")
    local.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    local.add_argument("--total-steps", type=int, help="Final step target shared across a split run and resume")
    local.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    local.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    local.add_argument("--out", type=Path, help="Optional output directory; defaults to Lab state")
    local.add_argument("--resume", type=Path)
    local.add_argument("--experiment")


def execute_advance(args, workspace: Workspace, store: Store) -> dict:
    if args.action != "local":
        raise ValueError("Unknown BTL Advance action")
    if not args.model.is_absolute() or not args.model.is_dir():
        raise ValueError("--model must be an existing absolute model directory")
    if args.steps <= 0 or args.group_size < 2 or args.max_seconds <= 0:
        raise ValueError("steps and max-seconds must be positive; group-size must be at least two")
    total_steps = getattr(args, "total_steps", None)
    if total_steps is not None and (total_steps < args.steps or total_steps <= 0):
        raise ValueError("total-steps must be at least steps")
    if args.resume and (not args.resume.is_absolute() or not args.resume.is_dir()):
        raise ValueError("--resume must be an existing absolute checkpoint directory")
    if args.experiment:
        from .research import Research
        experiment = Research(workspace, store).get(args.experiment)
        if experiment["project_id"] != "workspace/btl-train" or experiment["status"] == "decided":
            raise ValueError("Advance run must link to an open workspace/btl-train experiment")
    interpreter = args.python or Path(os.environ.get("BTL_MLX_PYTHON", str(workspace.root / ".mlxvenv/bin/python")))
    if not interpreter.is_file():
        raise ValueError(f"MLX interpreter is missing: {interpreter}")
    output = args.out
    if output is not None:
        output = output.resolve()
        if not output.is_relative_to(workspace.root):
            raise ValueError("--out must stay inside the workspace")
    # Store first so the durable run ID determines the output directory.
    plan = {"schema_version": 1, "project_id": "workspace/btl-train",
            "recipe_id": "advance:mlx-local-contextual-bandit-v1", "backend": "mlx-local",
            "engine": "BTL Advance", "model": str(args.model), "model_revision": args.model.name,
            "steps": args.steps, "total_steps": total_steps or args.steps,
            "group_size": args.group_size, "temperature": 2.0,
            "objective": "group-relative-policy-gradient", "seed": SEED,
            "experiment_id": args.experiment, "external_spend_usd": 0,
            "scope": "Local contextual-bandit RL execution; not agentic-RL or Tinfield capability evidence",
            "implementation_sha256": sha256(Path(__file__).resolve().parents[2] / "btl-train/btl_train/advance_local.py")}
    run_id = store.create(plan, kind="model-operation-advance-local")
    if args.out is None:
        output = workspace.state / "advance" / run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [str(interpreter), "-m", "btl_train.advance_local", "--model", str(args.model),
               "--out", str(output), "--steps", str(args.steps), "--group-size", str(args.group_size),
               "--max-seconds", str(args.max_seconds)]
    if total_steps is not None:
        command.extend(["--total-steps", str(total_steps)])
    if args.resume:
        command.extend(["--resume", str(args.resume)])
    env = os.environ.copy()
    env.update(PYTHONPATH=str(workspace.root / "platform/btl-train") + os.pathsep + env.get("PYTHONPATH", ""),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    store.transition(run_id, "running")
    try:
        process_log = output.parent / f"{run_id}.worker.log"
        process = run_process(command, workspace.root, process_log, args.max_seconds + 60, env)
        result = {"process": process, "worker": None, "external_spend_usd": 0,
                  "scope": plan["scope"], "evidence_class": "optimization"}
        worker_result = output / "result.json"
        if worker_result.is_file():
            result["worker"] = json.loads(worker_result.read_text())
        result["passed"] = process["exit_code"] == 0 and not process["timed_out"] and bool(result["worker"])
        receipt = output / "lab-result.json"
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        for file in sorted(output.iterdir()):
            if file.is_file():
                store.attach(run_id, artifact_record(workspace, str(file.relative_to(workspace.root)),
                                                     file.name, "local-optimization-only"))
        store.attach(run_id, artifact_record(workspace, str(process_log.relative_to(workspace.root)),
                                             process_log.name, "local-optimization-only"))
        store.transition(run_id, "passed" if result["passed"] else "failed", result)
    except KeyboardInterrupt:
        store.transition(run_id, "interrupted", {"error": "Interrupted by operator"})
        raise
    except Exception as error:
        store.transition(run_id, "failed", {"error": str(error), "type": type(error).__name__})
        raise
    return store.run(run_id)
