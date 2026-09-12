from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import btl_train.finetune_worker as worker
from btl_train.finetune_data import audit, digest, fingerprint, validate_config
from btl_train.process import run_process

from .checks import artifact_record
from .doctor import worker_environment
from .workspace import inside, read_json


def add_finetune(sub):
    cli = sub.add_parser("adapt", aliases=["finetune"], help="BTL Adapt: supervised LoRA/QLoRA engine")
    commands = cli.add_subparsers(dest="action", required=True)
    for name in ("check", "run"):
        parser = commands.add_parser(name)
        parser.add_argument("file", help="Fine-tune recipe JSON inside the workspace")
        parser.add_argument("--python", default=sys.executable, help="Interpreter of the pinned GPU environment")
        if name == "check":
            parser.add_argument("--data-only", action="store_true")
            parser.add_argument("--weights", action="store_true")
        else:
            parser.add_argument("--execute", action="store_true", help="Explicitly execute on existing compute; does not provision it")
            parser.add_argument("--authorization-ref", required=True, help="Reference to the approved resource/budget decision")
            parser.add_argument("--resume", help="Workspace-relative checkpoint directory")
            parser.add_argument("--experiment")


def execute_finetune(args, workspace, store):
    config = read_json(inside(workspace.root, args.file))
    validate_config(config)
    workspace.project(config["project_id"])
    checked = audit(workspace.root, config, check_weights=getattr(args, "weights", False))
    if args.action == "run" and not args.execute:
        raise ValueError("Use --execute only after approving the existing compute and its budget")
    resume = str(inside(workspace.root, args.resume)) if getattr(args, "resume", None) else None
    plan = {"project_id": config["project_id"], "recipe_id": "finetune:unsloth-v1",
            "stage": "dataset-check" if args.action == "check" else "execute",
            "config": config, "config_sha256": fingerprint(config),
            "scope": "Experimental single-GPU supervised LoRA/QLoRA",
            "external_spend_usd": None, "resource_management": "external; no provider billing cap is enforced"}
    if getattr(args, "experiment", None):
        from .research import Research
        experiment = Research(workspace, store).get(args.experiment)
        if experiment["project_id"] != config["project_id"] or experiment["status"] == "decided":
            raise ValueError("Experiment must be open and belong to this project")
        plan["experiment_id"] = experiment["id"]
    if args.action == "run":
        if not args.authorization_ref.strip():
            raise ValueError("A resource authorization reference is required")
        plan["authorization_ref"] = args.authorization_ref
    run_id = store.create(plan, kind="model-operation-finetune")
    directory = workspace.state / "finetune" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    request = {"workspace": str(workspace.root), "config": config,
               "implementation_sources": {name: digest(Path(worker.__file__).parent / name)
                   for name in ("finetune_worker.py", "finetune_data.py", "process.py")}}
    if resume:
        request["resume"] = resume
    request_file = directory / "request.json"
    request_file.write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
               HF_DATASETS_OFFLINE="1", UNSLOTH_DISABLE_STATISTICS="1",
               PYTHONPATH=str(Path(worker.__file__).parent.parent) + os.pathsep + env.get("PYTHONPATH", ""))
    command = [args.python, "-m", "btl_train.finetune_worker", "--request", str(request_file)]
    result = {"passed": True, "stage": "dataset-check", "data": checked["summary"],
              "weights_verified": checked["weights_verified"], "evidence_class": "integrity",
              "scope": "Input structural/integrity checks; no training or capability result"}
    try:
        if not getattr(args, "data_only", False):
            proc = subprocess.run([args.python, "-m", "btl_lab.runtime_probe"], input=json.dumps(config),
                                  capture_output=True, text=True, cwd=workspace.root,
                                  env=worker_environment("btl_lab", "btl_train"), timeout=30)
            if proc.returncode not in {0, 2}:
                raise ValueError("Backend environment check failed: " + proc.stderr[-1500:])
            runtime = json.loads(proc.stdout)
            result["runtime"] = runtime
            if not runtime["ready_for_worker_start"]:
                result["passed"] = False
                blocked_receipt = directory / "result.json"
                blocked_receipt.write_text(json.dumps(result, indent=2) + "\n")
                for file in (request_file, blocked_receipt):
                    store.attach(run_id, artifact_record(workspace, str(file.relative_to(workspace.root)),
                                                         file.name, "preflight-only"))
                store.transition(run_id, "blocked", result)
                return store.run(run_id)
        store.transition(run_id, "running")
        if args.action == "run":
            process = run_process(command + ["--output", str(directory)], workspace.root,
                                  directory / "worker.log", config["wall_seconds"], env)
            worker_result = directory / "worker-result.json"
            result = {"process": process, "stage": "execute", "evidence_class": "optimization",
                      "passed": process["exit_code"] == 0 and not process["timed_out"] and worker_result.is_file()}
            if worker_result.is_file():
                result["worker"] = json.loads(worker_result.read_text())
                result["passed"] &= result["worker"]["status"] == "completed"
            result["release_ready"] = False
        path = directory / "result.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        for file in (request_file, path, directory / "adapter-manifest.json", directory / "worker.log"):
            if file.is_file():
                store.attach(run_id, artifact_record(workspace, str(file.relative_to(workspace.root)),
                                                     file.name, "diagnostic" if args.action == "check" else "optimization-only"))
        store.transition(run_id, "passed" if result["passed"] else "failed", result)
    except KeyboardInterrupt:
        status = store.run(run_id)["status"]
        store.transition(run_id, "interrupted" if status == "running" else "blocked", {"error": "Interrupted"})
        raise
    except Exception as error:
        status = store.run(run_id)["status"]
        store.transition(run_id, "failed" if status == "running" else "blocked", {"error": str(error)})
        raise
    return store.run(run_id)
