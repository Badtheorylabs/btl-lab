from __future__ import annotations

import json
import time

from btl_train.operations import preflight, validate_spec

from .checks import artifact_record
from .store import Store
from .workspace import Workspace


def create_operation(workspace: Workspace, store: Store, spec: dict, experiment_id: str | None = None) -> dict:
    workspace.project(spec["project_id"])
    resolved = validate_spec(spec)
    resolved |= {"project_id": spec["project_id"], "recipe_id": "operation:" + spec["operation"]}
    if experiment_id:
        from .research import Research
        experiment = Research(workspace, store).get(experiment_id)
        if experiment["project_id"] != spec["project_id"]:
            raise ValueError("Operation and research project differ")
        if experiment["status"] == "decided":
            raise ValueError("Cannot add an operation to a closed experiment")
        resolved["experiment_id"] = experiment_id
        resolved["experiment_protocol_hash"] = experiment["protocol_hash"]
    run_id = store.create(resolved, kind="model-operation-plan")
    return store.run(run_id)


def preflight_operation(workspace: Workspace, store: Store, plan_id: str, max_hash_bytes: int) -> dict:
    parent = store.run(plan_id)
    if parent["kind"] != "model-operation-plan":
        raise ValueError("Pass a model-operation plan ID")
    resolved = parent["plan"] | {"parent_run_id": plan_id, "stage": "input-preflight"}
    run_id = store.create(resolved, kind="model-operation-preflight")
    store.transition(run_id, "running")
    start = time.monotonic()
    try:
        result = preflight(workspace.root, resolved, max_hash_bytes)
        result["elapsed_seconds"] = time.monotonic() - start
        directory = workspace.state / "results" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "result.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        store.attach(run_id, artifact_record(workspace, str(path.relative_to(workspace.root)),
                                             "operation-input-preflight", "diagnostic"))
        store.transition(run_id, "passed" if result["passed"] else "failed", result)
    except KeyboardInterrupt:
        store.transition(run_id, "interrupted", {"error": "Keyboard interrupt"})
        raise
    except Exception as error:
        store.transition(run_id, "failed", {"error": str(error)})
        raise
    return store.run(run_id)
