from __future__ import annotations

import json
import time
from pathlib import Path

from .checks import artifact_record, manifest_audit, registry_audit
from .planner import plan, validate_plan_inputs
from .store import Store
from .workspace import Workspace


def run_check(workspace: Workspace, store: Store, recipe_id: str, checkout: Path | None = None) -> dict:
    resolved = plan(workspace, recipe_id, checkout)
    run_id = store.create(resolved)
    return execute_check(workspace, store, run_id)


def execute_check(workspace: Workspace, store: Store, run_id: str) -> dict:
    resolved = store.run(run_id)["plan"]
    if not resolved["runnable"]:
        store.transition(run_id, "blocked", {"blockers": resolved["blockers"]})
        return store.run(run_id)
    store.transition(run_id, "running")
    started = time.monotonic()
    try:
        validate_plan_inputs(workspace, resolved)
        if resolved["operation"] == "registry-audit":
            result = registry_audit(workspace)
        elif resolved["operation"] == "manifest-audit":
            result = manifest_audit(workspace, resolved["recipe"]["manifest"])
        else:
            raise ValueError("Unsupported operation")
        result["elapsed_seconds"] = time.monotonic() - started
        result["external_spend_usd"] = 0
        result["model_calls"] = 0
        result["evidence_class"] = "integrity"
        directory = workspace.state / "results" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        receipt = directory / "result.json"
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        store.attach(run_id, artifact_record(workspace, str(receipt.relative_to(workspace.root)),
                                             "audit-result", "diagnostic"))
        store.transition(run_id, "passed" if result["passed"] else "failed", result)
    except KeyboardInterrupt:
        store.transition(run_id, "interrupted", {"error": "Keyboard interrupt"})
        raise
    except Exception as error:
        store.transition(run_id, "failed", {"error": str(error), "type": type(error).__name__})
        raise
    return store.run(run_id)
