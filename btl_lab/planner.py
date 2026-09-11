from __future__ import annotations

import json
import platform
from pathlib import Path

from .workspace import Workspace, inside, sha256
from btl_rl.prime_rl import inspect_prime_rl


def plan(workspace: Workspace, recipe_id: str, checkout: Path | None = None) -> dict:
    recipe = workspace.recipe(recipe_id)
    result = {"schema_version": 1, "recipe_id": recipe["id"],
              "project_id": recipe["project_id"], "backend": recipe["backend"],
              "operation": recipe["operation"], "recipe": recipe,
              "recipe_catalog_sha256": sha256(workspace.catalog),
              "host": {"system": platform.system(), "machine": platform.machine()},
              "runnable": False, "blockers": [], "inputs": [],
              "provisions_resources": False, "downloads_dependencies": False,
              "claim": "Plan only; no training, benchmark or capability result"}
    if recipe["backend"] == "local-audit":
        if recipe["operation"] not in {"registry-audit", "manifest-audit"}:
            raise ValueError("Unsupported local audit operation")
        if recipe["operation"] == "registry-audit":
            result["inputs"].append({"path": str(workspace.registry.relative_to(workspace.root)),
                                     "sha256": sha256(workspace.registry)})
        else:
            manifest = inside(workspace.root, recipe["manifest"])
            result["inputs"].append({"path": recipe["manifest"], "sha256": sha256(manifest)})
        result["runnable"] = True
        result["execution_scope"] = "Local read-only audit; writes only ledger and audit result"
        return result
    if recipe["backend"] != "prime-rl":
        raise ValueError(f"No planner implemented for backend: {recipe['backend']}")
    return inspect_prime_rl(recipe, checkout, result)


def validate_plan_inputs(workspace: Workspace, resolved: dict):
    if sha256(workspace.catalog) != resolved["recipe_catalog_sha256"]:
        raise ValueError("Recipe catalog changed after planning")
    for item in resolved["inputs"]:
        if sha256(inside(workspace.root, item["path"])) != item["sha256"]:
            raise ValueError(f"Input changed after planning: {item['path']}")
