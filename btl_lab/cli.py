from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .checks import artifact_record, verify_artifacts
from .planner import plan
from .runner import run_check
from .store import Store
from .workspace import Workspace, read_json
from .workflow_cli import add_workflows, execute_workflow
from .finetune import add_finetune, execute_finetune
from .advance import add_advance, execute_advance
from .measure import add_measure, execute_measure


def default_workspace() -> Path:
    if os.environ.get("BTL_WORKSPACE"):
        return Path(os.environ["BTL_WORKSPACE"])
    candidates = (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents)
    for parent in candidates:
        if (parent / ".btl-workspace.json").is_file():
            return parent
    for parent in candidates:
        if (parent / "btl-lab/projects.json").is_file():
            return parent
    return Path.cwd()


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="BTL projects, local checks, plans and evidence")
    cli.add_argument("--workspace", type=Path, default=default_workspace())
    cli.add_argument("--json", action="store_true", help="Machine-readable output")
    sub = cli.add_subparsers(dest="command", required=True)
    add_workflows(sub)
    add_finetune(sub)
    add_advance(sub)
    add_measure(sub)
    sub.add_parser("status")
    sub.add_parser("layout")
    projects = sub.add_parser("projects")
    projects.add_argument("--program")
    projects.add_argument("--lifecycle")
    project = sub.add_parser("project")
    project.add_argument("id")
    project.add_argument("--owner")
    project.add_argument("--lifecycle", choices=["untriaged", "proposed", "active", "paused", "completed", "archived"])
    project.add_argument("--next-gate")
    sub.add_parser("recipes")
    sub.add_parser("backends")
    for command in ("plan", "run"):
        item = sub.add_parser(command)
        item.add_argument("recipe")
        item.add_argument("--checkout", type=Path)
        if command == "plan":
            item.add_argument("--save", action="store_true")
    runs = sub.add_parser("runs")
    runs.add_argument("id", nargs="?")
    runs.add_argument("--verify", action="store_true")
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--project", required=True)
    evidence.add_argument("--file", required=True, help="Existing file inside the workspace")
    evidence.add_argument("--label", required=True)
    decision = sub.add_parser("decision")
    decision.add_argument("project")
    decision.add_argument("text")
    decision.add_argument("--run")
    return cli


def projects_with_status(workspace: Workspace, store: Store) -> list[dict]:
    overrides = store.overrides()
    return [project | overrides.get(project["id"], {}) for project in workspace.projects()]


def execute(args, workspace: Workspace, store: Store):
    if args.command in {"adapt", "finetune"}:
        return execute_finetune(args, workspace, store)
    if args.command == "advance":
        return execute_advance(args, workspace, store)
    if args.command == "measure":
        return execute_measure(args, workspace, store)
    if args.command in {"research", "operation"}:
        return execute_workflow(args, workspace, store)
    if args.command == "layout":
        marker = workspace.root / ".btl-workspace.json"
        if not marker.is_file():
            raise ValueError("This workspace has no grouped layout marker")
        layout = read_json(marker)
        return [{"group": name, "path": str(workspace.root / name),
                 "entries": sorted(p.name for p in (workspace.root / name).iterdir()
                                   if not p.is_symlink() and not p.name.startswith("."))}
                for name in layout.get("groups", [])]
    if args.command == "status":
        projects = projects_with_status(workspace, store)
        runs = store.runs()
        return {"workspace": str(workspace.root), "model_family": "Tinfield",
                "projects": len(projects), "untriaged": sum(p["lifecycle"] == "untriaged" for p in projects),
                "recorded_runs": len(runs), "preferred_rl": "prime-rl",
                "implemented": ["project registry", "decisions", "local integrity checks",
                                "artifact ledger", "pinned Prime-RL planning",
                                "frozen research protocols and comparisons", "explicit model-operation input preflight",
                                "linked evidence runs and readiness-gated research decisions",
                                "BTL Adapt A100 profile and recovery qualification",
                                "BTL Advance local RL mechanics and recovery qualification",
                                "BTL Measure per-item evaluation comparisons"],
                "not_enabled": ["production-qualified GPU recipes", "cloud provisioning", "paid model calls", "automatic publication"]}
    if args.command == "projects":
        return [p for p in projects_with_status(workspace, store)
                if (not args.program or p["primary_program"] == args.program)
                and (not args.lifecycle or p["lifecycle"] == args.lifecycle)]
    if args.command == "project":
        project = workspace.project(args.id)
        fields = {name: getattr(args, name) for name in ("owner", "lifecycle", "next_gate")
                  if getattr(args, name) is not None}
        if fields:
            store.update_project(args.id, fields)
        return project | store.overrides().get(args.id, {}) | {"decisions": store.decisions(args.id)}
    if args.command == "recipes":
        return workspace.recipes()
    if args.command == "backends":
        return read_json(workspace.lab / "backends.json")["lanes"]
    if args.command == "plan":
        resolved = plan(workspace, args.recipe, args.checkout)
        if args.save:
            resolved["run_id"] = store.create(resolved, kind="plan")
        return resolved
    if args.command == "run":
        return run_check(workspace, store, args.recipe, args.checkout)
    if args.command == "runs":
        if args.verify and not args.id:
            raise ValueError("--verify requires a run ID")
        if not args.id:
            return store.runs()
        run = store.run(args.id)
        return verify_artifacts(workspace, run) if args.verify else run
    if args.command == "evidence":
        workspace.project(args.project)
        artifact = artifact_record(workspace, args.file, args.label, "reported-unreviewed")
        resolved = {"project_id": args.project, "recipe_id": "external-evidence",
                    "label": args.label, "artifact": artifact,
                    "claim": "Imported existing evidence; not independently reproduced"}
        run_id = store.create(resolved, kind="external-evidence")
        store.attach(run_id, artifact)
        store.transition(run_id, "recorded", {"evidence_status": "reported-unreviewed"})
        return store.run(run_id)
    if args.command == "decision":
        workspace.project(args.project)
        store.decide(args.project, args.text, args.run)
        return store.decisions(args.project)
    raise ValueError("Unknown command")


def display(value, command: str, as_json: bool):
    if as_json:
        print(json.dumps(value, indent=2))
    elif command == "layout":
        for group in value:
            print(f"{group['group']}/")
            for entry in group["entries"]:
                print(f"  {entry}")
    elif command == "projects":
        for p in value:
            print(f"{p['id']}\t{p['primary_program']}\t{p['lifecycle']}\t{p.get('owner') or 'unassigned'}")
    elif command == "recipes":
        for r in value:
            print(f"{r['id']}\t{r['backend']}\t{r['status']}\t{r['title']}")
    elif command == "backends":
        for lane in value:
            print(f"{lane['id']}\t{lane['preferred']}\t{lane['status']}")
    elif command == "runs" and isinstance(value, list):
        for r in value:
            print(f"{r['id']}\t{r['recipe_id']}\t{r['kind']}\t{r['status']}")
    elif command in {"run", "evidence"}:
        print(f"Run: {value['id']}\nKind: {value['kind']}\nStatus: {value['status']}")
        result = value.get("result") or {}
        print(result.get("scope", value["plan"].get("claim", "")))
        for blocker in result.get("blockers", []):
            print(f"- {blocker}")
    else:
        print(json.dumps(value, indent=2))


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    store = None
    try:
        workspace = Workspace(args.workspace)
        store = Store(workspace.state / "lab.sqlite3")
        value = execute(args, workspace, store)
        display(value, args.command, args.json)
        if args.command == "run" and value["status"] != "passed":
            return 2
        if args.command in {"adapt", "finetune"} and value["status"] != "passed":
            return 2
        if args.command == "advance" and value["status"] != "passed":
            return 2
        if args.command == "measure" and value["status"] != "passed":
            return 2
        if args.command == "runs" and args.verify and not value["passed"]:
            return 2
        if args.command in {"research", "operation"}:
            if args.action in {"run", "preflight"} and value.get("status") != "passed":
                return 2
            if args.action == "compare" and not value["comparable"]:
                return 2
        return 0
    except (ValueError, OSError, KeyError, sqlite3.Error) as error:
        print(f"btl: {error}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
