from __future__ import annotations

import json
import uuid

from btl_train.operations import OPERATIONS

from .operation_runs import create_operation, preflight_operation
from .research import Research
from .workspace import inside, read_json


def add_workflows(sub):
    research = sub.add_parser("research", help="Protocols, bounded attempts, comparisons and decisions")
    r = research.add_subparsers(dest="action", required=True)
    r.add_parser("list")
    create = r.add_parser("create")
    create.add_argument("file", help="Workspace-relative protocol JSON")
    amend = r.add_parser("amend")
    amend.add_argument("id")
    amend.add_argument("file", help="Replacement draft JSON; previous draft stays in event history")
    for action in ("show", "freeze", "compare", "report"):
        r.add_parser(action).add_argument("id")
    ready = r.add_parser("ready")
    ready.add_argument("id")
    link = r.add_parser("link-run")
    link.add_argument("id")
    link.add_argument("run_id")
    link.add_argument("--role", required=True)
    link.add_argument("--arm", choices=["baseline", "candidate"])
    link.add_argument("--note", default="")
    run = r.add_parser("run")
    run.add_argument("id")
    run.add_argument("--arm", required=True, choices=["baseline", "candidate"])
    decide = r.add_parser("decide")
    decide.add_argument("id")
    decide.add_argument("--outcome", required=True, choices=["accept", "reject", "inconclusive"])
    decide.add_argument("--reason", required=True)
    operation = sub.add_parser("operation", help="Explicit model operation plans and input checks")
    op = operation.add_subparsers(dest="action", required=True)
    op.add_parser("types")
    create = op.add_parser("plan")
    create.add_argument("file", help="Workspace-relative operation JSON")
    create.add_argument("--experiment")
    preflight = op.add_parser("preflight")
    preflight.add_argument("id", help="Model-operation plan ID")
    preflight.add_argument("--max-hash-mb", type=int, default=64)


def execute_workflow(args, workspace, store):
    if args.command == "operation":
        if args.action == "types":
            return [{"operation": name, "required_inputs": sorted(rule["inputs"]),
                     "candidate_backends": sorted(rule["backends"]),
                     "implemented_stages": ["plan", "input-preflight"]}
                    for name, rule in OPERATIONS.items()]
        if args.action == "plan":
            return create_operation(workspace, store, read_json(inside(workspace.root, args.file)), args.experiment)
        return preflight_operation(workspace, store, args.id, args.max_hash_mb * 1024 * 1024)
    research = Research(workspace, store)
    if args.action == "list":
        return research.list()
    if args.action == "create":
        return research.create(read_json(inside(workspace.root, args.file)))
    if args.action == "amend":
        return research.amend(args.id, read_json(inside(workspace.root, args.file)))
    if args.action == "show":
        return research.get(args.id)
    if args.action == "freeze":
        return research.freeze(args.id)
    if args.action == "ready":
        return research.readiness(args.id)
    if args.action == "link-run":
        return research.link_run(args.id, args.run_id, args.role, args.arm, args.note)
    if args.action == "run":
        return research.run(args.id, args.arm)
    if args.action == "compare":
        return research.compare(args.id)
    if args.action == "decide":
        return research.decide(args.id, args.outcome, args.reason)
    if args.action == "report":
        experiment = research.get(args.id)
        comparison = research.compare(args.id) if experiment["frozen"] else {"comparable": False, "reasons": ["Draft protocol"]}
        directory = workspace.state / "research" / experiment["id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("report-" + uuid.uuid4().hex[:10] + ".json")
        path.write_text(json.dumps({"experiment": experiment, "comparison": comparison,
                                   "runs": [store.run(r["run_id"]) for r in experiment["runs"]],
                                   "operations": [store.run(r["id"]) for r in experiment["operations"]],
                                   "readiness": research.readiness(args.id)}, indent=2) + "\n")
        return {"path": str(path), "experiment_id": experiment["id"], "comparable": comparison["comparable"]}
    raise ValueError("Unknown workflow action")
