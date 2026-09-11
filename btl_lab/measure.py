from __future__ import annotations

import json
from pathlib import Path

from btl_measure.core import compare as compare_evaluations
from btl_measure.core import load as load_evaluation
from btl_measure.core import sha256, validate

from .checks import artifact_record
from .store import Store
from .workspace import Workspace, inside


def add_measure(sub):
    cli = sub.add_parser("measure", help="BTL Measure: independent per-item evaluation and comparison")
    actions = cli.add_subparsers(dest="action", required=True)
    validate_parser = actions.add_parser("validate")
    validate_parser.add_argument("file", help="Workspace-relative evaluation JSON")
    validate_parser.add_argument("--project", default="workspace/btl-lab")
    compare_parser = actions.add_parser("compare")
    compare_parser.add_argument("baseline", help="Workspace-relative baseline evaluation JSON")
    compare_parser.add_argument("candidate", help="Workspace-relative candidate evaluation JSON")
    compare_parser.add_argument("--project", default="workspace/btl-lab")
    compare_parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)


def execute_measure(args, workspace: Workspace, store: Store) -> dict:
    workspace.project(args.project)
    if args.action == "validate":
        input_path = inside(workspace.root, args.file)
        evaluation = load_evaluation(input_path)
        plan = {"schema_version": 1, "project_id": args.project, "recipe_id": "measure:validate-v1",
                "backend": "btl-measure", "operation": "validate", "input": args.file,
                "input_sha256": sha256(input_path), "scope": "Independent evaluation-record validation"}
        run_id = store.create(plan, kind="evaluation-validation")
        store.transition(run_id, "running")
        result = validate(evaluation)
        result["run_id"] = run_id
        return _finish(workspace, store, run_id, input_path, result, "evaluation-validation")
    baseline_path = inside(workspace.root, args.baseline)
    candidate_path = inside(workspace.root, args.candidate)
    baseline = load_evaluation(baseline_path)
    candidate = load_evaluation(candidate_path)
    plan = {"schema_version": 1, "project_id": args.project, "recipe_id": "measure:compare-v1",
            "backend": "btl-measure", "operation": "compare", "baseline": args.baseline,
            "candidate": args.candidate, "baseline_sha256": sha256(baseline_path),
            "candidate_sha256": sha256(candidate_path),
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "scope": "Paired descriptive evaluation comparison; no automatic promotion"}
    run_id = store.create(plan, kind="evaluation-comparison")
    store.transition(run_id, "running")
    result = compare_evaluations(baseline, candidate,
                                 minimum_relative_improvement=args.minimum_relative_improvement)
    result["run_id"] = run_id
    return _finish(workspace, store, run_id, [baseline_path, candidate_path], result, "evaluation-comparison")


def _finish(workspace, store: Store, run_id: str, inputs, result: dict, kind: str) -> dict:
    directory = workspace.state / "measure" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = directory / "report.json"
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    input_paths = inputs if isinstance(inputs, list) else [inputs]
    for path in input_paths:
        store.attach(run_id, artifact_record(workspace, str(path.relative_to(workspace.root)),
                                             "evaluation-input", "measurement-input"))
    store.attach(run_id, artifact_record(workspace, str(report.relative_to(workspace.root)),
                                         "measurement-report", "measurement"))
    passed = bool(result.get("passed", result.get("comparable", False)))
    store.transition(run_id, "passed" if passed else "failed", result)
    return store.run(run_id)
