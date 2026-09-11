import json

import pytest

from btl_lab.cli import main
from btl_lab.research import Research
from btl_lab.operation_runs import create_operation, preflight_operation
from btl_lab.checks import verify_artifacts
from btl_lab.workspace import sha256
from btl_train.operations import validate_spec, preflight


def protocol():
    return {"schema_version": 1, "project_id": "lab", "title": "Control-layer test",
            "question": "Can complete attempts be compared?", "hypothesis": "Candidate improves elapsed time",
            "falsification": "No improvement or invalid observations", "evidence_class": "integrity",
            "arms": {"baseline": {"recipe": "registry"}, "candidate": {"recipe": "registry"}},
            "controls": {"workload": "identical local integrity check"}, "inputs": ["source.txt"],
            "metric": {"field": "elapsed_seconds", "direction": "min", "minimum_relative_improvement": 2},
            "repeats": 1, "external_spend_cap_usd": 0}


def operation(workspace):
    ref = {"path": "source.txt", "sha256": sha256(workspace.root / "source.txt")}
    return {"schema_version": 1, "project_id": "lab", "name": "Test-only SFT contract",
            "operation": "sft", "backend": "unsloth-core",
            "model": {"id": "test/model", "revision": "a" * 40},
            "engine": {"id": "test/engine", "revision": "b" * 40},
            "inputs": {"data": ref, "evaluation": ref}, "context_length": 2048,
            "effective_batch_size": 4, "precision": "bf16",
            "budget": {"wall_seconds": 60, "external_spend_usd": 0}}


def test_research_lifecycle_retains_protocol_and_decision(lab):
    workspace, store = lab
    research = Research(workspace, store)
    original = protocol()
    exp = research.create(original)
    original["question"] = "changed outside the ledger"
    assert research.get(exp["id"])["spec"]["question"] != original["question"]
    research.freeze(exp["id"])
    research.run(exp["id"], "baseline")
    research.run(exp["id"], "candidate")
    report = research.compare(exp["id"])
    assert report["comparable"]
    assert len(report["values"]["baseline"]) == 1
    decided = research.decide(exp["id"], "inconclusive", "This test validates mechanics only")
    assert decided["status"] == "decided"
    assert decided["decision"]["comparison"]["protocol_hash"] == decided["protocol_hash"]
    with pytest.raises(ValueError, match="closed"):
        research.run(exp["id"], "baseline")


def test_draft_cannot_run_and_frozen_protocol_cannot_refreeze(lab):
    research = Research(*lab)
    exp = research.create(protocol())
    with pytest.raises(ValueError, match="Freeze"):
        research.run(exp["id"], "baseline")
    research.freeze(exp["id"])
    with pytest.raises(ValueError, match="draft"):
        research.freeze(exp["id"])


def test_changed_frozen_input_invalidates_comparison(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    (workspace.root / "source.txt").write_text("drift")
    with pytest.raises(ValueError, match="Frozen input"):
        research.run(exp["id"], "baseline")
    assert not research.compare(exp["id"])["comparable"]
    assert research.decide(exp["id"], "inconclusive", "Input drift invalidated protocol")["status"] == "decided"


def test_recipe_change_after_freeze_is_rejected(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    catalog = json.loads(workspace.catalog.read_text())
    catalog["recipes"][0]["operation"] = "manifest-audit"
    workspace.catalog.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="Recipe changed"):
        research.run(exp["id"], "baseline")


def test_failed_attempt_consumes_frozen_budget(lab):
    workspace, store = lab
    research = Research(workspace, store)
    spec = protocol()
    spec["inputs"] = []
    exp = research.freeze(research.create(spec)["id"])
    (workspace.root / "source.txt").unlink()
    assert research.run(exp["id"], "baseline")["status"] == "failed"
    with pytest.raises(ValueError, match="budget exhausted"):
        research.run(exp["id"], "baseline")
    with pytest.raises(ValueError, match="complete linked evidence"):
        research.decide(exp["id"], "accept", "Not allowed")


def test_checks_cannot_be_promoted_as_behavioral_evidence(lab):
    research = Research(*lab)
    spec = protocol()
    spec["evidence_class"] = "behavioral"
    exp = research.freeze(research.create(spec)["id"])
    research.run(exp["id"], "baseline")
    research.run(exp["id"], "candidate")
    report = research.compare(exp["id"])
    assert not report["comparable"]
    assert any("evidence class" in r for r in report["reasons"])


def test_comparison_rechecks_receipt_integrity(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    run = research.run(exp["id"], "baseline")
    research.run(exp["id"], "candidate")
    (workspace.root / run["artifacts"][0]["path"]).write_text("tampered")
    assert not research.compare(exp["id"])["comparable"]


def test_negative_result_cannot_pass_frozen_acceptance_threshold(lab):
    research = Research(*lab)
    exp = research.freeze(research.create(protocol())["id"])
    research.run(exp["id"], "baseline")
    research.run(exp["id"], "candidate")
    from btl_lab.runner import run_check
    evaluation = run_check(lab[0], lab[1], "registry")
    research.link_run(exp["id"], evaluation["id"], "evaluation")
    with pytest.raises(ValueError, match="threshold"):
        research.decide(exp["id"], "accept", "Threshold was not met")
    assert research.decide(exp["id"], "reject", "Declared criterion not met")["status"] == "decided"


@pytest.mark.parametrize("bad", [True, 0, 101, -1])
def test_invalid_attempt_limits_rejected(lab, bad):
    research = Research(*lab)
    spec = protocol()
    spec["repeats"] = bad
    with pytest.raises(ValueError, match="repeats"):
        research.create(spec)


def test_local_executor_rejects_paid_protocol(lab):
    research = Research(*lab)
    spec = protocol()
    spec["external_spend_cap_usd"] = 5
    with pytest.raises(ValueError, match="zero external spend"):
        research.create(spec)


def test_operation_plan_and_preflight_are_distinct_linked_runs(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.create(protocol())
    plan = create_operation(workspace, store, operation(workspace), exp["id"])
    checked = preflight_operation(workspace, store, plan["id"], 1024)
    assert plan["status"] == "planned"
    assert checked["status"] == "passed"
    assert checked["plan"]["parent_run_id"] == plan["id"]
    assert checked["result"]["training_ready"] is False
    assert len(research.get(exp["id"])["operations"]) == 2
    assert verify_artifacts(workspace, checked)["passed"]


def test_operation_requires_immutable_pins_and_required_inputs(lab):
    workspace, _ = lab
    spec = operation(workspace)
    spec["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="full commit"):
        validate_spec(spec)
    spec = operation(workspace)
    del spec["inputs"]["evaluation"]
    with pytest.raises(ValueError, match="input roles"):
        validate_spec(spec)


def test_distillation_requires_teacher_and_objective(lab):
    workspace, _ = lab
    spec = operation(workspace) | {"operation": "distill", "backend": "prime-rl"}
    with pytest.raises(ValueError, match="teacher"):
        validate_spec(spec)
    spec["teacher"] = {"id": "test/teacher", "revision": "c" * 40}
    with pytest.raises(ValueError, match="objective"):
        validate_spec(spec)


def test_rl_requires_environment_contract(lab):
    workspace, _ = lab
    spec = operation(workspace) | {"operation": "rl", "backend": "prime-rl"}
    with pytest.raises(ValueError, match="environment.taskset"):
        validate_spec(spec)


def test_operation_input_drift_and_hash_budget_fail_closed(lab):
    workspace, store = lab
    plan = create_operation(workspace, store, operation(workspace))
    assert preflight_operation(workspace, store, plan["id"], 1)["status"] == "failed"
    (workspace.root / "source.txt").write_text("different")
    assert preflight_operation(workspace, store, plan["id"], 1024)["status"] == "failed"


def test_operation_rejects_plan_mutation(lab):
    workspace, _ = lab
    resolved = validate_spec(operation(workspace))
    resolved["spec"]["precision"] = "fp32"
    with pytest.raises(ValueError, match="plan changed"):
        preflight(workspace.root, resolved)


def test_cli_research_report_and_operation_catalog(lab, capsys):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.create(protocol())
    args = ["--workspace", str(workspace.root), "--json"]
    assert main(args + ["research", "report", exp["id"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["comparable"]
    assert main(args + ["operation", "types"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 9


def test_question_can_precede_hypothesis_and_frozen_protocol(lab):
    research = Research(*lab)
    draft = {"schema_version": 1, "project_id": "lab", "title": "Open question",
             "question": "Which method should we investigate?", "literature": []}
    exp = research.create(draft)
    assert exp["status"] == "draft"
    with pytest.raises(ValueError, match="hypothesis"):
        research.freeze(exp["id"])
    research.amend(exp["id"], protocol())
    assert research.freeze(exp["id"])["status"] == "frozen"
    with pytest.raises(ValueError, match="Only drafts"):
        research.amend(exp["id"], protocol())
    assert any(e["event"] == "draft-amended" for e in research.get(exp["id"])["events"])


def test_implementation_change_invalidates_frozen_experiment(lab, monkeypatch):
    research = Research(*lab)
    exp = research.freeze(research.create(protocol())["id"])
    monkeypatch.setattr("btl_lab.research.implementation_sources", lambda: {"changed.py": "a" * 64})
    with pytest.raises(ValueError, match="implementation changed"):
        research.run(exp["id"], "baseline")
    assert not research.compare(exp["id"])["comparable"]


def test_readiness_reports_arm_and_evidence_blockers(lab):
    research = Research(*lab)
    exp = research.freeze(research.create(protocol())["id"])
    readiness = research.readiness(exp["id"])
    assert not readiness["ready_for_decision"]
    assert readiness["arms"]["baseline"]["linked"] == 0
    assert readiness["arms"]["candidate"]["required"] == 1
    assert "Missing required evidence role: evaluation" in readiness["blockers"]


def test_linked_runs_become_arm_observations_and_supporting_evidence(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    from btl_lab.runner import run_check
    baseline = run_check(workspace, store, "registry")
    candidate = run_check(workspace, store, "registry")
    evaluation = run_check(workspace, store, "registry")
    research.link_run(exp["id"], baseline["id"], "arm-observation", "baseline")
    research.link_run(exp["id"], candidate["id"], "arm-observation", "candidate")
    linked = research.link_run(exp["id"], evaluation["id"], "evaluation", note="Paired evaluation receipt")
    readiness = research.readiness(exp["id"])
    assert readiness["ready_for_decision"]
    assert readiness["evidence"]["evaluation"] == 1
    assert {link["role"] for link in linked["links"]} == {"arm-observation", "evaluation"}
    with pytest.raises(ValueError, match="already linked"):
        research.link_run(exp["id"], evaluation["id"], "evaluation", note="duplicate")


def test_linking_failed_arm_run_blocks_readiness(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    from btl_lab.runner import run_check
    (workspace.root / "source.txt").unlink()
    failed = run_check(workspace, store, "source")
    assert failed["status"] == "failed"
    research.link_run(exp["id"], failed["id"], "arm-observation", "baseline")
    assert any("incomplete or failed" in blocker for blocker in research.readiness(exp["id"])["blockers"])


def test_linking_failed_supporting_evidence_blocks_readiness(lab):
    workspace, store = lab
    research = Research(workspace, store)
    exp = research.freeze(research.create(protocol())["id"])
    from btl_lab.runner import run_check
    (workspace.root / "source.txt").unlink()
    failed = run_check(workspace, store, "source")
    assert failed["status"] == "failed"
    research.link_run(exp["id"], failed["id"], "evaluation")
    readiness = research.readiness(exp["id"])
    assert not readiness["ready_for_decision"]
    assert readiness["evidence_details"]["evaluation"][0]["valid"] is False
    assert any("Evidence link evaluation" in blocker for blocker in readiness["blockers"])


def test_decision_requires_ready_evidence_for_accept_or_reject(lab):
    research = Research(*lab)
    exp = research.freeze(research.create(protocol())["id"])
    with pytest.raises(ValueError, match="complete linked evidence"):
        research.decide(exp["id"], "accept", "Premature acceptance")
    with pytest.raises(ValueError, match="complete linked evidence"):
        research.decide(exp["id"], "reject", "Premature rejection")
    decided = research.decide(exp["id"], "inconclusive", "No observations yet")
    assert decided["status"] == "decided"


def test_required_evidence_roles_are_validated(lab):
    research = Research(*lab)
    spec = protocol()
    spec["required_evidence_roles"] = []
    with pytest.raises(ValueError, match="required_evidence_roles"):
        research.create(spec)
