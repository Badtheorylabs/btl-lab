import json
import math
from pathlib import Path

import pytest

from btl_lab.cli import main
from btl_measure.core import compare, fingerprint, load, validate


def document(tmp_path, name, model_revision="a" * 40, values=(1.0, 0.0, 1.0)):
    path = tmp_path / name
    path.write_text(json.dumps({
        "model_id": "test/model", "model_revision": model_revision,
        "taskset_id": "taskset", "taskset_revision": "b" * 40,
        "records": [{"id": f"case-{i}", "source_id": f"source-{i}", "score": value}
                    for i, value in enumerate(values)],
    }) + "\n")
    return path


def test_measure_recomputes_scores_from_records(tmp_path):
    evaluation = load(document(tmp_path, "eval.json"))
    result = validate(evaluation)
    assert result["records"] == 3
    assert result["mean_score"] == pytest.approx(2 / 3)
    assert result["pass_rate"] == pytest.approx(2 / 3)
    assert result["record_ids_sha256"] == fingerprint(["case-0", "case-1", "case-2"])


@pytest.mark.parametrize("records", [[], [{"id": "x", "score": 2}],
                                      [{"id": "x", "score": float("nan")}],
                                      [{"id": "x", "score": 1}, {"id": "x", "score": 0}]])
def test_malformed_evaluations_fail(records, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model_id": "m", "model_revision": "a" * 40,
                                "taskset_id": "t", "taskset_revision": "b" * 40,
                                "records": records}, allow_nan=True))
    with pytest.raises(ValueError):
        load(path)


def test_boolean_passed_is_normalized(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"model_id": "m", "model_revision": "a" * 40,
                                "taskset_id": "t", "taskset_revision": "b" * 40,
                                "records": [{"id": "x", "passed": True}, {"id": "y", "passed": False}]}))
    assert validate(load(path))["pass_rate"] == 0.5


def test_compare_is_paired_and_recomputes_regressions(tmp_path):
    base = load(document(tmp_path, "base.json", values=(1.0, 0.0, 0.0)))
    candidate = load(document(tmp_path, "candidate.json", model_revision="c" * 40, values=(0.0, 1.0, 1.0)))
    result = compare(base, candidate)
    assert result["comparable"]
    assert result["regression_ids"] == ["case-0"]
    assert result["improvement_ids"] == ["case-1", "case-2"]
    assert result["baseline"]["mean_score"] == pytest.approx(1 / 3)
    assert result["candidate"]["mean_score"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("change", [
    {"taskset_id": "other"},
    {"taskset_revision": "c" * 40},
    {"model_revision": "a" * 40},
    {"records": [{"id": "other", "score": 1.0}]},
])
def test_compare_rejects_noncomparable_pairs(tmp_path, change):
    base_path = document(tmp_path, "base.json")
    candidate_data = json.loads(base_path.read_text())
    candidate_data.update(change)
    candidate_data["model_revision"] = change.get("model_revision", "c" * 40)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate_data))
    result = compare(load(base_path), load(candidate_path))
    assert not result["comparable"]
    assert result["reasons"]


def test_compare_threshold_is_explicit(tmp_path):
    base = load(document(tmp_path, "base.json", values=(0.2, 0.2)))
    candidate = load(document(tmp_path, "candidate.json", model_revision="c" * 40, values=(0.4, 0.4)))
    assert compare(base, candidate, minimum_relative_improvement=0.9)["meets_declared_threshold"]
    assert not compare(base, candidate, minimum_relative_improvement=1.1)["meets_declared_threshold"]
    with pytest.raises(ValueError):
        compare(base, candidate, minimum_relative_improvement=math.nan)


def test_cli_measure_records_verifiable_comparison(lab, tmp_path, capsys):
    workspace, store = lab
    base = document(tmp_path, "base.json")
    candidate = document(tmp_path, "candidate.json", model_revision="c" * 40, values=(1.0, 1.0, 1.0))
    # The root CLI only accepts workspace-local inputs. Copy the small fixtures into it.
    base_workspace = workspace.root / "base-eval.json"
    candidate_workspace = workspace.root / "candidate-eval.json"
    base_workspace.write_bytes(base.read_bytes())
    candidate_workspace.write_bytes(candidate.read_bytes())
    code = main(["--workspace", str(workspace.root), "--json", "measure", "compare",
                 "base-eval.json", "candidate-eval.json", "--project", "lab"])
    assert code == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "passed"
    assert run["result"]["comparable"]
    assert len(run["artifacts"]) == 3
    assert main(["--workspace", str(workspace.root), "runs", run["id"], "--verify"]) == 0
    capsys.readouterr()


def test_cli_measure_incomparable_returns_nonzero_and_keeps_report(lab, tmp_path, capsys):
    workspace, store = lab
    base = document(tmp_path, "base.json")
    candidate = document(tmp_path, "candidate.json", model_revision="a" * 40)
    (workspace.root / "base.json").write_bytes(base.read_bytes())
    (workspace.root / "candidate.json").write_bytes(candidate.read_bytes())
    code = main(["--workspace", str(workspace.root), "--json", "measure", "compare",
                 "base.json", "candidate.json", "--project", "lab"])
    assert code == 2
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "failed"
    assert run["result"]["reasons"]
    assert run["artifacts"]


def test_measure_input_outside_workspace_rejected(lab, tmp_path):
    workspace, store = lab
    outside = document(tmp_path.parent, f"{tmp_path.name}-outside.json")
    assert main(["--workspace", str(workspace.root), "measure", "validate", str(outside),
                 "--project", "lab"]) == 2
