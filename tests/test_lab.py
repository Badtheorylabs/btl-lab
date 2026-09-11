import json
import subprocess
from pathlib import Path

import pytest

from btl_lab.checks import artifact_record, manifest_audit, verify_artifacts
from btl_lab.cli import main
from btl_lab.planner import plan, validate_plan_inputs
from btl_lab.runner import run_check
from btl_lab.store import Store
from btl_lab.workspace import Workspace, inside, sha256




def test_real_check_records_events_and_verifiable_result(lab):
    workspace, store = lab
    run = run_check(workspace, store, "source")
    assert run["status"] == "passed"
    assert [event["status"] for event in run["events"]] == ["planned", "running", "passed"]
    assert run["result"]["model_calls"] == 0
    assert verify_artifacts(workspace, run)["passed"]
    assert run["artifacts"][0]["evidence_status"] == "diagnostic"


def test_corrupted_source_fails_without_erasing_evidence(lab):
    workspace, store = lab
    (workspace.root / "source.txt").write_text("changed source")
    run = run_check(workspace, store, "source")
    assert run["status"] == "failed"
    assert run["artifacts"]
    assert run["result"]["checks"][0]["expected_sha256"] != run["result"]["checks"][0]["actual_sha256"]


def test_missing_registry_document_is_failure(lab):
    workspace, store = lab
    (workspace.root / "source.txt").unlink()
    assert run_check(workspace, store, "registry")["status"] == "failed"


def test_changed_plan_input_rejected(lab):
    workspace, _ = lab
    resolved = plan(workspace, "source")
    (workspace.root / "manifest.json").write_text('{"files": []}')
    with pytest.raises(ValueError, match="Input changed"):
        validate_plan_inputs(workspace, resolved)


def test_path_escape_and_symlink_rejected(lab, tmp_path):
    workspace, _ = lab
    external = tmp_path.parent / (tmp_path.name + "-outside.txt")
    external.write_text("outside")
    (workspace.root / "escape").symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        inside(workspace.root, "escape")
    with pytest.raises(ValueError, match="escapes"):
        artifact_record(workspace, str(external), "receipt", "reported-unreviewed")


def test_manifest_cannot_read_parent_directory(lab):
    workspace, _ = lab
    (workspace.root / "manifest.json").write_text(json.dumps({"files": [
        {"path": "../outside", "sha256": "a" * 64}
    ]}))
    result = manifest_audit(workspace, "manifest.json")
    assert not result["passed"]
    assert "escapes" in result["checks"][0]["error"]


def test_empty_manifest_does_not_pass(lab):
    workspace, store = lab
    (workspace.root / "manifest.json").write_text('{"files": []}')
    with pytest.raises(ValueError, match="nonempty"):
        run_check(workspace, store, "source")
    assert store.runs()[0]["status"] == "failed"


def test_artifact_drift_is_detected(lab):
    workspace, store = lab
    run = run_check(workspace, store, "registry")
    (workspace.root / run["artifacts"][0]["path"]).write_text("corrupt")
    assert not verify_artifacts(workspace, store.run(run["id"]))["passed"]


def test_external_evidence_is_recorded_not_promoted(lab, capsys):
    workspace, store = lab
    code = main(["--workspace", str(workspace.root), "--json", "evidence", "--project", "lab",
                 "--file", "source.txt", "--label", "historical receipt"])
    assert code == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "recorded"
    assert run["kind"] == "external-evidence"
    assert run["artifacts"][0]["evidence_status"] == "reported-unreviewed"
    with pytest.raises(ValueError):
        store.transition(run["id"], "passed")


def test_gpu_run_cannot_launch_from_local_cli(lab):
    workspace, store = lab
    run = run_check(workspace, store, "rl")
    assert run["status"] == "blocked"
    assert not run["plan"]["runnable"]
    assert "GPU launch" in run["result"]["blockers"][0]


def test_prime_pin_and_dirty_checkout_are_reported(lab):
    workspace, _ = lab
    checkout = workspace.root / "upstream"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "rl.toml").write_text('max_steps = 2\n[model]\nname = "test-only"\n')
    subprocess.run(["git", "-C", str(checkout), "add", "rl.toml"], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    result = plan(workspace, "rl", checkout)
    assert "Checkout revision does not match recipe pin" in result["blockers"]
    assert result["native_dry_run_command"][-1] == "--dry-run"
    (checkout / "rl.toml").write_text("max_steps = 3\n")
    assert plan(workspace, "rl", checkout)["tracked_changes"]


def test_project_edits_do_not_rewrite_original_inventory(lab):
    workspace, store = lab
    before = workspace.registry.read_bytes()
    store.update_project("lab", {"lifecycle": "active", "owner": "test-owner"})
    store.update_project("lab", {"next_gate": "review results"})
    assert workspace.registry.read_bytes() == before
    assert store.overrides()["lab"]["owner"] == "test-owner"
    assert store.overrides()["lab"]["next_gate"] == "review results"


def test_plan_snapshot_is_immutable_and_run_ids_are_unique(lab):
    workspace, store = lab
    resolved = plan(workspace, "registry")
    first = store.create(resolved)
    second = store.create(resolved)
    resolved["claim"] = "mutated"
    assert first != second
    assert store.run(first)["plan"]["claim"] != "mutated"
    with pytest.raises(ValueError):
        store.transition(first, "passed")


def test_cli_returns_nonzero_for_blocked_execution(lab, capsys):
    workspace, _ = lab
    assert main(["--workspace", str(workspace.root), "run", "rl"]) == 2
    assert "blocked" in capsys.readouterr().out


def test_grouped_workspace_uses_private_state_and_canonical_paths(lab):
    workspace, store = lab
    store.db.close()
    root = workspace.root
    (root / "platform").mkdir()
    (root / "private").mkdir()
    (root / "btl-lab").rename(root / "platform/btl-lab")
    (root / "platform/btl-lab/projects.json").rename(root / "private/projects.json")
    (root / "platform/btl-lab/.state").rename(root / "private/state")
    (root / ".btl-workspace.json").write_text(json.dumps({
        "schema_version": 1, "lab": "platform/btl-lab",
        "registry": "private/projects.json", "state": "private/state",
    }))
    grouped = Workspace(root)
    assert grouped.lab == root / "platform/btl-lab"
    assert grouped.registry == root / "private/projects.json"
    assert grouped.state == root / "private/state"
    resolved = plan(grouped, "registry")
    assert resolved["inputs"][0]["path"] == "private/projects.json"


def test_layout_marker_cannot_redirect_state_outside_workspace(lab):
    workspace, _ = lab
    (workspace.root / ".btl-workspace.json").write_text(json.dumps({
        "schema_version": 1, "lab": "btl-lab", "registry": "btl-lab/projects.json",
        "state": "../outside-state",
    }))
    with pytest.raises(ValueError, match="escapes"):
        Workspace(workspace.root)
