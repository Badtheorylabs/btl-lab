import json
import os
import sys

import pytest

from btl_lab.cli import main
from btl_train.finetune_data import PACKAGES, audit, check_row, digest, pad_batch, validate_config
from btl_train.finetune_worker import doctor, verify_resume
from btl_train.process import run_process


@pytest.fixture
def sft(lab):
    workspace, store = lab
    root = workspace.root
    model = root / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"unit-test fixture, not real weights")
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    identity = {"id": "test/tokenizer", "revision": "a" * 40}
    (root / "model-manifest.json").write_text(json.dumps({"tokenizer": identity, "files": [
        {"path": p.name, "sha256": digest(p)} for p in model.iterdir()]}))
    (root / "acceptance.json").write_text('{"scope":"test fixture only"}')
    for name, source, tokens in [("train", "train-source", [1, 2, 3]), ("evaluation", "eval-source", [4, 5, 6])]:
        (root / (name + ".jsonl")).write_text(json.dumps({"id": name, "source_id": source,
            "input_ids": tokens, "labels": [-100, -100, tokens[-1]]}) + "\n")
    manifest = {"schema_version": 1, "accepted": True, "tokenizer": identity,
                "acceptance_receipt": {"path": "acceptance.json", "sha256": digest(root / "acceptance.json")}}
    for name in ("train", "evaluation"):
        manifest[name] = {"path": name + ".jsonl", "sha256": digest(root / (name + ".jsonl"))}
    (root / "dataset.json").write_text(json.dumps(manifest))
    config = {"schema_version": 1, "project_id": "lab", "method": "lora", "precision": "bf16",
              "expected_model_type": "qwen3", "model_directory": "model",
              "dataset_manifest": "dataset.json", "dataset_manifest_sha256": digest(root / "dataset.json"),
              "model_manifest": "model-manifest.json", "model_manifest_sha256": digest(root / "model-manifest.json"),
              "max_seq_length": 32, "max_steps": 8, "micro_batch_size": 1, "gradient_accumulation_steps": 2,
              "save_steps": 2, "wall_seconds": 60, "rank": 8, "alpha": 16, "seed": 7,
              "learning_rate": 0.0001, "max_grad_norm": 1.0,
              "target_modules": ["q_proj", "v_proj"], "versions": {p: "0.0.0" for p in PACKAGES}}
    return workspace, store, config


def test_sft_audit_counts_causal_supervision_and_verifies_assets(sft):
    workspace, _, config = sft
    result = audit(workspace.root, config, check_weights=True)
    assert result["weights_verified"]
    assert result["summary"]["train"] == {"rows": 1, "tokens": 3, "supervised_tokens": 1}


def test_padding_preserves_loss_mask():
    batch = pad_batch([{"input_ids": [1, 2], "labels": [-100, 2]},
                       {"input_ids": [3, 4, 5], "labels": [-100, -100, 5]}], 0)
    assert batch["labels"][0] == [-100, 2, -100]
    assert batch["attention_mask"][0] == [1, 1, 0]
    assert batch["labels"][1] == [-100, -100, 5]


@pytest.mark.parametrize("tokens,labels", [([1, 2], [-100, -100]), ([1, 2], [-100, 3]),
                                           ([1, True], [-100, True]), ([1, 2], [1, 2])])
def test_invalid_supervision_rejected(tokens, labels):
    with pytest.raises(ValueError):
        check_row({"id": "r", "source_id": "s", "input_ids": tokens, "labels": labels}, 10)


def test_sequences_are_never_silently_truncated():
    with pytest.raises(ValueError, match="no truncation"):
        check_row({"id": "r", "source_id": "s", "input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, 2)


def test_source_overlap_is_rejected_even_when_tokens_differ(sft):
    workspace, _, config = sft
    root = workspace.root
    p = root / "evaluation.jsonl"
    row = json.loads(p.read_text());row["source_id"] = "train-source";p.write_text(json.dumps(row))
    manifest = json.loads((root / "dataset.json").read_text())
    manifest["evaluation"]["sha256"] = digest(p)
    (root / "dataset.json").write_text(json.dumps(manifest))
    config["dataset_manifest_sha256"] = digest(root / "dataset.json")
    with pytest.raises(ValueError, match="overlap"):
        audit(root, config)


def test_changed_model_weight_fails_full_integrity_check(sft):
    workspace, _, config = sft
    (workspace.root / "model/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed model asset"):
        audit(workspace.root, config, check_weights=True)


def test_hub_download_metadata_is_not_treated_as_a_model_asset(sft):
    workspace, _, config = sft
    cache = workspace.root / "model/.cache/huggingface"
    cache.mkdir(parents=True)
    (cache / "download.json").write_text('{"metadata": true}')
    assert audit(workspace.root, config, check_weights=True)["weights_verified"]


def test_no_implicit_mode_or_version_changes(sft):
    _, _, config = sft
    config["method"] = "full"
    with pytest.raises(ValueError, match="supervised"):
        validate_config(config)
    config["method"] = "lora"
    assert not doctor(config)["ready_for_worker_start"]


def test_cli_records_real_data_check_without_gpu_imports(sft, capsys):
    workspace, store, config = sft
    p = workspace.root / "sft.json";p.write_text(json.dumps(config))
    code = main(["--workspace", str(workspace.root), "--json", "finetune", "check", "sft.json", "--data-only"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["result"]["stage"] == "dataset-check"
    assert "torch" not in sys.modules


def test_subprocess_timeout_is_enforced(tmp_path):
    result = run_process([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
                         tmp_path, tmp_path / "worker.log", 1.0, os.environ.copy(), grace_seconds=0.1)
    assert result["timed_out"] and result["exit_code"] != 0
    assert result["elapsed_seconds"] < 5
    assert "started" in (tmp_path / "worker.log").read_text()


def test_child_failure_is_not_success(tmp_path):
    result = run_process([sys.executable, "-c", "raise SystemExit(7)"], tmp_path,
                         tmp_path / "worker.log", 5, os.environ.copy())
    assert result["exit_code"] == 7 and not result["timed_out"]


def test_resume_checks_contract_and_checkpoint_hashes(tmp_path):
    names = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"]
    for name in names:
        (tmp_path / name).write_text('{"global_step":2}' if name == "trainer_state.json" else "test")
    receipt = {"contract_sha256": "a" * 64, "files": {n: digest(tmp_path / n) for n in names}}
    (tmp_path / "btl-checkpoint.json").write_text(json.dumps(receipt))
    assert verify_resume(tmp_path, "a" * 64) == 2
    (tmp_path / "optimizer.pt").write_text("corrupt")
    with pytest.raises(ValueError, match="Changed checkpoint"):
        verify_resume(tmp_path, "a" * 64)
