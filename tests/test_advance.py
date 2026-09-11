import json
import sys
from pathlib import Path

import pytest

from btl_lab.advance import execute_advance
from btl_train.advance_local import (
    action_token_ids,
    contract_hash,
    group_advantages,
    make_tasks,
    reward,
    tree_zip_map,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {" A": [10], " B": [11]}[text]


def test_bandit_tasks_are_source_disjoint_and_deterministic():
    train = make_tasks("train", 12)
    evaluation = make_tasks("evaluation", 12)
    assert {x["id"] for x in train}.isdisjoint({x["id"] for x in evaluation})
    assert {x["source_id"] for x in train}.isdisjoint({x["source_id"] for x in evaluation})
    assert make_tasks("train", 12) == train


def test_action_tokens_are_single_and_distinct():
    assert action_token_ids(FakeTokenizer()) == [10, 11]


def test_group_advantage_is_centered_and_handles_dead_groups():
    values = group_advantages([0.0, 1.0, 0.0, 1.0])
    assert abs(sum(values)) < 1e-6
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        group_advantages([1.0])


def test_reward_is_binary_and_verifier_owned():
    task = make_tasks("train", 1)[0]
    assert reward(task, task["expected_action"]) == 1.0
    assert reward(task, 1 - task["expected_action"]) == 0.0
    with pytest.raises(ValueError):
        reward(task, 2)


def test_gradient_tree_addition_is_shape_safe():
    assert tree_zip_map(lambda a, b: a + b, {"a": 1, "b": [2]}, {"a": 3, "b": [4]}) == {"a": 4, "b": [6]}
    with pytest.raises(ValueError):
        tree_zip_map(lambda a, b: a + b, {"a": 1}, {"b": 2})


def test_contract_hash_ignores_dict_insertion_order():
    assert contract_hash({"b": 2, "a": 1}) == contract_hash({"a": 1, "b": 2})


def test_local_command_rejects_missing_mlx_environment(lab, monkeypatch):
    workspace, store = lab
    model = workspace.root / "model"
    model.mkdir()
    with pytest.raises(ValueError, match="MLX interpreter"):
        execute_advance(
            type("Args", (), {"action": "local", "model": model, "python": Path("/missing/python"),
                               "steps": 2, "group_size": 4, "max_seconds": 10, "out": None,
                               "resume": None, "experiment": None})(), workspace, store
        )


def test_split_target_changes_the_frozen_protocol():
    assert contract_hash({"steps": 6}) != contract_hash({"steps": 3})
