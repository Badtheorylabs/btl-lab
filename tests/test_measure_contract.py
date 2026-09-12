import json

import pytest

from btl_measure.core import compare, load


def evaluation(path, records, revision="a"):
    path.write_text(json.dumps({"model_id": "m", "model_revision": revision, "taskset_id": "t",
                               "taskset_revision": "v1", "records": records}))
    return load(path)


@pytest.mark.parametrize("record", [{"score": 0, "reward": 1}, {"score": 0.5, "passed": True},
                                     {"score": 1, "passed": False}, {"score": 0.5, "reward": "x"}])
def test_conflicting_scores_are_rejected(tmp_path, record):
    with pytest.raises(ValueError):
        evaluation(tmp_path / "bad.json", [{"id": "x", **record}])


def test_partial_credit_regressions_are_reported(tmp_path):
    base = evaluation(tmp_path / "base.json", [{"id": "x", "score": 0.8}])
    candidate = evaluation(tmp_path / "candidate.json", [{"id": "x", "score": 0.6}], "b")
    assert compare(base, candidate)["regression_ids"] == ["x"]


def test_pairing_rejects_a_changed_source_under_same_item_id(tmp_path):
    base = evaluation(tmp_path / "base.json", [{"id": "x", "score": 1, "source_id": "a"}])
    candidate = evaluation(tmp_path / "candidate.json", [{"id": "x", "score": 1, "source_id": "b"}], "b")
    assert not compare(base, candidate)["comparable"]


def test_equal_score_is_not_an_improvement(tmp_path):
    base = evaluation(tmp_path / "base.json", [{"id": "x", "score": 1}])
    candidate = evaluation(tmp_path / "candidate.json", [{"id": "x", "score": 1}], "b")
    assert not compare(base, candidate)["meets_declared_threshold"]
