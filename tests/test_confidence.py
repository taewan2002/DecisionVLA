import numpy as np
import pytest

from decisionvla.analyze_confidence import association, auc, within_task_auc
from decisionvla.codec import ActionCodec
from decisionvla.confidence import summarize, validate_diagnostics


def uniform():
    p = np.full((8, 7, 256), 1 / 256)
    p[:, 6] = 0
    p[:, 6, [0, 255]] = 0.5
    return p


def test_uniform_and_point_mass_endpoints():
    value = summarize(uniform())
    for key in ["choice_confidence", "top2_margin", "ordered_confidence"]:
        np.testing.assert_allclose(value["metrics"][key], 0)
    np.testing.assert_allclose(value["metrics"]["normalized_entropy"], 1)
    ids = np.full((8, 7), 50)
    ids[:, 6] = 255
    value = summarize(np.eye(256)[ids])
    assert value["chunk_confidence"] == 1
    np.testing.assert_allclose(value["metrics"]["std"], 0)
    validate_diagnostics(value, ActionCodec(256).decode(ids))


def test_adjacent_ambiguity_has_less_distance_than_opposite_modes():
    near, far = uniform(), uniform()
    near[:, :6] = far[:, :6] = 0
    near[:, :6, 127:129] = 0.5
    far[:, :6, [0, 255]] = 0.5
    a, b = summarize(near), summarize(far)
    np.testing.assert_equal(a["metrics"]["selected_probability"], b["metrics"]["selected_probability"])
    assert a["chunk_confidence"] > b["chunk_confidence"] + 0.8


def test_reject_malformed_or_inconsistent_probabilities():
    for invalid in [np.zeros((8, 7, 256)), np.full((8, 7, 256), np.nan), np.ones((1, 256))]:
        with pytest.raises(ValueError):
            summarize(invalid)
    invalid = uniform()
    invalid[0, 6, 0] -= 0.1
    invalid[0, 6, 5] = 0.1
    with pytest.raises(ValueError):
        summarize(invalid)
    value = summarize(uniform())
    with pytest.raises(ValueError):
        validate_diagnostics(value, np.ones((8, 7)))


def test_auc_ties_direction_and_single_class():
    assert auc([0, 1], [0, 1]) == 1
    assert auc([0, 1], [1, 0]) == 0
    assert auc([0, 1], [1, 1]) == 0.5
    assert auc([1, 1], [0, 1]) is None
    # Task difficulty alone gives perfect pooled AUC, but no within-task evidence.
    assert within_task_auc(np.array([0, 0, 1, 1], bool), np.array([0, 0, 1, 1]),
                           np.array([0, 0, 1, 1])) is None


def test_failure_analysis_handles_undefined_and_reproducible_bootstrap():
    rows = [{"task_id": i // 4, "failure": i % 2, "first_query_uncertainty": float(i % 2),
             "first_query_entropy": float(i % 2), "first_query_low_probability": float(i % 2)} for i in range(8)]
    a = association(rows, iterations=40)
    assert a == association(rows, iterations=40)
    assert a["primary_within_task_auroc"] == 1 and a["primary_ci95"] == [1, 1]
    for row in rows:
        row["failure"] = 1
    assert association(rows, iterations=5)["primary_within_task_auroc"] is None


def test_confidence_trace_records_full_distribution_and_executed_actions():
    import json

    from test_rollout import CONFIG, FakeEnvironment

    from decisionvla.rollout import run_episode
    from decisionvla.rpc import MAX_MESSAGE

    class Policy:
        def predict_with_confidence(self, image, instruction, steps, seed, verify_parity=False):
            value = summarize(uniform())
            return ActionCodec(256).decode(np.asarray(value["action_bins"])), 0.01, value

    result, trace = run_episode(FakeEnvironment(), [0], Policy(), "task",
                                {**CONFIG, "record_confidence": True}, 0, 0)
    assert result["policy_queries"] == 2
    assert trace["probabilities"].shape == (2, 8, 7, 256)
    assert trace["ordered_confidence"].shape == (2, 8, 7)
    assert len(json.dumps(summarize(uniform())).encode()) < MAX_MESSAGE
    np.testing.assert_array_equal(trace["actions"], trace["chunks"].reshape(-1, 7)[:10])
