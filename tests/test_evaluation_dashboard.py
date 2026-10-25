import json

import numpy as np
import pytest

from decisionvla.dashboard.evaluation import EvaluationRecorder, evaluation_view
from decisionvla.rollout import run_episode
from test_dashboard import prediction
from test_rollout import CONFIG, FakeEnvironment


class ConfidencePolicy:
    def __init__(self):
        self.calls = []

    def predict_with_confidence(self, image, instruction, steps, seed, **kwargs):
        self.calls.append((image.tobytes(), instruction, steps, seed, kwargs))
        value = prediction()
        return value["actions"], .188, value["diagnostics"]


CFG = {**CONFIG, "suite": "libero_spatial", "record_confidence": True}


def test_observed_episode_preserves_actions_seeds_states_probabilities_and_rng(tmp_path):
    plain, observed = ConfidencePolicy(), ConfidencePolicy()
    rng_before = np.random.get_state()
    expected, trace = run_episode(FakeEnvironment(success_step=11), [0], plain, "task", CFG, 2, 3)
    with EvaluationRecorder(tmp_path / "session", {}, expected=1) as observer:
        actual, instrumented = run_episode(FakeEnvironment(success_step=11), [0], observed,
                                           "task", CFG, 2, 3, observer=observer)
    assert actual == expected and plain.calls == observed.calls
    for key in trace:
        if key != "query_seconds":  # Wall-clock latency is measured, not deterministic.
            np.testing.assert_array_equal(trace[key], instrumented[key])
    rng_after = np.random.get_state()
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    saved = json.loads((tmp_path / "session/evaluation.json").read_text())
    assert saved["status"] == "completed" and saved["expected"] == 1
    assert saved["episodes"][0]["success"] is True
    frame = json.loads((tmp_path / "session/frames/000001.json").read_text())
    assert frame["context"]["query"] == 1 and frame["context"]["control_step"] == 8
    assert frame["context"]["episode"] == "libero_spatial/task-02-init-03"
    assert "recorded_success" not in frame["context"]  # No future outcome in a live query.
    assert frame["policy_seconds"] == .188


def test_dashboard_does_not_silently_enable_confidence(tmp_path):
    with EvaluationRecorder(tmp_path / "session", {}) as observer:
        policy = ConfidencePolicy()
        with pytest.raises(ValueError, match="confidence-enabled"):
            run_episode(FakeEnvironment(), [0], policy, "task", CONFIG, 0, 0, observer=observer)
        assert policy.calls == []


def test_telemetry_disk_failure_does_not_change_policy_result(tmp_path, monkeypatch):
    with EvaluationRecorder(tmp_path / "session", {}) as observer:
        def broken():
            raise OSError("simulated telemetry disk failure")

        monkeypatch.setattr(observer, "_save", broken)
        policy = ConfidencePolicy()
        result, _ = run_episode(FakeEnvironment(), [0], policy, "task", CFG, 0, 0, observer=observer)
        assert not result["success"] and len(policy.calls) == 2
        assert observer.recorder.errors == 2


def test_evaluation_errors_are_not_failures_and_suite_ids_do_not_collide(tmp_path):
    folder = tmp_path / "session"
    with pytest.raises(RuntimeError, match="simulator failed"):
        with EvaluationRecorder(folder, {}, expected=3) as observer:
            for suite in ["libero_spatial", "libero_goal"]:
                run_episode(FakeEnvironment(success_step=5), [0], ConfidencePolicy(), "task",
                            {**CFG, "suite": suite}, 0, 0, observer=observer)
            observer.begin("libero_object/task-00-init-00", suite="libero_object", task_id=0,
                           initial_id=0, instruction="task", max_steps=10)
            raise RuntimeError("simulator failed")
    index = json.loads((folder / "index.json").read_text())
    view = evaluation_view(folder, {}, index, {})
    assert (view["completed"], view["successes"], view["failures"], view["errors"]) == (2, 2, 0, 1)
    assert len(view["suites"]) == 3 and view["status"] == "error"


def test_replay_never_substitutes_later_query_for_first_confidence(tmp_path):
    index = {"frames": [
        {"confidence": .99, "context": {"episode": "failed", "query": 8, "recorded_success": False}},
        {"confidence": .5, "context": {"episode": "success", "query": 0, "recorded_success": True}},
        {"confidence": .9, "context": {"episode": "pending", "query": 0}},
    ]}
    view = evaluation_view(tmp_path, {"selection": "Selected examples"}, index, {})
    assert view["scope"] == "recorded_subset" and view["expected"] is None
    assert view["completed"] == 2 and view["successes"] == 1 and view["failures"] == 1
    assert view["episodes"][0]["first_confidence"] is None
    assert view["episodes"][1]["first_confidence"] == .5


def test_episode_list_is_bounded_without_truncating_totals(tmp_path):
    rows = [{"episode": str(i), "suite": "libero_spatial", "status": "completed", "success": i % 2 == 0}
            for i in range(501)]
    (tmp_path / "evaluation.json").write_text(json.dumps({"status": "completed", "expected": 501,
                                                        "active": None, "episodes": rows}))
    view = evaluation_view(tmp_path, {}, {"frames": []}, {})
    assert view["completed"] == 501 and view["successes"] == 251
    assert view["listed_episodes"] == 200 and view["total_episodes"] == 501
