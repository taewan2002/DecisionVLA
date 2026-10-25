import socket
import struct

import numpy as np
import pytest

from decisionvla.rollout import aggregate, camera_image, run_episode
from decisionvla.rpc import MAX_MESSAGE, receive, send, validate_actions


class FakeEnvironment:
    def __init__(self, success_step=None, done_step=None):
        self.success_step, self.done_step = success_step, done_step

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.steps = 0
        self.executed = []

    def set_init_state(self, state):
        return {"agentview_image": np.zeros((128, 128, 3), dtype=np.uint8)}

    def get_sim_state(self):
        return np.array([self.steps], dtype=np.float64)

    def check_success(self):
        return self.steps == self.success_step

    def step(self, action):
        self.steps += 1
        self.executed.append(action)
        return self.set_init_state(None), 0, self.steps == self.done_step, {}


class FakePolicy:
    def __init__(self):
        self.seeds = []

    def predict(self, image, instruction, steps, seed):
        self.seeds.append(seed)
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 0] = np.arange(8) / 8
        actions[:, 6] = 1
        return actions, 0.01


CONFIG = {"environment_seed": 0, "policy_seed": 42, "settle_steps": 2,
          "max_policy_steps": 10, "execute_chunk_steps": 8, "denoising_steps": 4}


def test_chunk_execution_settling_seeds_and_horizon():
    env, policy = FakeEnvironment(), FakePolicy()
    result, trace = run_episode(env, [0], policy, "task", CONFIG, 3, 7)
    assert result["policy_steps"] == 10 and result["policy_queries"] == 2 and not result["success"]
    assert len(env.executed) == 12 and env.executed[0] == [0, 0, 0, 0, 0, 0, -1]
    assert policy.seeds == [307042, 307043]
    np.testing.assert_array_equal(trace["actions"][:, 0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1]) / 8)
    assert trace["states"].shape == (11, 1)
    assert trace["query_steps"].tolist() == [0, 8]


@pytest.mark.parametrize("success,done,expected", [(5, None, True), (None, 5, False)])
def test_success_is_not_just_done(success, done, expected):
    result, _ = run_episode(FakeEnvironment(success, done), [0], FakePolicy(), "task", CONFIG, 0, 0)
    assert result["success"] is expected and result["policy_steps"] == 3


def test_rotation_control_validation_and_invalid_settling():
    raw = np.zeros((128, 128, 3), np.uint8)
    raw[0, 0] = [1, 2, 3]
    np.testing.assert_array_equal(np.asarray(camera_image({"agentview_image": raw}))[-1, -1], [1, 2, 3])
    for invalid in [np.zeros((7, 7)), np.zeros((8, 7)), np.full((8, 7), np.nan), np.full((8, 7), 2)]:
        with pytest.raises(ValueError):
            validate_actions(invalid)
    with pytest.raises(RuntimeError):
        run_episode(FakeEnvironment(success_step=1), [0], FakePolicy(), "task", CONFIG, 0, 0)


def test_preflight_repeats_identical_inputs_before_execution():
    policy = FakePolicy()
    result, _ = run_episode(FakeEnvironment(), [0], policy, "task",
                            {**CONFIG, "check_repeat_prediction": True}, 0, 0)
    assert result["repeated_prediction_verified"]
    assert policy.seeds == [42, 42, 43]


def test_rpc_roundtrip_limits_and_task_macro():
    left, right = socket.socketpair()
    with left, right:
        send(left, {"seed": 42, "actions": [[-1, 1]]})
        assert receive(right) == {"seed": 42, "actions": [[-1, 1]]}
        left.sendall(struct.pack("!I", MAX_MESSAGE + 1))
        with pytest.raises(ValueError):
            receive(right)
    result = aggregate([{"task_id": 0, "success": True}, {"task_id": 1, "success": False}], [0, 1])
    assert result["successes"] == 1 and result["task_macro_success_rate"] == 0.5
