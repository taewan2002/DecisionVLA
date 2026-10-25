"""Numerical decision/compiler integration; tiny real native Transformer on CPU."""
import json
import threading

import numpy as np
import pytest
import torch
import transformers

if transformers.__version__ != "5.11.0":
    pytest.skip("Use the native reference runtime", allow_module_level=True)

from test_native import tiny as native_tiny_factory

from decisionvla.decision_canvas import (
    DecisionCanvas,
    prepare_decision_observation,
    robot_action_probabilities,
    robot_option_targets,
)
from decisionvla.native_policy import NativeDecisionVLA
from decisionvla.native_sampling import generate_actions
from decisionvla.native_training import native_training_loss, prepare_robot_example
from decisionvla.typed_decisions import DecisionSchema, robot_schema


@pytest.fixture
def tiny():
    return native_tiny_factory.__wrapped__()


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_question_readout_leaves_native_sampler_and_self_conditioning_exact(tiny, passes):
    model, original, inputs = tiny
    labels = ((48, 63), (50, 51, 52), (49, 54), (60, 61), (48, 50, 53, 56), (63, 48))
    canvas = DecisionCanvas(original.template, original.slots, 64, 48, 16, action_dim=1, label_ids=labels)
    trace_a, trace_b = [], []
    generate_actions(model, inputs, original, passes, 78, trace=trace_a)
    result = generate_actions(model, inputs, canvas, passes, 78, trace=trace_b)
    for a, b in zip(trace_a, trace_b, strict=True):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    p = result["readout"]["probabilities"][0]
    for row, ids in zip(p, labels, strict=True):
        assert row[len(ids):].count_nonzero() == 0
        torch.testing.assert_close(row.sum(), torch.tensor(1.))
    targets = torch.tensor([[1, 2, 0, 1, 3, 0]])
    assert canvas.targets(targets)[0, list(canvas.slots)].tolist() == [63, 52, 49, 61, 56, 63]
    # A first pass has temperature .8; verify an independent conditional softmax.
    raw = trace_b[-1]["processed_logits"][0] * (0.4 + 0.4 / passes)
    for i, ids in enumerate(labels):
        torch.testing.assert_close(p[i, :len(ids)], raw[canvas.slots[i], list(ids)].softmax(-1))
    loss, _ = native_training_loss(model, inputs, targets, torch.ones_like(targets, dtype=torch.bool),
                                   canvas, torch.Generator().manual_seed(4), full_noise_probability=1)
    assert torch.isfinite(loss)


def test_robot_mapping_ties_and_target_roundtrip():
    options = np.zeros((56, 256), dtype=np.float32)
    options[:, 0] = 1
    options[6::7, 0] = .5
    options[6::7, 1] = .5
    p = robot_action_probabilities(options)
    assert np.all(p[:, 6, 255] == .5) and np.all(p.argmax(-1)[:, 6] == 0)
    bins = torch.zeros(1, 56, dtype=torch.long)
    bins[:, 6::14] = 255
    targets = robot_option_targets(bins)
    assert targets[0, 6::7].tolist() == [0, 1] * 4
    assert np.allclose(p.sum(-1), 1)


class Processor:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return {"input_ids": torch.tensor([[2, 13, 14]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_training_and_inference_use_identical_question_prefix():
    # Bindings of the production robot schema, without loading model weights.
    schema = robot_schema()
    labels = tuple((262143, 261888) if i % 7 == 6 else tuple(range(261888, 262144)) for i in range(56))
    canvas = type("Canvas", (), {"label_ids": labels})()
    sample = {"image": object(), "task": "lift the bowl", "tokens": np.zeros(56, dtype=np.int64),
              "valid": np.ones(56, dtype=bool)}
    processor = Processor()
    prepared, targets, valid = prepare_robot_example(processor, sample, canvas)
    predicted = prepare_decision_observation(processor, sample["image"], sample["task"], schema, canvas)
    assert processor.messages[0] == processor.messages[1]
    torch.testing.assert_close(prepared["input_ids"], predicted["input_ids"], rtol=0, atol=0)
    assert targets.shape == valid.shape == (1, 56) and targets[0, 6] == 1


def test_policy_typed_custom_path_uses_native_output_and_never_emits_actions(tiny):
    model, original, _ = tiny
    schema = DecisionSchema.parse([
        {"type": "predicate", "instructions": "Should the command be positive?", "name": "positive"},
        {"type": "choice", "instructions": "Choose direction", "choices": [{"value": "left"}, {"value": "right"}]},
        {"type": "score", "instructions": "Rate height", "levels": [{"label": "low"}, {"label": "high"}]},
    ])
    canvas = DecisionCanvas(original.template, original.slots[:3], 64, 48, 16, action_dim=1,
                            label_ids=((48, 63), (48, 49), (50, 51)))
    policy = NativeDecisionVLA.__new__(NativeDecisionVLA)
    policy.model, policy.processor = model, Processor()
    policy.questions, policy.canvas = robot_schema(), canvas
    policy.device, policy.config, policy.lock = torch.device("cpu"), {"visual_tokens": 280}, threading.Lock()
    with pytest.raises(ValueError, match="unvalidated"):
        policy.decide(object(), "lift", questions=schema.as_list())
    result = policy._infer(object(), "lift", schema, canvas, 2, 42, typed=True)
    assert "actions" not in result
    assert len(result["decisions"]["answers"]) == 3
    assert result["decisions"]["semantic_validation"] == "unvalidated_custom_questions"
    assert len(result["diagnostics"]["allowed_mass"]) == 3
    json.dumps(result, allow_nan=False)
    prompt = policy.processor.messages[0][0]["content"][1]["text"]
    assert "Rate height" in prompt and "Choose direction" in prompt


def test_robot_typed_answers_actions_and_observer_parity(monkeypatch):
    import decisionvla.native_policy as module
    from decisionvla.codec import ActionCodec
    from decisionvla.confidence import validate_diagnostics

    p = torch.zeros(1, 56, 256)
    p[:, :, 23] = 1
    p[:, 6::7] = 0
    p[:, 6::7, 0] = .7
    p[:, 6::7, 1] = .3
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        return {"readout": {"probabilities": p, "bins": p.argmax(-1),
                           "allowed_mass": torch.full((1, 56), .01),
                           "allowed_log_mass": torch.full((1, 56), float(np.log(.01)))},
                "generation_config": {"sampler_config": {"entropy_bound": .1}}, "trajectory": []}

    labels = tuple((262143, 261888) if i % 7 == 6 else tuple(range(261888, 262144)) for i in range(56))
    canvas = type("Canvas", (), {"label_ids": labels, "sha256": "test-layout"})()
    policy = NativeDecisionVLA.__new__(NativeDecisionVLA)
    policy.model, policy.processor = object(), Processor()
    policy.questions, policy.canvas = robot_schema(), canvas
    policy.device, policy.config, policy.lock = torch.device("cpu"), {"visual_tokens": 280}, threading.Lock()
    monkeypatch.setattr(module, "generate_actions", generate)
    a = policy.predict_with_confidence(object(), "lift")
    b = policy.decide(object(), "lift")
    assert len(calls) == 2  # exactly one generate per call, even with full typed answers
    np.testing.assert_array_equal(a["actions"], b["actions"])
    assert b["decisions"]["answers"][0]["choice"] == "23"
    assert b["decisions"]["answers"][6]["probability"] == pytest.approx(.7)
    expected = np.full((8, 7), 23)
    expected[:, 6] = 255
    np.testing.assert_array_equal(b["actions"], ActionCodec(256).decode(expected))
    validate_diagnostics(b["diagnostics"], b["actions"])
