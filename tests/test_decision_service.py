import base64
import io

import numpy as np
import pytest
from PIL import Image

from decisionvla.native_serve import handle_request


class Policy:
    metadata = {"decision_schema": "decisionvla.decisions.v1"}

    def __init__(self):
        self.calls = []

    def predict_with_confidence(self, image, instruction, steps, seed, typed=False):
        self.calls.append((steps, seed, typed))
        return {"actions": np.zeros((8, 7)), "diagnostics": {"source": "model"}, "timing": {}}

    def decide(self, image, instruction, **kwargs):
        self.calls.append(kwargs)
        return {"decisions": {"answers": [{"type": "predicate", "probability": .7}]},
                "diagnostics": {"allowed_mass": [.01]}, "timing": {}}


def request(**kwargs):
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    return {"op": "decide", "image_png": base64.b64encode(buffer.getvalue()).decode(),
            "instruction": "lift", **kwargs}


def test_service_returns_ordered_answers_without_fake_actions():
    policy = Policy()
    result = handle_request(policy, request())
    assert result["ok"] and result["answers"][0]["probability"] == .7
    assert "actions" not in result and len(policy.calls) == 1
    assert policy.calls[0]["experimental"] is False


@pytest.mark.parametrize("bad", [{"steps": True}, {"seed": -1}, {"experimental": "true"},
                                {"instruction": ""}, {"op": "other"}, {"extra": 0}])
def test_service_rejects_invalid_requests_before_model(bad):
    policy = Policy()
    with pytest.raises(ValueError):
        handle_request(policy, request(**bad))
    assert not policy.calls
