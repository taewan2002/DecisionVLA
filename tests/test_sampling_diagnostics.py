import torch

from decisionvla import model as sampling


def test_probability_readout_preserves_sampler_actions_and_rng(monkeypatch):
    config = {"horizon": 8, "action_dim": 7, "bins": 256}
    model = torch.nn.Linear(1, 1)
    calls = []

    def logits(model, cache, noisy, config):
        calls.append(noisy.clone())
        support = torch.arange(256, dtype=torch.float32)
        output = -(support - noisy[..., None]).abs() / 50
        output[:, 6::7, 1:-1] = -torch.inf
        return output

    monkeypatch.setattr(sampling, "action_logits", logits)
    original_rng = torch.Generator().manual_seed(42)
    original = sampling.sample_actions(model, None, config, 4, original_rng)
    originals = calls.copy()
    calls.clear()
    instrumented_rng = torch.Generator().manual_seed(42)
    actions, probabilities = sampling.sample_actions(model, None, config, 4, instrumented_rng,
                                                     return_probabilities=True)
    assert len(calls) == len(originals) == 4
    for a, b in zip(calls, originals):
        assert torch.equal(a, b)
    assert torch.equal(original, actions)
    assert torch.equal(original_rng.get_state(), instrumented_rng.get_state())
    assert torch.equal(actions, probabilities.argmax(-1))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones((1, 56)))


def test_step_observer_preserves_canvas_actions_and_rng_even_if_snapshots_mutated(monkeypatch):
    config = {"horizon": 8, "action_dim": 7, "bins": 256}
    model = torch.nn.Linear(1, 1)
    calls, observed = [], []

    def logits(model, cache, noisy, config):
        calls.append(noisy.clone())
        output = -(torch.arange(256)[None, None] - noisy[..., None]).abs().float() / 50
        output[:, 6::7, 1:-1] = -torch.inf
        return output

    def observer(step, canvas, probabilities):
        observed.append((step, canvas.clone(), probabilities.clone()))
        canvas.zero_()
        probabilities.zero_()

    monkeypatch.setattr(sampling, "action_logits", logits)
    original_rng = torch.Generator().manual_seed(731)
    original = sampling.sample_actions(model, None, config, 4, original_rng)
    original_calls = calls.copy()
    calls.clear()
    traced_rng = torch.Generator().manual_seed(731)
    traced = sampling.sample_actions(model, None, config, 4, traced_rng, step_observer=observer)
    assert len(calls) == len(observed) == 4
    assert torch.equal(original, traced) and torch.equal(original_rng.get_state(), traced_rng.get_state())
    for index, ((step, canvas, probabilities), actual, expected) in enumerate(zip(observed, calls, original_calls)):
        assert step == index and torch.equal(canvas, actual) and torch.equal(actual, expected)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones((1, 56)))
