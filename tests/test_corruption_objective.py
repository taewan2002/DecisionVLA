import torch

from decisionvla import model as sampling


def test_legacy_corruption_and_rng_are_unchanged():
    config = {"bins": 256, "full_noise_probability": 0.2}
    target = torch.full((1, 56), 128)
    target[:, 6::7] = 255
    valid = torch.arange(56)[None] < 49
    a, b = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    original, _ = sampling.corrupt_actions(target, config, a)
    actual, mask = sampling.training_corruption(target, valid, config, b)
    assert torch.equal(original, actual) and torch.equal(mask, valid)
    assert torch.equal(a.get_state(), b.get_state())


def test_unchanged_and_padded_tokens_receive_no_target_loss(monkeypatch):
    target = torch.tensor([[3, 4, 5, 6, 7, 8, 255]])
    noisy = torch.tensor([[3, 2, 5, 9, 7, 8, 255]])
    valid = torch.tensor([[True, True, True, False, False, False, False]])
    replaced = torch.tensor([[False, True, False, True, False, False, False]])
    monkeypatch.setattr(sampling, "corrupt_actions", lambda *args, **kwargs: (noisy.clone(), None, replaced))
    actual, mask = sampling.training_corruption(target, valid, {"bins": 256,
                                                "denoising_loss_scope": "resampled_only"}, None)
    assert torch.equal(actual, noisy)
    assert mask.tolist() == [[False, True, False, False, False, False, False]]
    logits = torch.randn(1, 7, 256, requires_grad=True)
    sampling.denoising_loss(logits, target, mask).backward()
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[~mask].abs().sum() == 0


def test_empty_selection_resamples_one_valid_position_without_forcing_a_flip(monkeypatch):
    target = torch.full((1, 7), 128)
    target[:, 6] = 0
    monkeypatch.setattr(sampling, "corrupt_actions", lambda *args, **kwargs:
                        (target.clone(), None, torch.zeros_like(target, dtype=torch.bool)))
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 6] = True
    outcomes = []
    for seed in range(20):
        noisy, mask = sampling.training_corruption(target, valid, {"bins": 256,
                                                   "denoising_loss_scope": "resampled_only"},
                                                   torch.Generator().manual_seed(seed))
        assert mask.sum() == 1 and mask[0, 6]
        torch.testing.assert_close(noisy[:, :6], target[:, :6])
        outcomes.append(int(noisy[0, 6]))
    assert set(outcomes) == {0, 255}


def test_coincidental_noise_matches_remain_supervised_to_prevent_binary_anticopy(monkeypatch):
    target = torch.tensor([[3, 4, 5, 6, 7, 8, 255]])
    selected = torch.ones_like(target, dtype=torch.bool)
    monkeypatch.setattr(sampling, "corrupt_actions", lambda *args, **kwargs: (target.clone(), None, selected))
    _, mask = sampling.training_corruption(target, selected, {"bins": 256,
                                           "denoising_loss_scope": "resampled_only"}, None)
    assert torch.equal(mask, selected)
    logits = torch.randn(1, 7, 256, requires_grad=True)
    sampling.denoising_loss(logits, target, mask).backward()
    assert logits.grad[0, 6, 255] < 0
