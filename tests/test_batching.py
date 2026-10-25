from types import SimpleNamespace

import pytest
import torch

from decisionvla.batching import mean_sample_loss, pack_observations
from decisionvla.model import action_logits


def test_right_padding_preserves_images_and_masks():
    rows = [{"input_ids": torch.tensor([[3, 4, 5]]),
             "attention_mask": torch.ones(1, 3, dtype=torch.long),
             "mm_token_type_ids": torch.tensor([[1, 1, 0]]),
             "pixel_values": torch.ones(1, 2, 3)},
            {"input_ids": torch.tensor([[6, 7]]),
             "attention_mask": torch.ones(1, 2, dtype=torch.long),
             "mm_token_type_ids": torch.tensor([[1, 0]]),
             "pixel_values": torch.zeros(1, 2, 3)}]
    packed = pack_observations(rows, 99)
    assert packed["input_ids"].tolist() == [[3, 4, 5], [6, 7, 99]]
    assert packed["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert packed["mm_token_type_ids"].tolist() == [[1, 1, 0], [1, 0, 0]]
    assert torch.equal(packed["pixel_values"], torch.cat([r["pixel_values"] for r in rows]))


def test_equal_observation_weight_and_gradient_match_sequential():
    logits = torch.randn(2, 7, 4, requires_grad=True)
    targets = torch.randint(4, (2, 7))
    masks = torch.tensor([[True] + [False] * 6, [True] * 7])
    expected = sum(torch.nn.functional.cross_entropy(logits[i][masks[i]], targets[i][masks[i]])
                   for i in range(2)) / 2
    actual = mean_sample_loss(logits, targets, masks)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])
    with pytest.raises(ValueError):
        mean_sample_loss(logits, targets, torch.zeros_like(masks))


def test_decoder_masks_padding_and_uses_unpadded_positions():
    recorded = {}

    class Decoder:
        embed_tokens = SimpleNamespace(action_weight=lambda: torch.ones(4, 2))

        def __call__(self, **kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(last_hidden_state=torch.ones(2, 8, 2))

    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(pad_token_id=0)),
                            model=SimpleNamespace(decoder=Decoder()), final_logit_softcapping=30)
    cache = SimpleNamespace(get_seq_length=lambda: 3)
    config = {"horizon": 1, "action_dim": 7, "canvas_length": 8, "bins": 4, "action_token_start": 20}
    output = action_logits(model, cache, torch.zeros(2, 7, dtype=torch.long), config,
                           prefix_attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert recorded["decoder_attention_mask"][1].tolist() == [True, True, False] + [True] * 7 + [False]
    assert recorded["decoder_position_ids"].tolist() == [list(range(3, 11)), list(range(2, 10))]
    assert output.shape == (2, 7, 4)
    assert torch.isneginf(output[:, 6, 1:3]).all()
