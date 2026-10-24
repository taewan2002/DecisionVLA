import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from decisionvla.adapters import LoRALinear
from decisionvla.codec import ActionCodec
from decisionvla.data import LiberoDataset
from decisionvla.model import corrupt_actions, denoising_loss


def test_control_quantization_and_gripper():
    codec = ActionCodec()
    actions = np.random.default_rng(42).uniform(-1, 1, (100, 7))
    actions[:, 6] = np.where(actions[:, 6] > 0, 1, -1)
    reconstructed = codec.decode(codec.encode(actions))
    assert np.max(np.abs(actions[:, :6] - reconstructed[:, :6])) <= 1 / 255 + 1e-6
    np.testing.assert_array_equal(actions[:, 6], reconstructed[:, 6])
    with pytest.raises(ValueError):
        codec.encode(np.full((8, 7), np.nan))


def test_episode_split_post_action_alignment_and_tail(tmp_path):
    config = json.load(open("configs/libero_spatial.json"))
    path = tmp_path / config["suite"]
    path.mkdir()
    with h5py.File(path / "test_task_demo.hdf5", "w") as f:
        data = f.create_group("data")
        for i in [0, 9]:
            group = data.create_group(f"demo_{i}")
            actions = np.zeros((4, 7), np.float32)
            actions[:, 0] = np.arange(4) * 0.1
            group.create_dataset("actions", data=actions)
            group.create_dataset("obs/agentview_rgb", data=np.zeros((4, 8, 8, 3), np.uint8))
    train = LiberoDataset(tmp_path, config)
    val = LiberoDataset(tmp_path, config, "validation")
    assert len(train) == len(val) == 3
    assert train.episodes[0][1] == "demo_0" and val.episodes[0][1] == "demo_9"
    np.testing.assert_allclose(train[0]["actions"][:3, 0], [0.1, 0.2, 0.3])
    assert train[0]["valid"].sum() == 21
    assert train[2]["valid"].sum() == 7
    train.close()
    val.close()


def test_lora_initial_parity_and_shared_gradient():
    torch.manual_seed(1)
    base = nn.Linear(8, 12, bias=False).requires_grad_(False)
    adapter = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 8)
    torch.testing.assert_close(adapter(x), base(x), rtol=0, atol=0)
    (adapter(x).sum() + adapter(x * 2).sum()).backward()
    assert adapter.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None
    with torch.no_grad():
        adapter.lora_B.normal_()
    merged = base.weight + adapter.lora_B @ adapter.lora_A * adapter.scale
    torch.testing.assert_close(adapter(x), torch.nn.functional.linear(x, merged))


def test_noise_and_padding_loss_are_well_defined():
    config = json.load(open("configs/libero_spatial.json"))
    clean = torch.full((1, 56), 128, dtype=torch.long)
    clean[:, 6::7] = 0
    generator = torch.Generator().manual_seed(1)
    noisy, rate = corrupt_actions(clean, config, generator)
    assert 0 <= rate.item() <= 1
    assert set(noisy[:, 6::7].flatten().tolist()) <= {0, 255}
    logits = torch.randn(1, 56, 256, requires_grad=True)
    valid = torch.arange(56)[None] < 49
    loss = denoising_loss(logits, clean, valid)
    loss.backward()
    assert logits.grad[:, 49:].abs().sum() == 0
    assert logits.grad[:, :49].abs().sum() > 0
