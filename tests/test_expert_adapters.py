import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations.moe import grouped_mm_experts_forward

from decisionvla.adapters import LoRAExperts, install_adapters
from decisionvla.checkpoints import load_adapter, save_adapter


class Experts(nn.Module):
    def __init__(self, experts=4, hidden=32, intermediate=16):
        super().__init__()
        self.num_experts = experts
        self.has_bias = self.is_transposed = self.has_post_expert_norm = self._is_expert_parallel = False
        self.is_concatenated = self.has_gate = True
        self.gate_up_proj = nn.Parameter(torch.randn(experts, 2 * intermediate, hidden) / hidden**0.5)
        self.down_proj = nn.Parameter(torch.randn(experts, hidden, intermediate) / intermediate**0.5)

    def _apply_gate(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up

    def forward(self, x, indices, weights):
        return grouped_mm_experts_forward(self, x, indices, weights)


def dense_reference(adapter, x, indices, weights):
    gate_up = adapter.base.gate_up_proj + adapter.gate_up_B @ adapter.gate_up_A * adapter.scale
    down = adapter.base.down_proj + adapter.down_B @ adapter.down_A * adapter.scale
    # Independent per-token reference, including repeated and unused experts.
    result = []
    for token, ids, probabilities in zip(x, indices, weights, strict=True):
        routes = [F.linear(adapter.base._apply_gate(F.linear(token, gate_up[e])), down[e]) * p
                  for e, p in zip(ids, probabilities, strict=True)]
        result.append(torch.stack(routes).sum(0))
    return torch.stack(result)


def test_grouped_expert_parity_gradients_and_unused_experts():
    torch.manual_seed(42)
    base = Experts().requires_grad_(False)
    adapter = LoRAExperts(base, rank=4, alpha=8)
    x = torch.randn(5, 32, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0], [2, 0], [2, 1], [0, 2]])
    weights = torch.randn(5, 2).softmax(-1).requires_grad_()
    torch.testing.assert_close(adapter(x, indices, weights), base(x, indices, weights), rtol=0, atol=0)
    adapter(x, indices, weights).square().sum().backward()
    assert adapter.gate_up_B.grad[:3].abs().sum() > 0
    assert adapter.down_B.grad[:3].abs().sum() > 0
    assert adapter.gate_up_B.grad[3].abs().sum() == 0
    assert base.gate_up_proj.grad is base.down_proj.grad is None
    with torch.no_grad():
        adapter.gate_up_B.normal_(std=0.02)
        adapter.down_B.normal_(std=0.02)
    actual = adapter(x, indices, weights)
    expected = dense_reference(adapter, x, indices, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    parameters = [x, weights, adapter.gate_up_A, adapter.gate_up_B, adapter.down_A, adapter.down_B]
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=5e-4, atol=2e-5)


def test_extended_adapters_share_encoder_decoder_and_keep_router_frozen():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
            self.mlp = nn.ModuleDict({key: nn.Linear(32, 32, bias=False)
                                      for key in ["gate_proj", "up_proj", "down_proj"]})
            self.experts = Experts()
            self.router = nn.Linear(32, 4, bias=False)

    enc, dec = Layer(), Layer()
    for name, left in enc.named_parameters():
        parent, key = name.rsplit(".", 1)
        setattr(dec.get_submodule(parent), key, left)
    embedding = nn.Embedding(64, 32)
    embedding.register_buffer("embed_scale", torch.tensor(32**0.5))
    encoder = nn.Module()
    encoder.layers = nn.ModuleList([enc])
    encoder.embed_tokens = embedding
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([dec])
    decoder.embed_tokens = embedding
    model = nn.Module()
    model.model = nn.Module()
    model.model.encoder = nn.Module()
    model.model.encoder.language_model = encoder
    model.model.decoder = decoder
    config = {"lora_rank": 4, "lora_alpha": 8, "lora_targets": ["q_proj"],
              "lora_mlp_targets": ["gate_proj", "up_proj", "down_proj"], "lora_experts": True,
              "action_token_start": 48, "bins": 16, "train_action_embeddings": True}
    info = install_adapters(model, config)
    assert enc.self_attn.q_proj is dec.self_attn.q_proj
    assert enc.mlp.gate_proj is dec.mlp.gate_proj
    assert enc.experts is dec.experts
    assert not enc.router.weight.requires_grad
    assert len(info["shared_mlp_modules"]) == 3
    assert len(info["shared_expert_modules"]) == 1
    assert sum(info["trainable_by_family"].values()) == info["trainable_parameters"]


@pytest.mark.parametrize("limit", [100000, 2000])
def test_single_and_sharded_adapter_round_trip(tmp_path, limit):
    torch.manual_seed(1)
    model = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
    model[0].requires_grad_(False)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = save_adapter(model, tmp_path, max_shard_bytes=limit)
    assert (result["shards"] > 1) == (limit == 2000)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_adapter(model, tmp_path)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_sharded_release_excludes_training_state_and_has_valid_hashes(tmp_path):
    from decisionvla.checkpoints import file_sha256

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
    save_adapter(model, checkpoint, max_shard_bytes=2000)
    for name in ("adapter_config.json", "training_metadata.json"):
        (checkpoint / name).write_text("{}")
    (checkpoint / "training_state.pt").write_bytes(b"private optimizer state")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("Test model card")
    output = tmp_path / "release"
    completed = subprocess.run([sys.executable, "scripts/package_adapter.py", str(checkpoint),
                    "--output", str(output), "--model-card", str(card), "--sharded-release"],
                   check=True, capture_output=True, text=True)
    manifest = json.loads(completed.stdout)
    assert not (output / "training_metadata.json").exists()
    assert not (output / "release-manifest.json").exists()
    assert "training_state.pt" not in manifest["files"]
    assert "adapter.safetensors.index.json" in manifest["files"]
    assert "sha256.json" in manifest["files"]
    # The public evaluator reads this flat mapping, not release-manifest.json.
    server_hashes = json.loads((output / "sha256.json").read_text())
    assert "training_state.pt" not in server_hashes
    assert server_hashes["adapter_config.json"] == file_sha256(output / "adapter_config.json")
    for shard in output.glob("*.safetensors"):
        assert server_hashes[shard.name] == file_sha256(shard)
    for name, expected in manifest["files"].items():
        assert file_sha256(output / name) == expected["sha256"]
