# Copyright 2026 the HuggingFace Team. All rights reserved.
# Modifications: DecisionVLA contributors, 2026.
# SPDX-License-Identifier: Apache-2.0
# NativeExpertsLoRA adapts DiffusionGemmaTextExperts' eager traversal with LoRA.
# See third_party/transformers-LICENSE for the license and docs/third-party.md.
"""Fresh shared LoRA and one tied action-row delta for the native reference.

This correctness reference materializes one effective embedding matrix inside
parametrize.cached(). It is not the old 256-row head or a 5090 memory optimization.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .adapters import LoRALinear


class ActionRows(nn.Module):
    def __init__(self, weight, start, bins):
        super().__init__()
        self.start = start
        self.delta = nn.Parameter(torch.zeros(bins, weight.shape[1], device=weight.device, dtype=torch.float32))

    def forward(self, weight):
        return torch.cat((weight[:self.start],
                          weight[self.start:self.start + len(self.delta)] + self.delta.to(weight.dtype),
                          weight[self.start + len(self.delta):]), dim=0)


class TiedHead(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


class NativeExpertsLoRA(nn.Module):
    """Reference per-expert order, with low-rank additions; router is untouched."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        count, twice, width = base.gate_up_proj.shape
        device = base.gate_up_proj.device
        self.gate_up_A = nn.Parameter(torch.empty(count, rank, width, device=device))
        self.gate_up_B = nn.Parameter(torch.zeros(count, twice, rank, device=device))
        self.down_A = nn.Parameter(torch.empty(count, rank, twice // 2, device=device))
        self.down_B = nn.Parameter(torch.zeros(count, width, rank, device=device))
        nn.init.uniform_(self.gate_up_A, -1 / math.sqrt(width), 1 / math.sqrt(width))
        nn.init.uniform_(self.down_A, -1 / math.sqrt(twice // 2), 1 / math.sqrt(twice // 2))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.base.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for index in hit:
            e = index[0]
            route, token = torch.where(mask[e])
            x = hidden_states[token]
            update = F.linear(F.linear(x, self.gate_up_A[e].to(x.dtype)), self.gate_up_B[e].to(x.dtype))
            gate, up = (F.linear(x, self.base.gate_up_proj[e]) + update * self.scale).chunk(2, -1)
            x = self.base.act_fn(gate) * up
            update = F.linear(F.linear(x, self.down_A[e].to(x.dtype)), self.down_B[e].to(x.dtype))
            x = F.linear(x, self.base.down_proj[e]) + update * self.scale
            x = x * top_k_weights[token, route, None]
            result.index_add_(0, token, x.to(result.dtype))
        return result


def install_native_adapters(model, config):
    if model.config._experts_implementation != "eager":
        raise ValueError("Native reference LoRA requires explicit eager experts; do not mix grouped/eager arithmetic")
    model.requires_grad_(False)
    enc, dec = model.model.encoder.language_model, model.model.decoder
    rank, alpha = config["lora_rank"], config["lora_alpha"]
    for left, right in zip(enc.layers, dec.layers, strict=True):
        for family, targets in (("self_attn", config["lora_targets"]), ("mlp", config["lora_mlp_targets"])):
            for name in targets:
                a, b = getattr(getattr(left, family), name), getattr(getattr(right, family), name)
                if a is None and b is None:
                    continue
                if a.weight is not b.weight:
                    raise ValueError(f"Untied native {family}.{name}")
                shared = LoRALinear(a, rank, alpha)
                setattr(getattr(left, family), name, shared)
                setattr(getattr(right, family), name, shared)
        if config.get("lora_experts"):
            if (left.experts.gate_up_proj is not right.experts.gate_up_proj
                    or left.experts.down_proj is not right.experts.down_proj):
                raise ValueError("Untied native expert weights")
            left.experts = right.experts = NativeExpertsLoRA(left.experts, rank, alpha)
    embedding = dec.embed_tokens
    if enc.embed_tokens.weight is not embedding.weight or model.lm_head.weight is not embedding.weight:
        raise ValueError("Native input/output embeddings must be tied")
    torch.testing.assert_close(enc.embed_tokens.embed_scale, embedding.embed_scale, rtol=0, atol=0)
    parametrize.register_parametrization(embedding, "weight", ActionRows(
        embedding.weight, config["action_token_start"], config["bins"]))
    embedding.parametrizations.weight[0].delta.requires_grad_(config["train_action_embeddings"])
    enc.embed_tokens = dec.embed_tokens = embedding
    model.lm_head = TiedHead(embedding)
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "tied_action_rows": "input_full_output_self_conditioning", "expert_kernel": "native_reference"}
