"""Small shared LoRA adapters; encoder/decoder ties are explicit and checked."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        # FP32 master parameters keep small updates representable; GEMMs use base dtype.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        adapted = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + adapted * self.scale


class ActionEmbedding(nn.Module):
    def __init__(self, base, start, bins):
        super().__init__()
        self.base = base
        self.start = start
        self.bins = bins
        self.delta = nn.Parameter(torch.zeros(bins, base.weight.shape[1], device=base.weight.device))

    @property
    def weight(self):
        # Full soft self-conditioning is deliberately disabled in this pilot.
        return self.base.weight

    @property
    def embed_scale(self):
        return self.base.embed_scale

    def action_weight(self):
        return self.base.weight[self.start:self.start + self.bins] + self.delta.to(self.base.weight.dtype)

    def forward(self, ids):
        action_ids = ids - self.start
        active = (action_ids >= 0) & (action_ids < self.bins)
        update = F.embedding(action_ids.clamp(0, self.bins - 1), self.delta).to(self.base.weight.dtype)
        return self.base(ids) + update * active.unsqueeze(-1) * self.embed_scale


def install_adapters(model, config):
    model.requires_grad_(False)
    encoder = model.model.encoder.language_model
    decoder = model.model.decoder
    shared = []
    for layer_index, (enc, dec) in enumerate(zip(encoder.layers, decoder.layers, strict=True)):
        for target in config["lora_targets"]:
            left, right = getattr(enc.self_attn, target), getattr(dec.self_attn, target)
            if left is None:
                assert right is None
                continue
            if left.weight is not right.weight:
                raise ValueError(f"Unshared base weight at {layer_index}/{target}")
            adapter = LoRALinear(left, config["lora_rank"], config["lora_alpha"])
            setattr(enc.self_attn, target, adapter)
            setattr(dec.self_attn, target, adapter)
            shared.append(f"layers.{layer_index}.self_attn.{target}")
    if encoder.embed_tokens.weight is not decoder.embed_tokens.weight:
        raise ValueError("Encoder/decoder embeddings must be tied")
    embedding = ActionEmbedding(decoder.embed_tokens, config["action_token_start"], config["bins"])
    embedding.delta.requires_grad_(config["train_action_embeddings"])
    encoder.embed_tokens = decoder.embed_tokens = embedding
    return {"shared_attention_modules": shared,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def adapter_state(model):
    return {name: param.detach().float().cpu().contiguous() for name, param in model.named_parameters()
            if param.requires_grad}


def load_adapter_state(model, state):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if state.keys() != expected.keys():
        raise ValueError("Adapter keys do not match this architecture")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name])
