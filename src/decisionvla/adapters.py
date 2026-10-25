"""Shared attention, dense MLP and routed-expert LoRA adapters."""
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


class LoRAExperts(nn.Module):
    """Per-expert LoRA without materializing a dense update of the expert weights.

    The fused gate/up projection shares one rank-r input factor. Every expert
    has independent factors; encoder and decoder share this entire module.
    Routing and the frozen base grouped GEMMs follow the pinned Transformers
    implementation. Expert parallelism and biased experts are not supported.
    """

    def __init__(self, base, rank, alpha, kernel="torch_grouped_mm"):
        super().__init__()
        if (base.has_bias or base.is_transposed or not base.is_concatenated
                or base.has_post_expert_norm or base._is_expert_parallel):
            raise ValueError("Unsupported expert layout for DecisionVLA LoRA")
        self.base = base
        self.scale = alpha / rank
        experts, twice_intermediate, hidden = base.gate_up_proj.shape
        intermediate = twice_intermediate // 2
        if base.down_proj.shape != (experts, hidden, intermediate):
            raise ValueError("Unexpected expert projection shapes")
        device = base.gate_up_proj.device
        from .grouped_gemm import check_expert_kernel

        check_expert_kernel(kernel, device)
        self.kernel = kernel
        self.gate_up_A = nn.Parameter(torch.empty(experts, rank, hidden, device=device))
        self.gate_up_B = nn.Parameter(torch.zeros(experts, twice_intermediate, rank, device=device))
        self.down_A = nn.Parameter(torch.empty(experts, rank, intermediate, device=device))
        self.down_B = nn.Parameter(torch.zeros(experts, hidden, rank, device=device))
        # Each expert is a separate linear map; 3D kaiming_uniform_ would count
        # the rank axis in fan_in and incorrectly shrink the initialization.
        nn.init.uniform_(self.gate_up_A, -hidden**-0.5, hidden**-0.5)
        nn.init.uniform_(self.down_A, -intermediate**-0.5, intermediate**-0.5)

    def _linear(self, x, weight, offsets, *, frozen_base=False):
        from .grouped_gemm import grouped_linear

        kernel = self.kernel
        if kernel == "triton_sm120_lora":
            kernel = "torch_grouped_mm" if frozen_base else "triton_sm120"
        return grouped_linear(x, weight.to(x.dtype), offsets, kernel)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        count, width = hidden_states.shape
        top_k = top_k_index.shape[-1]
        expert_ids, order = torch.sort(top_k_index.reshape(-1))
        x = hidden_states[order // top_k]
        histc_input = expert_ids.int() if x.device.type == "cuda" else expert_ids.float()
        counts = torch.histc(histc_input, bins=self.base.num_experts,
                            min=0, max=self.base.num_experts - 1)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        gate_up = self._linear(x, self.base.gate_up_proj, offsets, frozen_base=True)
        low = self._linear(x, self.gate_up_A, offsets)
        gate_up = gate_up + self._linear(low, self.gate_up_B, offsets) * self.scale
        activated = self.base._apply_gate(gate_up)
        down = self._linear(activated, self.base.down_proj, offsets, frozen_base=True)
        low = self._linear(activated, self.down_A, offsets)
        down = down + self._linear(low, self.down_B, offsets) * self.scale
        weighted = down * top_k_weights.reshape(-1)[order, None]
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return weighted[inverse].view(count, top_k, width).sum(dim=1).to(hidden_states.dtype)


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
    shared, shared_mlp, shared_experts = [], [], []
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
        for target in config.get("lora_mlp_targets", []):
            left, right = getattr(enc.mlp, target), getattr(dec.mlp, target)
            if left.weight is not right.weight:
                raise ValueError(f"Unshared MLP weight at {layer_index}/{target}")
            adapter = LoRALinear(left, config["lora_rank"], config["lora_alpha"])
            setattr(enc.mlp, target, adapter)
            setattr(dec.mlp, target, adapter)
            shared_mlp.append(f"layers.{layer_index}.mlp.{target}")
        if config.get("lora_experts", False):
            if (enc.experts.gate_up_proj is not dec.experts.gate_up_proj
                    or enc.experts.down_proj is not dec.experts.down_proj):
                raise ValueError(f"Unshared expert weights at {layer_index}")
            adapter = LoRAExperts(enc.experts, config["lora_rank"], config["lora_alpha"],
                                 config.get("expert_kernel", "torch_grouped_mm"))
            enc.experts = dec.experts = adapter
            shared_experts.append(f"layers.{layer_index}.experts")
    if encoder.embed_tokens.weight is not decoder.embed_tokens.weight:
        raise ValueError("Encoder/decoder embeddings must be tied")
    embedding = ActionEmbedding(decoder.embed_tokens, config["action_token_start"], config["bins"])
    embedding.delta.requires_grad_(config["train_action_embeddings"])
    encoder.embed_tokens = decoder.embed_tokens = embedding
    counts = {family: sum(p.numel() for name, p in model.named_parameters()
                         if p.requires_grad and match(name)) for family, match in {
        "attention": lambda n: ".self_attn." in n,
        "mlp": lambda n: ".mlp." in n,
        "experts": lambda n: ".experts." in n,
        "action_embeddings": lambda n: n.endswith("embed_tokens.delta"),
    }.items()}
    return {"expert_kernel": config.get("expert_kernel", "torch_grouped_mm"),
            "shared_attention_modules": shared, "shared_mlp_modules": shared_mlp,
            "shared_expert_modules": shared_experts, "trainable_by_family": counts,
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
