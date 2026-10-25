"""Experimental NVFP4 expert weights with a fused SM120 W4A16 MoE.

Pinned to FlashInfer 0.7.2.post1's prepared-weight API. Expert LoRA is merged
in a separate inference copy. BF16 activations, attention, dense MLP, vision,
router and KV caches are retained. This is not W4A4 Tensor Core inference.
"""

import dataclasses
import gc

import torch
from torch import nn

from decisionvla.adapters import LoRAExperts


@torch.no_grad()
def pack_experts(weight):
    """Standard E2M1 / E4M3 block16 NVFP4, per-expert FP32 global scale."""
    from flashinfer import nvfp4_quantize

    packed, block_scales, global_scales = [], [], []
    for expert in weight:
        expert = expert.to(device="cuda", dtype=torch.bfloat16).contiguous()
        scale = expert.float().abs().amax().clamp_min(1e-8) / (448 * 6)
        codes, blocks = nvfp4_quantize(
            expert, scale.reciprocal(), nvfp4_4over6=None, backend="cute-dsl"
        )
        packed.append(codes.view(torch.uint8))
        block_scales.append(blocks.flatten())
        global_scales.append(scale)
    return torch.stack(packed), torch.stack(block_scales), torch.stack(global_scales)


@torch.no_grad()
def merge_experts(module):
    """Merge BF16 execution factors in FP32, then round the new weight to BF16.

    Separate LoRA GEMMs and merged weights need not be numerically equivalent;
    measure that change explicitly before attributing all error to quantization.
    """
    for weight_name, a_name, b_name in (
        ("gate_up_proj", "gate_up_A", "gate_up_B"),
        ("down_proj", "down_A", "down_B"),
    ):
        weight = getattr(module.base, weight_name)
        a, b = getattr(module, a_name), getattr(module, b_name)
        for e in range(weight.shape[0]):
            delta = b[e].to("cuda", torch.bfloat16).float() @ a[e].to("cuda", torch.bfloat16).float()
            merged = weight[e].cuda().float() + delta * module.scale
            weight[e].copy_(merged.to(device="cpu", dtype=torch.bfloat16))
        # Retain a zero residual only for the optional merge-only reference run.
        b.zero_()


class FusedNVFP4Experts(nn.Module):
    """Prepared buffers only: no second resident copy of the packed weights."""

    def __init__(self, prepared):
        super().__init__()
        self._metadata = {}
        for field in dataclasses.fields(prepared):
            value = getattr(prepared, field.name)
            if isinstance(value, torch.Tensor):
                self.register_buffer(field.name, value)
            else:
                self._metadata[field.name] = value
        self.fast_math = False
        # Atomic FC2 accumulation changes roundoff order between calls. Small
        # changes can flip MoE routes and discrete denoising decisions. Use the
        # existing per-route output plus fixed-order top-k reduction instead.
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel
        from stable_routes import stable_pack

        moe_w4a16_kernel._TC_DECODE_MAX_M = 0
        moe_w4a16_kernel.pack_topk_routes_by_expert = stable_pack

    def prepared(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
            W4A16PackedWeights,
        )

        return W4A16PackedWeights(**self._metadata, **dict(self.named_buffers()))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _launch_sm120_w4a16_moe,
        )

        output = torch.empty_like(hidden_states)
        # The pinned prepared-weight path never dereferences the source arguments.
        return _launch_sm120_w4a16_moe(
            a=hidden_states.contiguous(),
            topk_ids=top_k_index.int().contiguous(),
            topk_weights=top_k_weights.float().contiguous(),
            w1_weight=None,
            w1_weight_sf=None,
            w1_alpha=None,
            w2_weight=None,
            w2_weight_sf=None,
            w2_alpha=None,
            num_experts=self._metadata["num_experts"],
            num_local_experts=self._metadata["num_experts"],
            top_k=top_k_index.shape[-1],
            scatter_output=output,
            activation="gelu_tanh",
            fast_math=self.fast_math,
            _prepared_weights=self.prepared(),
        )


@torch.no_grad()
def quantize_model(model, *, already_merged=False):
    import flashinfer
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )

    if flashinfer.__version__ != "0.7.2.post1":
        raise RuntimeError("The prepared-weight API is pinned to FlashInfer 0.7.2.post1")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This experimental path is validated only on SM120")
    encoder, decoder = model.model.encoder.language_model, model.model.decoder
    count = 0
    for enc, dec in zip(encoder.layers, decoder.layers, strict=True):
        module = enc.experts
        assert module is dec.experts and isinstance(module, LoRAExperts)
        assert all(p.device.type == "cpu" for p in module.parameters())
        if not already_merged:
            merge_experts(module)
        # Transformers stores gate/up; B12x expects up/gate.
        gate, up = module.base.gate_up_proj.chunk(2, dim=1)
        w1 = pack_experts(torch.cat((up, gate), dim=1))
        w2 = pack_experts(module.base.down_proj)
        prepared = prepare_w4a16_packed_weights(
            *w1, *w2, activation="gelu_tanh", params_dtype=torch.bfloat16
        )
        replacement = FusedNVFP4Experts(prepared)
        enc.experts = dec.experts = replacement
        del module, gate, up, w1, w2, prepared
        count += 1
        print(f"fused NVFP4 expert layer {count}/30", flush=True)
    gc.collect()
    model.to("cuda")
    torch.cuda.empty_cache()
    return {
        "layers": count,
        "format": "NVFP4 E2M1 weights / E4M3 block16 scales / FP32 per-expert global scale",
        "activation_precision": "BF16",
        "backend": "FlashInfer 0.7.2.post1 SM120 prepared W4A16 fused MoE",
        "activation": "GeGLU tanh",
        "expert_lora_merged": True,
        "other_lora_merged": False,
        "weight_offload": False,
        "activation_calibration": None,
        "fp4_tensor_core_w4a4": False,
    }


@torch.no_grad()
def load_packed_experts(model, directory, manifest):
    """Load a verified private export into a fresh inference-only model copy."""
    import hashlib
    import json
    from pathlib import Path

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        W4A16PackedWeights,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file

    directory = Path(directory)
    for i, (enc, dec) in enumerate(zip(
        model.model.encoder.language_model.layers, model.model.decoder.layers, strict=True
    )):
        path = directory / f"layer-{i:02d}.safetensors"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest[path.name]
        with safe_open(path, framework="pt") as handle:
            metadata = json.loads(handle.metadata()["settings"])
        assert metadata["params_dtype"] == "torch.bfloat16"
        metadata["params_dtype"] = torch.bfloat16
        prepared = W4A16PackedWeights(**metadata, **load_file(path, device="cuda"))
        enc.experts = dec.experts = FusedNVFP4Experts(prepared)
    gc.collect()
    model.to("cuda")
    model.requires_grad_(False)
    torch.cuda.empty_cache()
