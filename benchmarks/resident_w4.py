"""Experimental resident MoE W4A16 reference: RTN group64, GPU dequant + original GEMM.

Only frozen expert weights are packed. LoRA, attention, router, vision and embeddings
retain their original dtypes. This is not an NVFP4 export or a fused INT4 GEMM.
"""

import gc

import torch
import triton
import triton.language as tl
from torch import nn

from decisionvla.adapters import LoRAExperts
from decisionvla.grouped_gemm import grouped_linear


@triton.jit
def _unpack(packed, scale, output, N: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + index // 2, index < N, other=0).to(tl.int32)
    q = ((byte >> ((index % 2) * 4)) & 15) - 8
    s = tl.load(scale + index // GROUP, index < N, other=0)
    tl.store(output + index, q.to(tl.float32) * s, index < N)


class PackedExpertWeight(nn.Module):
    def __init__(self, weight, group=64):
        super().__init__()
        self.shape = tuple(weight.shape)
        self.group = group
        assert weight.shape[-1] % group == 0
        # Chunking bounds conversion workspace independently of full weight size.
        packs, scales = [], []
        for expert in weight:
            groups = expert.to("cuda", dtype=torch.float32).reshape(-1, group)
            scale = groups.abs().amax(-1) / 7
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = (groups / scale[:, None]).round().clamp(-7, 7).to(torch.int8).flatten() + 8
            packs.append(q[0::2].to(torch.uint8) | (q[1::2].to(torch.uint8) << 4))
            scales.append(scale)
        self.register_buffer("packed", torch.cat(packs))
        self.register_buffer("scale", torch.cat(scales))

    def dequantize(self):
        output = torch.empty(self.shape, dtype=torch.bfloat16, device=self.packed.device)
        _unpack[(triton.cdiv(output.numel(), 4096),)](
            self.packed, self.scale, output, output.numel(), self.group, 4096
        )
        return output


class ResidentW4Experts(LoRAExperts):
    def _linear(self, x, weight, offsets, *, frozen_base=False):
        if isinstance(weight, PackedExpertWeight):
            # Dequantization is GPU-to-GPU; packed weights remain resident throughout.
            weight = weight.dequantize()
        return grouped_linear(x, weight.to(x.dtype), offsets, self.kernel)


@torch.inference_mode()
def quantize_experts(model):
    assert all(p.device.type == "cpu" for p in model.parameters())
    count = original = packed = 0
    for module in list(model.modules()):
        if not isinstance(module, LoRAExperts):
            continue
        for name in ("gate_up_proj", "down_proj"):
            weight = module.base._parameters[name]
            original += weight.numel() * weight.element_size()
            converted = PackedExpertWeight(weight)
            packed += sum(t.numel() * t.element_size() for t in converted.buffers())
            del module.base._parameters[name]
            module.base.add_module(name, converted)
            del weight
        module.__class__ = ResidentW4Experts
        count += 1
        print(f"packed layer {count}/30", flush=True)
    gc.collect()
    model.to("cuda")
    torch.cuda.empty_cache()
    return {
        "expert_modules": count,
        "original_expert_bytes": original,
        "packed_expert_bytes": packed,
        "group_size": 64,
        "format": "symmetric signed INT4 [-7,7], FP32 scale per64 weights",
        "compute": "GPU dequantization to BF16 + original cached cuBLAS grouped GEMM",
        "calibration": None,
        "lora_merged": False,
        "cpu_weight_offload": False,
        "nvfp4": False,
        "fused_int4_gemm": False,
    }


@torch.inference_mode()
def check_quantizer():
    torch.manual_seed(7)
    weight = torch.randn(3, 17, 64, dtype=torch.bfloat16)
    weight[0, 0] = 0
    q = PackedExpertWeight(weight)
    groups = weight.cuda().float().reshape(-1, 64)
    scales = groups.abs().amax(-1) / 7
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    expected = (
        ((groups / scales[:, None]).round().clamp(-7, 7) * scales[:, None])
        .to(torch.bfloat16)
        .reshape(weight.shape)
    )
    result = q.dequantize()
    assert torch.equal(result, expected), (result - expected).abs().max()
    assert torch.isfinite(result).all()
    return {
        "gpu_unpack_matches_independent_torch_reference": True,
        "zero_group_exact": bool((result[0, 0] == 0).all()),
        "max_abs_error": float((result - weight.cuda()).abs().max()),
    }
