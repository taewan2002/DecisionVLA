"""Calibrated experimental W4A4 NVFP4 Tensor Core experts for SM120.

Per-layer activation scales come only from fixed training observations run
through the W4A16 policy. Static fused MoE avoids the dynamic path's observed
nondeterminism. Encoder/decoder share one converted expert module per layer.
"""

import gc

import torch
from resident_nvfp4 import merge_experts, pack_experts
from torch import nn
from torch.nn import functional as F

from decisionvla.adapters import LoRAExperts


class FusedNVFP4A4Experts(nn.Module):
    def __init__(self, w1, w2, calibration):
        super().__init__()
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        # Both shapes passed independent numerical/repeat checks; dynamic did
        # not reproduce bitwise at production geometry and is not selected.
        moe_dispatch._FORCED_BACKEND = "static"
        e, twice_i, half_h = w1[0].shape
        self.e, self.i, self.h = e, twice_i // 2, half_h * 2
        self.register_buffer("w1", w1[0])
        self.register_buffer("w2", w2[0])
        self.register_buffer("s1", convert_sf_to_mma_layout(w1[1], m=twice_i, k=self.h, num_groups=e))
        self.register_buffer("s2", convert_sf_to_mma_layout(w2[1], m=self.h, k=self.i, num_groups=e))
        self.register_buffer("input_scale", torch.tensor([calibration["input_global_scale"]], device="cuda"))
        self.register_buffer("fc2_scale", torch.tensor([calibration["fc2_global_scale"]], device="cuda"))
        self.register_buffer("alpha1", w1[2] * self.input_scale)
        self.register_buffer("alpha2", w2[2] * self.fc2_scale)
        self._prepare_views()

    def _prepare_views(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        moe_dispatch._FORCED_BACKEND = "static"
        self._views = moe_dispatch._get_weight_views(
            self.w1, self.s1, self.w2, self.s2, self.alpha1, self.alpha2,
            n=self.i, k=self.h, quant_mode="nvfp4",
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import launch_sm120_moe

        return launch_sm120_moe(
            a=hidden_states.contiguous(),
            topk_ids=top_k_index.int().contiguous(),
            topk_weights=top_k_weights.float().contiguous(),
            w1_weight=self.w1, w1_weight_sf=self.s1, w1_alpha=self.alpha1,
            w2_weight=self.w2, w2_weight_sf=self.s2, w2_alpha=self.alpha2,
            input_global_scale=self.input_scale, fc2_input_scale=self.fc2_scale,
            input_scales_are_reciprocal=False,
            num_experts=self.e, num_local_experts=self.e, top_k=top_k_index.shape[-1],
            scatter_output=torch.empty_like(hidden_states),
            activation="gelu_tanh", quant_mode="nvfp4", fast_math=False,
            _weight_views=self._views,
        )


@torch.no_grad()
def quantize_model_a4(model, calibration):
    import flashinfer

    if flashinfer.__version__ != "0.7.2.post1" or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This backend is pinned to FlashInfer 0.7.2.post1 / SM120")
    enc, dec = model.model.encoder.language_model, model.model.decoder
    for i, (left, right) in enumerate(zip(enc.layers, dec.layers, strict=True)):
        module = left.experts
        assert module is right.experts and isinstance(module, LoRAExperts)
        merge_experts(module)
        gate, up = module.base.gate_up_proj.chunk(2, dim=1)
        intermediate = gate.shape[1]
        aligned = (intermediate + 255) // 256 * 256
        # Pad once before packing; avoid retaining both full packed copies in
        # the backend's automatic-padding cache on a 32 GiB GPU.
        gate = F.pad(gate, (0, 0, 0, aligned - intermediate))
        up = F.pad(up, (0, 0, 0, aligned - intermediate))
        w1 = pack_experts(torch.cat((up, gate), dim=1))
        w2 = pack_experts(F.pad(module.base.down_proj, (0, aligned - intermediate)))
        replacement = FusedNVFP4A4Experts(w1, w2, calibration["layers"][str(i)])
        left.experts = right.experts = replacement
        del module, gate, up, w1, w2
        print(f"W4A4 NVFP4 expert layer {i + 1}/30", flush=True)
    gc.collect()
    model.to("cuda")
    model.requires_grad_(False)
    torch.cuda.empty_cache()
    return {
        "format": "NVFP4 E2M1 weights and activations / E4M3 block16 / FP32 global scales",
        "compute": "SM120 FP4 Tensor Core static fused MoE, GeGLU tanh",
        "calibration": calibration["method"],
        "expert_lora_merged": True,
        "cpu_offload": False,
        "intermediate_padding": "704 to768; zero weights, inference only",
    }


@torch.no_grad()
def load_packed_a4(model, directory, manifest):
    """Restore a hash-verified private inference export without requantizing."""
    import hashlib
    import json
    from pathlib import Path

    import flashinfer
    from safetensors import safe_open
    from safetensors.torch import load_file

    if flashinfer.__version__ != "0.7.2.post1" or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This backend is pinned to FlashInfer 0.7.2.post1 / SM120")
    for i, (enc, dec) in enumerate(zip(
        model.model.encoder.language_model.layers, model.model.decoder.layers, strict=True
    )):
        path = Path(directory) / f"layer-{i:02d}.safetensors"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest[path.name]
        buffers = load_file(path, device="cuda")
        with safe_open(path, framework="pt") as handle:
            strides = json.loads(handle.metadata()["strides"])
        # MMA scale-factor tensors are permuted views. Restore their actual
        # storage layout after safetensors' required contiguous serialization.
        for name, value in list(buffers.items()):
            stride = tuple(strides[name])
            if value.stride() != stride:
                restored = torch.empty_strided(value.shape, stride, dtype=value.dtype, device=value.device)
                restored.copy_(value)
                buffers[name] = restored
        assert set(buffers) == {"w1", "w2", "s1", "s2", "input_scale", "fc2_scale", "alpha1", "alpha2"}
        module = FusedNVFP4A4Experts.__new__(FusedNVFP4A4Experts)
        nn.Module.__init__(module)
        for name, value in buffers.items():
            module.register_buffer(name, value)
        module.e, twice_i, half_h = module.w1.shape
        module.i, module.h = twice_i // 2, half_h * 2
        module._prepare_views()
        enc.experts = dec.experts = module
    gc.collect()
    model.to("cuda")
    model.requires_grad_(False)
    torch.cuda.empty_cache()
