"""Trace MoE forward divergence with CPU masters; never attach to a training process.

The reference and candidate use identical inputs, shapes and weights. CUDA arithmetic
is retained while complete attention/MLP/expert modules are streamed from CPU. This
is a numerical diagnostic, not a latency benchmark or a training migration gate.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from decisionvla.adapters import LoRAExperts
from decisionvla.checkpoints import load_adapter
from decisionvla.data import LiberoDataset
from decisionvla.distributed import prepare_microbatch, restore_rng, sample_rank_indices
from decisionvla.model import action_logits, denoising_loss, encode_prefix, load_model


def difference(reference, candidate):
    a, b = reference.float(), candidate.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite diagnostic tensor")
    return {"relative_l2": float((a-b).norm()/a.norm().clamp_min(1e-20)),
            "max_abs": float((a-b).abs().max()),
            "different_fraction": float((a != b).float().mean())}


def stream_modules(model):
    """Offload whole execution blocks, preserving shared Parameter identities."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Load the model on CPU first")
    model.requires_grad_(False)
    selected = []
    for layer in [*model.model.encoder.language_model.layers, *model.model.decoder.layers]:
        selected.extend([layer.self_attn, layer.mlp, layer.experts])
    selected = list({id(module): module for module in selected}.values())
    masters = {id(p): p.detach() for module in selected for p in module.parameters()}
    model._apply(lambda t: t if id(t) in masters else t.to("cuda"))
    handles = []
    for module in selected:
        parameters = list(module.parameters())

        def enter(module, args, parameters=parameters):
            if torch.is_grad_enabled():
                raise RuntimeError("Read-only streaming requires inference mode")
            for p in parameters:
                p.data = masters[id(p)].to("cuda")

        def leave(module, args, output, parameters=parameters):
            for p in parameters:
                p.data = masters[id(p)]

        handles.extend([module.register_forward_pre_hook(enter),
                        module.register_forward_hook(leave, always_call=True)])
    return {"streamed_modules": len(selected), "cpu_master_gib": sum(
        p.numel()*p.element_size() for p in masters.values())/2**30,
        "resident_gib": sum(p.numel()*p.element_size() for p in model.parameters()
                            if p.device.type == "cuda")/2**30}, handles


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "data", "parent", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu-budget-gib", type=float, default=6)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        args.gpu_budget_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    config = json.loads((args.parent/"adapter_config.json").read_text())
    config["expert_kernel"] = "torch_grouped_mm"
    start = time.time()
    model, processor, _ = load_model(str(args.model), config, "cpu")
    load_adapter(model, args.parent)
    model.eval()
    model.gradient_checkpointing_disable()
    placement, handles = stream_modules(model)
    print(json.dumps({"event": "loaded", **placement}), flush=True)
    data = LiberoDataset(args.data, config)
    state = torch.load(args.parent/"training_state.pt", map_location="cpu", mmap=True, weights_only=False)
    sampler = np.random.default_rng()
    noise = torch.Generator(device="cuda")
    restore_rng(state["rank_rng"][0], sampler, noise)
    del state
    indices = sample_rank_indices(sampler, len(data), config, 0)[0]
    inputs, noisy, targets, masks, keys, _ = prepare_microbatch(
        model, processor, data, config, indices, noise, torch.device("cuda"))
    digest = hashlib.sha256()
    for name, value in [*sorted(inputs.items()), ("noisy", noisy), ("targets", targets), ("masks", masks)]:
        digest.update(name.encode())
        digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    events = (args.output/"events.jsonl").open("x", buffering=1)
    reference, reference_routes, phase, occurrence = {}, {}, "reference", {}
    rows, linear_rows = {}, []

    def emit(row):
        events.write(json.dumps({"phase": phase, **row})+"\n")

    def hook(name, kind):
        def observe(module, arguments, output):
            index = occurrence.get(name, 0)
            occurrence[name] = index+1
            key = f"{name}#{index}"
            if kind == "router":
                probabilities, weights, ids = output
                if phase == "reference":
                    reference_routes[key] = tuple(t.cpu().clone() for t in output)
                else:
                    ref_p, _, ref_ids = reference_routes[key]
                    ref_ids = ref_ids.to(ids.device)
                    changed = (ref_ids.sort(-1).values != ids.sort(-1).values).any(-1)
                    sorted_p = ref_p.sort(-1, descending=True).values
                    margin = sorted_p[:, 7]-sorted_p[:, 8]
                    row = {"key": key, "kind": kind, **difference(ref_p, probabilities.cpu()),
                           "changed_expert_set": int(changed.sum()), "tokens": len(ids),
                           "reference_zero_margin_tokens": int((margin == 0).sum()),
                           "changed_reference_margin_mean": float(margin[changed.cpu()].mean())
                           if changed.any() else None}
                    rows[phase].append(row)
                    emit(row)
                    if phase == "triton_fixed_routes":
                        weights = probabilities.gather(-1, ref_ids)
                        weights = weights/weights.sum(-1, keepdim=True)
                        return probabilities, weights*module.per_expert_scale[ref_ids], ref_ids
            else:
                value = output[0] if isinstance(output, tuple) else output
                value = value.detach().cpu()
                if phase == "reference":
                    reference[key] = value.clone()
                else:
                    row = {"key": key, "kind": kind, **difference(reference[key], value)}
                    rows[phase].append(row)
                    emit(row)
        return observe

    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls == "DiffusionGemmaTextRouter":
            handles.append(module.register_forward_hook(hook(name, "router")))
        elif cls in {"LoRAExperts", "DiffusionGemmaDecoderLayer"}:
            handles.append(module.register_forward_hook(hook(name, "expert" if cls == "LoRAExperts" else "layer")))
    import decisionvla.grouped_gemm as grouped
    original_linear = grouped.grouped_linear

    def shadow(x, weight, offsets, kernel):
        result = original_linear(x, weight, offsets, kernel)
        if phase == "reference":
            candidate = original_linear(x, weight, offsets, "triton_sm120")
            row = {"call": len(linear_rows), "shape": list(weight.shape),
                   "tokens": len(x), **difference(result, candidate)}
            linear_rows.append(row)
            emit({"kind": "same_input_linear", **row})
            if len(linear_rows) <= 6:
                torch.save({"x": x.cpu(), "weight": weight.cpu(), "offsets": offsets.cpu()},
                           args.output/f"private-linear-{len(linear_rows):02d}.pt")
        return result

    grouped.grouped_linear = shadow
    logits_by_phase = {}
    for phase in ["reference", "reference_repeat", "triton", "triton_fixed_routes", "triton_lora",
                  "cublas_forward"]:
        occurrence.clear()
        rows[phase] = []
        kernel = ("cublas_sm120_forward" if phase == "cublas_forward" else
                  "torch_grouped_mm" if phase.startswith("reference") else
                  "triton_sm120_lora" if phase == "triton_lora" else "triton_sm120")
        for module in model.modules():
            if isinstance(module, LoRAExperts):
                module.kernel = kernel
        print(json.dumps({"event": "phase_start", "phase": phase}), flush=True)
        cache = encode_prefix(model, inputs)
        logits = action_logits(model, cache, noisy, config, prefix_attention_mask=inputs["attention_mask"])
        allowed = torch.isfinite(logits)
        loss = float(denoising_loss(logits, targets, masks))
        logits_by_phase[phase] = logits.cpu()
        result = {"kind": "readout", "loss": loss,
                  "elapsed_seconds": time.time()-start,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
        if phase != "reference":
            ref = logits_by_phase["reference"].to(logits.device)
            result.update(difference(ref[allowed], logits[allowed]))
            result["argmax_agreement"] = float((ref.argmax(-1) == logits.argmax(-1)).float().mean())
            result["probability_l1_mean"] = float((ref.softmax(-1)-logits.softmax(-1)).abs().sum(-1).mean())
        rows[phase].append(result)
        emit(result)
        print(json.dumps({"phase": phase, **result}), flush=True)
        del cache, logits
    torch.save(logits_by_phase, args.output/"private-logits.pt")
    report = {"status": "completed", "updates": 0, "parent_step": config["step"],
              "sample_keys": keys, "input_sha256": digest.hexdigest(), "placement": placement,
              "scope": "rank0 first microbatch from preserved step38; inference-only causal diagnostic",
              "training_unchanged": True, "timing_contended_not_benchmark": True,
              "linear": linear_rows, "phases": rows}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    events.close()
    data.close()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
