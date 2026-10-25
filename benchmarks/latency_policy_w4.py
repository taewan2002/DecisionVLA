"""Paired BF16-streamed / resident W4 policy latency on real fixed LIBERO images."""

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from resident_w4 import check_quantizer, quantize_experts

from decisionvla.adapters import LoRAExperts
from decisionvla.checkpoints import load_adapter
from decisionvla.codec import ActionCodec
from decisionvla.confidence import validate_diagnostics
from decisionvla.model import load_model
from decisionvla.policy import DecisionVLA

sys.path.insert(0, "/source/scripts")
from diagnose_forward_numerics import stream_modules

ROOT = Path("/output")


def save(name, value):
    target = ROOT / name
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(target)


def summary(rows):
    values = np.array([r["seconds"] for r in rows])
    return {
        "n": len(rows),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "mean_seconds": float(values.mean()),
    }


def run(policy, samples, phase, repeats, warmups):
    rows, arrays = [], {}
    for i in range(warmups + repeats):
        sample = samples[(i - warmups) % len(samples)] if i >= warmups else samples[0]
        image = Image.open(Path("/inputs") / sample["file"]).convert("RGB")
        assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == sample["image_sha256"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = policy.predict_with_confidence(image, sample["instruction"], steps=4, seed=sample["seed"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        actions = output["actions"]
        diagnostics = output["diagnostics"]
        validate_diagnostics(diagnostics, actions)
        p = np.array(diagnostics["probabilities"])
        assert np.isfinite(actions).all() and np.isfinite(p).all()
        assert np.allclose(p.sum(-1), 1, atol=2e-6)
        assert actions.shape == (8, 7)
        assert np.isin(actions[:, 6], [-1.0, 1.0]).all()
        row = {
            "i": i - warmups,
            "warmup": i < warmups,
            "file": sample["file"],
            "seed": sample["seed"],
            "seconds": elapsed,
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        }
        print(json.dumps({"phase": phase, **row}), flush=True)
        with (ROOT / "events.jsonl").open("a") as f:
            f.write(json.dumps({"phase": phase, **row}) + "\n")
        if i >= warmups:
            rows.append(row)
            arrays[f"actions_{i - warmups}"] = actions
            arrays[f"probabilities_{i - warmups}"] = p
    np.savez_compressed(ROOT / f"{phase}.npz", **arrays)
    result = {"summary": summary(rows), "samples": rows}
    save(f"{phase}.json", result)
    return result, arrays


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.cuda.set_per_process_memory_fraction(29 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    config = json.loads(Path("/adapter/adapter_config.json").read_text())
    samples = json.loads(Path("/inputs/samples.json").read_text())
    report = {
        "status": "loading",
        "checkpoint_step": 4500,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "steps": 4,
        "action_shape": [8, 7],
        "images": len(samples),
        "visual_tokens": config["visual_tokens"],
        "timing_boundary": "PIL image + instruction -> preprocessing, vision/prefix, 4 denoise passes, actions + full confidence diagnostics on CPU",
        "excluded": [
            "image acquisition",
            "disk image decode",
            "RPC/network",
            "robot actuation",
            "model load",
            "warmups",
        ],
        "bf16_stream_is_same_hardware_reference": True,
        "quantizer_check": check_quantizer(),
        "quantized_closed_loop_success": "not_evaluated",
    }
    save("report.json", report)
    policy = DecisionVLA.__new__(DecisionVLA)
    policy.config = config
    policy.model, policy.processor, _ = load_model(
        "/base", {**config, "expert_kernel": "torch_grouped_mm"}, "cpu"
    )
    load_adapter(policy.model, "/adapter")
    policy.model.eval()
    policy.model.gradient_checkpointing_disable()
    original_hooks = {
        (id(m), kind, key)
        for m in policy.model.modules()
        for kind in ("_forward_hooks", "_forward_pre_hooks")
        for key in getattr(m, kind)
    }
    for module in policy.model.modules():
        if isinstance(module, LoRAExperts):
            module.kernel = config["expert_kernel"]
    policy.device = "cuda"
    policy.codec = ActionCodec(config["bins"])
    reference_dir = os.environ.get("DECISIONVLA_BF16_REFERENCE")
    handles = []
    stream_hook_ids = set()
    if reference_dir:
        reference_dir = Path(reference_dir)
        previous = json.loads((reference_dir / "report.json").read_text())
        for key in ("checkpoint_step", "gpu", "steps", "action_shape", "images", "visual_tokens"):
            assert previous[key] == report[key], key
        reference = json.loads((reference_dir / "bf16_stream.json").read_text())
        assert len(reference["samples"]) == len(samples)
        for row, sample in zip(reference["samples"], samples, strict=True):
            assert row["file"] == sample["file"] and row["seed"] == sample["seed"]
        with np.load(reference_dir / "bf16_stream.npz", allow_pickle=False) as data:
            ref_arrays = {k: data[k] for k in data.files}
        report["reference_reused_sha256"] = {
            name: hashlib.sha256((reference_dir / name).read_bytes()).hexdigest()
            for name in ("report.json", "bf16_stream.json", "bf16_stream.npz")
        }
        for name in ("bf16_stream.json", "bf16_stream.npz"):
            (ROOT / name).write_bytes((reference_dir / name).read_bytes())
        report["streaming_hooks_installed_in_this_process"] = False
    else:
        placement, handles = stream_modules(policy.model)
        stream_hook_ids = {h.id for h in handles}
        report["bf16_stream_placement"] = placement
        report["status"] = "bf16_stream"
        save("report.json", report)
        reference, ref_arrays = run(policy, samples, "bf16_stream", len(samples), 1)
        report["streaming_hooks_installed_in_this_process"] = True
    report["bf16_stream"] = reference
    report["status"] = "quantizing"
    save("report.json", report)
    for handle in handles:
        handle.remove()
    handles.clear()
    policy.model.cpu()
    policy.model.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    report["quantization"] = quantize_experts(policy.model)
    assert all(t.device.type == "cuda" for t in [*policy.model.parameters(), *policy.model.buffers()])
    remaining_hooks = {
        (id(m), kind, key)
        for m in policy.model.modules()
        for kind in ("_forward_hooks", "_forward_pre_hooks")
        for key in getattr(m, kind)
    }
    assert remaining_hooks == original_hooks
    assert not any(key in stream_hook_ids for _, _, key in remaining_hooks)
    report["original_model_hooks_preserved"] = {
        name: {
            kind: [getattr(f, "__qualname__", str(type(f))) for f in getattr(m, kind).values()]
            for kind in ("_forward_hooks", "_forward_pre_hooks")
        }
        for name, m in policy.model.named_modules()
        if m._forward_hooks or m._forward_pre_hooks
    }
    report["cpu_weight_streaming_hooks_present"] = False
    report["all_parameters_and_buffers_cuda"] = True
    report["resident_tensor_gib"] = (
        sum(p.numel() * p.element_size() for p in [*policy.model.parameters(), *policy.model.buffers()])
        / 2**30
    )
    report["status"] = "resident_benchmark"
    save("report.json", report)
    torch.cuda.reset_peak_memory_stats()
    quantized, q_arrays = run(policy, samples, "resident_w4", 20, 3)
    report["resident_w4"] = quantized
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    differences = []
    for i, sample in enumerate(samples):
        a, b = ref_arrays[f"actions_{i}"], q_arrays[f"actions_{i}"]
        p, q = ref_arrays[f"probabilities_{i}"], q_arrays[f"probabilities_{i}"]
        assert np.array_equal(b, q_arrays[f"actions_{i + 10}"])
        assert np.array_equal(q, q_arrays[f"probabilities_{i + 10}"])
        differences.append(
            {
                "file": sample["file"],
                "continuous_action_mae": float(np.abs(a[:, :6] - b[:, :6]).mean()),
                "action_max_abs": float(np.abs(a - b).max()),
                "gripper_agreement": float((a[:, 6] == b[:, 6]).mean()),
                "probability_mean_tv": float(np.abs(p - q).sum(-1).mean() / 2),
            }
        )
    report["paired_differences"] = differences
    report["repeated_output_exact"] = True
    report["status"] = "completed"
    save("report.json", report)
    print(
        json.dumps(
            {"status": "completed", "bf16": reference["summary"], "resident_w4": quantized["summary"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save("error.json", {"type": type(e).__name__, "message": str(e)})
        raise
