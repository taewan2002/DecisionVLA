"""Checkpoint-only denoising probe on preselected held-out observations; no updates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from decisionvla.checkpoints import load_adapter
from decisionvla.adapters import LoRAExperts
from decisionvla.codec import ActionCodec
from decisionvla.confidence import summarize
from decisionvla.data import LiberoDataset, prepare_observation
from decisionvla.diagnostic_offload import offload_experts
from decisionvla.model import action_logits, encode_prefix, load_model, sample_actions
from decisionvla.grouped_gemm import check_expert_kernel
from decisionvla.rollout import digest, write_json
from decisionvla.train import to_device


def measure(probabilities, canvas, item, codec):
    p = probabilities.float().cpu().numpy().reshape(8, 7, 256)
    stats = summarize(p)
    ids = np.asarray(stats["action_bins"])
    decoded = codec.decode(ids)
    target = item["tokens"].reshape(8, 7)
    valid = item["valid"].reshape(8, 7)
    canvas = canvas.cpu().numpy().reshape(8, 7)
    wrong = valid & (ids != target)
    return {"confidence": stats["chunk_confidence"],
            "entropy": float(np.mean(stats["metrics"]["normalized_entropy"])),
            "peak_probability": float(np.mean(stats["metrics"]["selected_probability"])),
            "continuous_mae": float(np.abs(decoded[:, :6] - item["actions"][:, :6])[valid[:, :6]].mean()),
            "translation_mae": float(np.abs(decoded[:, :3] - item["actions"][:, :3])[valid[:, :3]].mean()),
            "token_accuracy": float((ids == target)[valid].mean()),
            "input_copy_fraction": float((ids == canvas)[valid].mean()),
            "wrong_token_confidence": float(np.asarray(stats["metrics"]["ordered_confidence"])[wrong].mean())
            if wrong.any() else None,
            "gripper_accuracy": float((ids[:, 6] == target[:, 6])[valid[:, 6]].mean()),
            "prediction_bins": ids.tolist()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resident-gib", type=float, default=25)
    parser.add_argument("--expected-step", type=int, default=2000)
    parser.add_argument("--full-gpu", action="store_true", help="Only with exclusive GPU ownership")
    parser.add_argument("--max-hours", type=float, default=1)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    assert config["step"] == args.expected_step
    dataset = LiberoDataset(args.data, config, "validation")
    groups = {}
    for index, (episode, frame) in enumerate(dataset.frames):
        groups.setdefault(dataset.episodes[episode][0].name, []).append(index)
    rng = np.random.default_rng(20261024)
    indices = [int(i) for name in sorted(groups) for i in rng.choice(groups[name], 2, replace=False)]
    items = [dataset[i] for i in indices]
    write_json(args.output / "plan.json", {"keys": [item["key"] for item in items],
               "seeds": [42, 43, 44], "steps": 4, "new_training_updates": 0,
               "scope": "known-task held-out demos, post-hoc mechanism diagnosis",
               "controls": ["ground_truth_canvas", "shifted_translation_and_flipped_gripper_canvas"],
               "interpretation": "confidence growth alone is expected; compare paired accuracy and input dependence",
               "adapter_sha256": digest(args.adapter / "sha256.json"),
               "gpu_allocator_limit_fraction": 0.95 if args.full_gpu else 0.34,
               "resident_gib": None if args.full_gpu else args.resident_gib})
    torch.cuda.set_per_process_memory_fraction(0.95 if args.full_gpu else 0.34)
    torch.set_num_threads(4)
    start = time.monotonic()
    # Construction/offload masters are CPU-resident; select the validated CUDA
    # backend after placement, before any model arithmetic.
    model, processor, _ = load_model(args.model, {**config, "expert_kernel": "torch_grouped_mm"}, "cpu")
    load_adapter(model, args.adapter)
    model.eval()
    if args.full_gpu:
        model.requires_grad_(False).to("cuda")
        placement, handles = {"kind": "exclusive_full_gpu"}, []
    else:
        placement, handles = offload_experts(model, args.resident_gib)
    kernel = config.get("expert_kernel", "torch_grouped_mm")
    check_expert_kernel(kernel, "cuda")
    for module in model.modules():
        if isinstance(module, LoRAExperts):
            module.kernel = kernel
    write_json(args.output / "placement.json", placement)
    print(json.dumps({"event": "loaded", **placement}), flush=True)
    codec, rows, trace_hashes = ActionCodec(256), [], {}
    with (args.output / "events.jsonl").open("x", buffering=1) as events:
        for observation, item in enumerate(items):
            raw = {"target_tokens": item["tokens"], "target_actions": item["actions"], "valid": item["valid"]}
            if time.monotonic() - start > args.max_hours * 3600:
                raise TimeoutError("Probe deadline reached")
            cache = encode_prefix(model, to_device(prepare_observation(
                processor, item["image"], item["task"], config["visual_tokens"]), "cuda"))
            for seed in (42, 43, 44):
                snapshots = []
                generator = torch.Generator(device="cuda").manual_seed(seed)

                def observe(step, canvas, probabilities, snapshots=snapshots):
                    snapshots.append((step, canvas, probabilities))

                ids = sample_actions(model, cache, config, 4, generator, step_observer=observe)
                if observation == 0 and seed == 42:
                    baseline_generator = torch.Generator(device="cuda").manual_seed(seed)
                    baseline = sample_actions(model, cache, config, 4, baseline_generator)
                    assert torch.equal(ids, baseline)
                    assert torch.equal(generator.get_state(), baseline_generator.get_state())
                    write_json(args.output / "observer-parity.json", {"actions_exact": True, "rng_exact": True})
                for step, canvas, probabilities in snapshots:
                    raw[f"seed_{seed}_pass_{step + 1}_canvas"] = canvas.cpu().numpy()
                    raw[f"seed_{seed}_pass_{step + 1}_probabilities"] = probabilities.cpu().numpy()
                    row = {"kind": "sample", "observation": observation, "key": item["key"], "seed": seed,
                           "pass": step + 1, **measure(probabilities, canvas, item, codec)}
                    rows.append(row)
                    events.write(json.dumps(row) + "\n")
                del snapshots
            target = torch.tensor(item["tokens"], device="cuda")[None]
            shifted = target.reshape(1, 8, 7).clone()
            shifted[:, :, :3] = (shifted[:, :, :3] + 64) % 256
            shifted[:, :, 6] = 255 - shifted[:, :, 6]
            for kind, canvas in [("ground_truth_canvas", target),
                                 ("shifted_translation_and_flipped_gripper_canvas", shifted.reshape(1, 56))]:
                probabilities = action_logits(model, cache, canvas, config).softmax(-1)
                raw[kind + "_canvas"] = canvas.cpu().numpy()
                raw[kind + "_probabilities"] = probabilities.cpu().numpy()
                row = {"kind": kind, "observation": observation, "key": item["key"],
                       **measure(probabilities, canvas, item, codec)}
                rows.append(row)
                events.write(json.dumps(row) + "\n")
            filename = f"observation-{observation:02d}.npz"
            np.savez_compressed(args.output / filename, **raw)
            trace_hashes[filename] = digest(args.output / filename)
            write_json(args.output / "status.json", {"status": "running", "observations": observation + 1,
                       "expected": len(items), "elapsed_seconds": time.monotonic() - start,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
            print(json.dumps({"event": "observation_complete", "observation": observation}), flush=True)
            del cache
    fields = ["confidence", "continuous_mae", "translation_mae", "token_accuracy", "input_copy_fraction",
              "wrong_token_confidence", "gripper_accuracy", "entropy", "peak_probability"]
    summary = {"status": "completed", "observations": len(items), "sampled_trajectories": len(items) * 3,
               "new_training_updates": 0, "placement": placement, "trace_sha256": trace_hashes,
               "elapsed_seconds": time.monotonic() - start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "steps": {step: {field: float(np.mean([r[field] for r in rows if r["kind"] == "sample"
                          and r["pass"] == step and r[field] is not None])) for field in fields}
                         for step in range(1, 5)},
               "controls": {kind: {field: float(np.mean([r[field] for r in rows if r["kind"] == kind
                             and r[field] is not None])) for field in fields}
                            for kind in ["ground_truth_canvas", "shifted_translation_and_flipped_gripper_canvas"]}}
    write_json(args.output / "report.json", summary)
    write_json(args.output / "status.json", {"status": "completed", "observations": len(items)})
    print(json.dumps(summary), flush=True)
    dataset.close()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
