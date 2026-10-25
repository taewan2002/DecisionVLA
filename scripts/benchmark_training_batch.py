"""Disposable effective-batch-preserving throughput and gradient comparison.

Run each microbatch size in a separate process on an otherwise free GPU.
Never export these temporary updates as trained checkpoints.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from decisionvla.batching import backward_batch
from decisionvla.checkpoints import file_sha256
from decisionvla.data import LiberoDataset
from decisionvla.model import load_model
from decisionvla.train import restore_checkpoint, update


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gradient_comparison(model, reference):
    squared_diff, squared_ref, maximum = 0.0, 0.0, 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.grad.detach().float().cpu()
        original = reference.pop(name)
        difference = current - original
        squared_diff += float(difference.double().square().sum())
        squared_ref += float(original.double().square().sum())
        maximum = max(maximum, float(difference.abs().max()))
    if reference:
        raise ValueError("Unexpected reference gradient names")
    return {"relative_l2": (squared_diff / max(squared_ref, 1e-30)) ** 0.5,
            "max_abs": maximum, "reference_l2": squared_ref ** 0.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"batch-{args.microbatch}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    config = json.loads(args.config.read_text())
    expected = json.loads((args.checkpoint / "sha256.json").read_text())
    for name, digest in expected.items():
        if Path(name).name != name or file_sha256(args.checkpoint / name) != digest:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    model, processor, info = load_model(args.model, config)
    dataset = LiberoDataset(args.data, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"],
                                 foreach=config.get("optimizer_foreach"))
    sampler = np.random.default_rng(42)
    noise = torch.Generator(device="cuda").manual_seed(43)
    step = restore_checkpoint(model, optimizer, args.checkpoint, sampler, noise)
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"]
    result = {"microbatch": args.microbatch, "effective_batch": 8,
              "accumulation": 8 // args.microbatch, "parent_step": step,
              "parent_manifest_sha256": file_sha256(args.checkpoint / "sha256.json"),
              "trainable_parameters": info["trainable_parameters"],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "status": "running", "disposable_updates": True,
              "candidate_adopted": False, "rows": [],
              "comparison_limits": {"first_loss_relative": 0.01, "gradient_relative_l2": 0.05}}
    save(result_path, result)
    try:
        for number in range(args.updates):
            model.train()
            torch.cuda.reset_peak_memory_stats()
            if args.microbatch == 1:
                row = update(model, processor, dataset, optimizer, config, sampler, noise, "cuda")
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                row = backward_batch(model, processor, dataset, config, sampler, noise, "cuda", args.microbatch)
                norm = torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
                if float(norm) <= 0:
                    raise ValueError("No training gradient")
                optimizer.step()
                torch.cuda.synchronize()
                row.update(grad_norm=float(norm), seconds=time.perf_counter() - start,
                           peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
            row["noise_rng_sha256"] = hashlib.sha256(noise.get_state().cpu().numpy().tobytes()).hexdigest()
            row["sampler_state"] = sampler.bit_generator.state
            row["number"] = number
            result["rows"].append(row)
            if number == 0:
                if args.microbatch == 1:
                    torch.save({name: parameter.grad.detach().float().cpu()
                                for name, parameter in model.named_parameters() if parameter.requires_grad},
                               args.output / "reference-gradients.pt")
                else:
                    reference = torch.load(args.output / "reference-gradients.pt", map_location="cpu", weights_only=True)
                    result["first_gradient_comparison"] = gradient_comparison(model, reference)
            save(result_path, result)
            print(json.dumps({k: v for k, v in row.items() if k not in {"sample_keys", "sampler_state"}}), flush=True)
        measured = result["rows"][2:]
        result.update(status="completed", median_seconds=statistics.median(row["seconds"] for row in measured),
                      mean_seconds=statistics.mean(row["seconds"] for row in measured),
                      measured_updates=len(measured), excluded_warmup_updates=2,
                      peak_allocated_GiB=max(row["peak_allocated_GiB"] for row in result["rows"]))
        if args.microbatch > 1:
            baseline = json.loads((args.output / "batch-1.json").read_text())
            result["sample_and_rng_exact"] = all(
                all(a[key] == b[key] for key in ("sample_keys", "supervised_tokens", "valid_tokens",
                                                "noise_rng_sha256", "sampler_state"))
                for a, b in zip(baseline["rows"], result["rows"], strict=True))
            loss_relative = abs(baseline["rows"][0]["loss"] - result["rows"][0]["loss"]) / abs(baseline["rows"][0]["loss"])
            result["first_loss_relative"] = loss_relative
            result["numerical_screen_passed"] = bool(result["sample_and_rng_exact"]
                and loss_relative < 0.01 and result["first_gradient_comparison"]["relative_l2"] < 0.05)
            result["speedup_vs_batch_1"] = baseline["median_seconds"] / result["median_seconds"]
    except BaseException as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                      error=repr(error))
        raise
    finally:
        save(result_path, result)
        dataset.close()


if __name__ == "__main__":
    main()
