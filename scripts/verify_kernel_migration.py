"""Compare full global-batch loss and raw DDP gradients from a preserved state.

Both disposable updates restore the same adapter, optimizer and all-rank RNG.
Their resulting weights are never used to continue production training.
"""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from decisionvla.adapters import LoRAExperts
from decisionvla.grouped_gemm import EXPERT_KERNELS
from decisionvla.continuation import learning_rate_at
from decisionvla.data import LiberoDataset
from decisionvla.distributed import DenoisingObjective, assert_adapter_replicas, gather_objects, restore_rng
from decisionvla.model import load_model
from decisionvla.train_distributed import restore_checkpoint, update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-kernel", choices=EXPERT_KERNELS, default="triton_sm120")
    p.add_argument("--reference-kernel", choices=EXPERT_KERNELS, default="torch_grouped_mm")
    args = p.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    config = json.loads(args.config.read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model, processor, info = load_model(args.model, config, device)
    ddp = DistributedDataParallel(DenoisingObjective(model, config), device_ids=[rank],
        broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=64,
        find_unused_parameters=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
    sampler = np.random.default_rng(config["seed"])
    noise = torch.Generator(device=device)
    data = LiberoDataset(args.data, config, "train")
    reference = {}
    metrics = {}
    input_hashes = []
    import decisionvla.train_distributed as trainer

    original_prepare = trainer.prepare_microbatch

    def traced_prepare(*arguments, **kwargs):
        batch = original_prepare(*arguments, **kwargs)
        inputs, noisy, targets, masks, _, _ = batch
        digest = hashlib.sha256()
        for name, value in [*sorted(inputs.items()), ("noisy", noisy), ("targets", targets), ("masks", masks)]:
            digest.update(name.encode())
            digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        input_hashes.append(digest.hexdigest())
        return batch

    trainer.prepare_microbatch = traced_prepare

    def capture(m):
        for name, parameter in m.named_parameters():
            if parameter.requires_grad:
                reference[name] = parameter.grad.detach().cpu().clone()

    def compare(m):
        sums = {}
        for name, parameter in m.named_parameters():
            if not parameter.requires_grad:
                continue
            a = reference.pop(name).to(device)
            b = parameter.grad.detach()
            family = next(f for f in ("experts", "self_attn", "mlp", "embed_tokens") if f".{f}." in name)
            row = sums.setdefault(family, [0.0] * 4)
            values = torch.stack([a.square().sum(), b.square().sum(), (a - b).square().sum(), (a * b).sum()])
            for i, value in enumerate(values.tolist()):
                row[i] += value
        sums["all"] = [sum(v[i] for v in sums.values()) for i in range(4)]
        for family, (a2, b2, d2, ab) in sums.items():
            metrics[family] = {"relative_l2": math.sqrt(d2 / max(a2, 1e-30)),
                "cosine": ab / max(math.sqrt(a2 * b2), 1e-30),
                "norm_ratio": math.sqrt(b2 / max(a2, 1e-30))}

    rows, initial_shas, all_inputs = [], [], []
    for kernel, observer in [(args.reference_kernel, capture), (args.candidate_kernel, compare)]:
        opt.zero_grad(set_to_none=True)
        opt.state.clear()
        for module in model.modules():
            if isinstance(module, LoRAExperts):
                module.kernel = kernel
        first, rng, _ = restore_checkpoint(model, opt, args.parent, config,
                                           allow_kernel_migration=True, diagnostic=True)
        restore_rng(rng, sampler, noise)
        initial_shas.append(assert_adapter_replicas(model))
        for group in opt.param_groups:
            group["lr"] = learning_rate_at(first + 1, config)
        input_hashes.clear()
        if rank == 0:
            print(f"Starting disposable update: {kernel}", flush=True)
        row = update(ddp, processor, data, opt, config, sampler, noise, device,
                     audit=True, gradient_observer=observer)
        all_inputs.append(gather_objects(list(input_hashes)))
        rows.append({"kernel": kernel, **row})
    gradients = gather_objects(metrics)
    same_keys = [[x["sample_keys"] for x in row["ranks"]] for row in rows]
    loss_relative = abs(rows[0]["loss"] - rows[1]["loss"]) / max(abs(rows[0]["loss"]), 1e-12)
    passed = (initial_shas[0] == initial_shas[1] and same_keys[0] == same_keys[1] and all_inputs[0] == all_inputs[1]
              and loss_relative < 0.01 and all(m["all"]["relative_l2"] < 0.05
                  and m["all"]["cosine"] > 0.999
                  and abs(m["all"]["norm_ratio"] - 1) < 0.02 for m in gradients))
    result = {"status": "passed" if passed else "failed", "parent_step": first,
              "reference_kernel": args.reference_kernel, "candidate_kernel": args.candidate_kernel,
              "global_batch": config["global_batch_size"], "initial_adapter_shas": initial_shas,
              "same_sample_keys": same_keys[0] == same_keys[1], "loss_relative_error": loss_relative,
              "same_inputs_and_corruption": all_inputs[0] == all_inputs[1], "input_hashes": all_inputs,
              "thresholds": {"loss_relative": 0.01, "gradient_relative_l2": 0.05,
                             "gradient_cosine_min": 0.999, "gradient_norm_relative": 0.02},
              "gradient_comparison_by_rank": gradients, "updates": rows,
              "scope": "two disposable updates from the same checkpoint; timing includes gradient capture/comparison",
              "trainable_parameters": info["trainable_parameters"]}
    if rank == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "updates"}), flush=True)
    data.close()
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("Full-model numerical migration gate failed; no production resume")


if __name__ == "__main__":
    main()
