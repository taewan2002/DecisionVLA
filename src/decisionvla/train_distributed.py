"""Native torchrun DDP training with a fixed global batch and atomic checkpoints."""
import argparse
import json
import math
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoints import file_sha256, load_adapter, save_adapter
from .continuation import learning_rate_at, training_boundary, validate_resume_config
from .data import LiberoDataset
from .distributed import (
    DenoisingObjective,
    assert_adapter_replicas,
    backward_microbatch,
    capture_rng,
    check_batch_config,
    gather_objects,
    prepare_microbatch,
    restore_rng,
    sample_rank_indices,
)
from .model import load_model
from .train import validate, write_json


def save_checkpoint(model, optimizer, output, step, config, sampler, noise, metadata):
    replica_sha = assert_adapter_replicas(model)
    rng = gather_objects(capture_rng(sampler, noise))
    target = output / f"checkpoint-{step:06d}"
    if dist.get_rank() == 0:
        if target.exists():
            raise FileExistsError(target)
        staging = output / f".checkpoint-{step:06d}.tmp"
        staging.mkdir()
        save_adapter(model, staging)
        write_json(staging / "adapter_config.json", {
            **config, "format": "decisionvla-shared-lora-v2", "step": step})
        write_json(staging / "training_metadata.json", {
            **metadata, "adapter_replica_sha256": replica_sha})
        torch.save({"format": "decisionvla-ddp-state-v1", "optimizer": optimizer.state_dict(),
                    "step": step, "world_size": config["world_size"], "rank_rng": rng},
                   staging / "training_state.pt")
        write_json(staging / "sha256.json", {
            p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()})
        staging.replace(target)
        write_json(output / "latest.json", {"step": step, "checkpoint": target.name})
    dist.barrier()
    return target


def restore_checkpoint(model, optimizer, checkpoint, config, *, allow_kernel_migration=False,
                       diagnostic=False):
    previous = json.loads((checkpoint / "adapter_config.json").read_text())
    migration = validate_resume_config(previous, config, allow_kernel_migration=allow_kernel_migration)
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if metadata.get("diagnostic_only") and not diagnostic:
        raise ValueError("Disposable diagnostic weights must never initialize production")
    expected = json.loads((checkpoint / "sha256.json").read_text())
    if "training_state.pt" not in expected:
        raise ValueError("A verified local optimizer/RNG checkpoint is required")
    # Rank0 validates the shared filesystem; every rank waits before deserialization.
    if dist.get_rank() == 0:
        for name, sha in expected.items():
            if Path(name).name != name or file_sha256(checkpoint / name) != sha:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
    dist.barrier()
    load_adapter(model, checkpoint)
    # This file is locally generated trusted training state, never a release asset.
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if (state["format"] != "decisionvla-ddp-state-v1"
            or state["world_size"] != dist.get_world_size()
            or len(state["rank_rng"]) != dist.get_world_size()):
        raise ValueError("Incompatible DDP training state")
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], state["rank_rng"][dist.get_rank()], migration


def update(ddp, processor, dataset, optimizer, config, sampler, noise, device, audit=False,
           gradient_observer=None):
    ddp.train()
    optimizer.zero_grad(set_to_none=True)
    rank = dist.get_rank()
    indices = sample_rank_indices(sampler, len(dataset), config, rank, dataset)
    losses, keys = [], []
    valid_count = supervised_count = 0
    start = time.perf_counter()
    for micro, batch_indices in enumerate(indices):
        inputs, noisy, targets, masks, batch_keys, valid = prepare_microbatch(
            ddp.module.model, processor, dataset, config, batch_indices, noise, device)
        # DDP averages ranks; dividing by accumulation gives the global sample mean.
        losses.append(backward_microbatch(ddp, (inputs, noisy, targets, masks), micro,
                                         config["gradient_accumulation"]))
        keys.extend(batch_keys)
        valid_count += valid
        supervised_count += int(masks.sum())
        del inputs, noisy, targets, masks
    parameters = [p for p in ddp.module.model.parameters() if p.requires_grad]
    if gradient_observer is not None:
        gradient_observer(ddp.module.model)
    families = {}
    if audit:
        for family in ("self_attn", "mlp", "experts", "embed_tokens"):
            values = [p.grad.detach().float().norm().square()
                      for name, p in ddp.module.model.named_parameters()
                      if p.requires_grad and p.grad is not None and f".{family}." in name]
            if values:
                families[family] = float(torch.stack(values).sum().sqrt())
        required = {"self_attn", "mlp", "experts", "embed_tokens"}
        if not required <= families.keys() or not all(math.isfinite(v) and v > 0 for v in families.values()):
            raise RuntimeError(f"Missing/nonfinite training gradients: {families}")
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
    if not float(norm) > 0:
        raise FloatingPointError("No global learning signal")
    optimizer.step()
    torch.cuda.synchronize(device)
    rows = gather_objects({"rank": rank, "loss": sum(losses) / len(losses), "grad_norm": float(norm),
        "seconds": time.perf_counter() - start, "sample_keys": keys,
        "supervised_tokens": supervised_count, "valid_tokens": valid_count,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30,
        "family_gradient_norms": families})
    if sum(len(row["sample_keys"]) for row in rows) != config["global_batch_size"]:
        raise RuntimeError("Wrong global observation count")
    if max(row["grad_norm"] for row in rows) != min(row["grad_norm"] for row in rows):
        raise RuntimeError("Averaged gradient norms differ across ranks")
    return {"loss": sum(row["loss"] for row in rows) / len(rows), "grad_norm": float(norm),
        "seconds": max(row["seconds"] for row in rows), "global_batch_size": config["global_batch_size"],
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        "valid_tokens": sum(row["valid_tokens"] for row in rows), "ranks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--allow-kernel-migration", action="store_true")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Disposable <=5-update run; skip validation and mark checkpoints non-production")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("deterministic_algorithms", False):
        # Configure cuBLAS before creating any CUDA context. Unsupported
        # nondeterministic operations must fail instead of silently continuing.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    rank = dist.get_rank()
    check_batch_config(config, dist.get_world_size())
    stopped = False

    def interrupt(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "events.jsonl").exists() and args.resume is None:
            raise FileExistsError("Use a new run directory or explicit --resume")
    dist.barrier()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    sampler = np.random.default_rng(config["seed"])
    noise = torch.Generator(device=device).manual_seed(config["seed"] + 1 + rank * 1_000_003)
    train = LiberoDataset(args.data, config, "train")
    validation = LiberoDataset(args.data, config, "validation")
    assert {(p, d) for p, d, *_ in train.episodes}.isdisjoint(
        {(p, d) for p, d, *_ in validation.episodes})
    metadata = {"config": config, "train": train.manifest(), "validation": validation.manifest(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "world_size": dist.get_world_size(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "status": "loading", "initialization": "fresh LoRA from pinned base; no pilot weights or optimizer",
        "robot_success_evaluated": False, "diagnostic_only": args.diagnostic}
    if rank == 0:
        manifest = Path(args.data) / "manifest.json"
        if manifest.exists():
            metadata["dataset_manifest"] = json.loads(manifest.read_text())
        write_json(args.output / "run.json", metadata)
    model, processor, info = load_model(args.model, config, device)
    metadata.update(info)
    ddp = DistributedDataParallel(DenoisingObjective(model, config), device_ids=[local_rank],
        broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=64,
        find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
    first = 0
    if args.resume:
        first, rng, migration = restore_checkpoint(model, optimizer, args.resume, config,
            allow_kernel_migration=args.allow_kernel_migration, diagnostic=args.diagnostic)
        restore_rng(rng, sampler, noise)
        metadata["resume_checkpoint"] = args.resume.name
        metadata["initialization"] = "preserved adapter, optimizer and all-rank RNG from verified checkpoint"
        metadata["resume_manifest_sha256"] = file_sha256(args.resume / "sha256.json")
        metadata["kernel_migration"] = migration
    stop = training_boundary(config, first, args.stop_at_step)
    if args.diagnostic and (args.stop_at_step is None or stop - first > 5):
        raise ValueError("Disposable diagnostics must specify a boundary within five updates")
    metadata.update(status="training", starting_step=first)
    if rank == 0:
        write_json(args.output / "run.json", metadata)
    initial_sha = assert_adapter_replicas(model)
    events = (args.output / "events.jsonl").open("a", buffering=1) if rank == 0 else None

    def record(value):
        if rank == 0:
            events.write(json.dumps(value, allow_nan=False) + "\n")
            compact = {key: val for key, val in value.items() if key != "ranks"}
            print(json.dumps(compact, allow_nan=False), flush=True)

    record({"event": "training_segment", "from_step": first, "stop_at_step": stop,
            "schedule_end_step": config["steps"], "initial_adapter_replica_sha256": initial_sha})
    if first == 0 and rank == 0 and not args.diagnostic:
        record({"event": "validation_before", "step": 0,
                **validate(model, processor, validation, config, device)})
    dist.barrier()
    start = time.monotonic()
    checkpoint = args.resume
    for step in range(first + 1, stop + 1):
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        row = update(ddp, processor, train, optimizer, config, sampler, noise, device,
                     audit=step == first + 1)
        record({"event": "update", "step": step, "lr": lr, **row})
        stop_request = torch.tensor(int(stopped or (args.output / "STOP").exists()
            or time.monotonic() - start > config["max_training_hours"] * 3600), device=device)
        dist.all_reduce(stop_request, op=dist.ReduceOp.MAX)
        stopped = bool(stop_request.item())
        if not args.diagnostic and (step % config["validation_every"] == 0 or step == stop):
            if rank == 0:
                record({"event": "validation", "step": step,
                        **validate(model, processor, validation, config, device)})
            dist.barrier()
        if step % config["checkpoint_every"] == 0 or step == stop or stopped:
            checkpoint = save_checkpoint(model, optimizer, args.output, step, config, sampler, noise, metadata)
        if rank == 0:
            write_json(args.output / "status.json", {"status": "stopped" if stopped else "running",
                "step": step, "target_steps": config["steps"], "global_batch_size": config["global_batch_size"],
                "elapsed_seconds": time.monotonic() - start, "last_update_seconds": row["seconds"],
                "last_loss": row["loss"], "latest_checkpoint": checkpoint.name if checkpoint else None})
        if stopped:
            break
    terminal = "stopped" if stopped else ("completed" if step == config["steps"] else "awaiting_evaluation")
    if rank == 0:
        metadata.update(status=terminal, completed_steps=step, updates_this_segment=step-first)
        write_json(args.output / "run.json", metadata)
        status = json.loads((args.output / "status.json").read_text())
        write_json(args.output / "status.json", {**status, "status": terminal})
        record({"event": terminal, "step": step})
        events.close()
    train.close()
    validation.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
