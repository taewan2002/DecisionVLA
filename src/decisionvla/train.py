"""Single-GPU LoRA pilot with deterministic sampling, atomic checkpoints and held-out demos."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .adapters import adapter_state
from .checkpoints import file_sha256, load_adapter, save_adapter
from .continuation import learning_rate_at, training_boundary, validate_extension
from .data import LiberoDataset, prepare_observation
from .model import (
    action_logits,
    denoising_loss,
    encode_prefix,
    load_model,
    sample_actions,
    training_corruption,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def to_device(inputs, device):
    return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()}


def save_checkpoint(model, optimizer, output, step, config, sampler, noise, metadata):
    target = output / f"checkpoint-{step:06d}"
    if target.exists():
        raise FileExistsError(target)
    staging = output / f".checkpoint-{step:06d}.tmp"
    staging.mkdir()
    save_adapter(model, staging)
    adapter_format = "decisionvla-shared-lora-v2" if config.get("lora_experts") else "decisionvla-shared-lora-v1"
    write_json(staging / "adapter_config.json", {**config, "format": adapter_format, "step": step})
    write_json(staging / "training_metadata.json", metadata)
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "sampler": sampler.bit_generator.state,
                "noise": noise.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, staging / "training_state.pt")
    write_json(staging / "sha256.json", {p.name: file_sha256(p)
                                       for p in staging.iterdir() if p.is_file()})
    staging.replace(target)
    write_json(output / "latest.json", {"checkpoint": target.name, "step": step})
    return target


def restore_checkpoint(model, optimizer, path, sampler, noise):
    # training_state.pt is local, self-generated optimizer/RNG state. Release only safetensors.
    load_adapter(model, path)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    sampler.bit_generator.state = state["sampler"]
    noise.set_state(state["noise"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return state["step"]


@torch.no_grad()
def validate(model, processor, dataset, config, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(9001)
    indices = dataset.validation_indices()
    error, grip, tokens, losses = [], [], [], []
    per_suite = {}
    for index in indices:
        item = dataset[int(index)]
        inputs = to_device(prepare_observation(processor, item["image"], item["task"], config["visual_tokens"]), device)
        cache = encode_prefix(model, inputs)
        target = torch.from_numpy(item["tokens"])[None].to(device)
        valid = torch.from_numpy(item["valid"])[None].to(device)
        predicted = sample_actions(model, cache, config, config["validation_denoising_steps"], generator)
        decoded = dataset.codec.decode(predicted.cpu().numpy().reshape(-1, 7))
        mask = item["valid"].reshape(-1, 7)[:, 0]
        error.extend(np.abs(decoded[mask, :6] - item["actions"][mask, :6]).ravel().tolist())
        grip.extend((np.sign(decoded[mask, 6]) == np.sign(item["actions"][mask, 6])).tolist())
        tokens.extend((predicted == target)[valid].cpu().tolist())
        noisy, loss_mask = training_corruption(target, valid, config, generator)
        losses.append(float(denoising_loss(action_logits(model, cache, noisy, config), target, loss_mask)))
        if "suites" in config:
            row = per_suite.setdefault(item["suite"], {"continuous_action_mae": [],
                "gripper_accuracy": [], "action_token_accuracy": [], "denoising_ce": []})
            row["continuous_action_mae"].append(float(np.abs(decoded[mask, :6] - item["actions"][mask, :6]).mean()))
            row["gripper_accuracy"].append(float((np.sign(decoded[mask, 6]) == np.sign(item["actions"][mask, 6])).mean()))
            row["action_token_accuracy"].append(float((predicted == target)[valid].float().mean()))
            row["denoising_ce"].append(losses[-1])
    model.train()
    result = {"continuous_action_mae": float(np.mean(error)), "gripper_accuracy": float(np.mean(grip)),
            "action_token_accuracy": float(np.mean(tokens)), "denoising_ce": float(np.mean(losses)),
            "denoising_loss_scope": config.get("denoising_loss_scope", "all_valid"),
            "samples": len(indices), "kind": "held_out_demonstration_open_loop_not_rollout_success"}
    if per_suite:
        result["per_suite_observation_means"] = {suite: {"samples": len(row["denoising_ce"]),
            **{key: float(np.mean(values)) for key, values in row.items()}} for suite, row in per_suite.items()}
    return result


def update(model, processor, dataset, optimizer, config, sampler, noise, device, audit=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, keys, prefix_grad = [], [], None
    supervised_tokens, valid_tokens = 0, 0
    start = time.perf_counter()
    for micro in range(config["gradient_accumulation"]):
        index = int(sampler.integers(len(dataset)))
        item = dataset[index]
        keys.append(item["key"])
        inputs = to_device(prepare_observation(processor, item["image"], item["task"], config["visual_tokens"]), device)
        target = torch.from_numpy(item["tokens"])[None].to(device)
        valid = torch.from_numpy(item["valid"])[None].to(device)
        noisy, loss_mask = training_corruption(target, valid, config, noise)
        supervised_tokens += int(loss_mask.sum())
        valid_tokens += int(valid.sum())
        cache = encode_prefix(model, inputs)
        key_cache = cache.layers[0].keys
        if audit and micro == 0:
            assert key_cache.requires_grad, "Prefix was detached from LoRA"
            key_cache.retain_grad()
        logits = action_logits(model, cache, noisy, config)
        loss = denoising_loss(logits, target, loss_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        (loss / config["gradient_accumulation"]).backward()
        if audit and micro == 0:
            prefix_grad = float(key_cache.grad.float().norm())
            assert prefix_grad > 0 and math.isfinite(prefix_grad)
        losses.append(float(loss.detach()))
        del cache, key_cache, logits, loss
    trainable = [p for p in model.parameters() if p.requires_grad]
    family_gradients = {}
    if audit:
        for family in ("self_attn", "mlp", "experts", "embed_tokens"):
            values = [p.grad.detach().float().norm().square() for name, p in model.named_parameters()
                      if p.requires_grad and p.grad is not None and f".{family}." in name]
            if values:
                value = float(torch.stack(values).sum().sqrt())
                assert value > 0 and math.isfinite(value), f"No finite gradient in {family}"
                family_gradients[family] = value
        required = {"self_attn"}
        if config["train_action_embeddings"]:
            required.add("embed_tokens")
        if config.get("lora_mlp_targets"):
            required.add("mlp")
        if config.get("lora_experts"):
            required.add("experts")
        assert required.issubset(family_gradients), f"Missing gradient families: {required - family_gradients.keys()}"
    norm = torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
    assert float(norm) > 0, "No learning signal"
    optimizer.step()
    torch.cuda.synchronize()
    return {"loss": float(np.mean(losses)), "grad_norm": float(norm), "seconds": time.perf_counter() - start,
            "supervised_tokens": supervised_tokens, "valid_tokens": valid_tokens,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
            "sample_keys": keys, "prefix_gradient_norm": prefix_grad,
            "family_gradient_norms": family_gradients}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/libero_spatial.json"))
    parser.add_argument("--output", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--extend-from", type=Path,
                                help="New run carrying parent optimizer/RNG with an explicit extended schedule")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-at-step", type=int,
                        help="Save and exit at this step without shortening the configured LR schedule")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke and args.stop_at_step is not None:
        raise ValueError("Use a separate run for a scheduled boundary check, not --smoke")
    if config["microbatch"] != 1:
        raise ValueError("Pilot supports microbatch=1; use gradient accumulation")
    if args.smoke:
        if args.extend_from:
            parent_step = json.loads((args.extend_from / "adapter_config.json").read_text())["step"]
            config = {**config, "steps": parent_step + 3, "validation_samples": 2}
        else:
            config = {**config, "gradient_accumulation": 1, "steps": 3, "validation_samples": 2}
    if args.resume:
        previous = json.loads((args.resume / "adapter_config.json").read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Resume requires the same training config, including schedule and split")
    extension = None
    parent_path = args.resume or args.extend_from
    if parent_path:
        expected = json.loads((parent_path / "sha256.json").read_text())
        for name, sha in expected.items():
            if Path(name).name != name or file_sha256(parent_path / name) != sha:
                raise ValueError(f"Parent checkpoint hash mismatch: {name}")
        if "training_state.pt" not in expected:
            raise ValueError("Continuation requires verified local optimizer and RNG state")
    if args.extend_from:
        previous = json.loads((args.extend_from / "adapter_config.json").read_text())
        changes = validate_extension(previous, config)
        extension = {"kind": "schedule_extension", "parent_checkpoint": args.extend_from.name,
                     "parent_step": previous["step"], "config_changes": changes,
                     "parent_manifest_sha256": file_sha256(args.extend_from / "sha256.json"),
                     "carried": ["adapter", "optimizer", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng"],
                     "original_schedule_modified": False}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "events.jsonl").exists() and args.resume is None:
        raise FileExistsError("Use a new output directory or --resume")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = "cuda"
    sampler = np.random.default_rng(config["seed"])
    noise = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    train = LiberoDataset(args.data, config, "train")
    validation = LiberoDataset(args.data, config, "validation")
    train_ids = {(x[0].name, x[1]) for x in train.episodes}
    val_ids = {(x[0].name, x[1]) for x in validation.episodes}
    assert train_ids.isdisjoint(val_ids)
    metadata = {"config": config, "train": train.manifest(), "validation": validation.manifest(),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "status": "loading", "robot_success_evaluated": False}
    if extension:
        metadata["initialization"] = extension
    if args.resume:
        previous_metadata = json.loads((args.resume / "training_metadata.json").read_text())
        if "initialization" in previous_metadata:
            metadata["initialization"] = previous_metadata["initialization"]
        metadata["resume_checkpoint"] = args.resume.name
    manifest = Path(args.data) / "manifest.json"
    if manifest.exists():
        metadata["dataset_manifest"] = json.loads(manifest.read_text())
    write_json(args.output / "run.json", metadata)
    model, processor, info = load_model(args.model, config, device)
    metadata.update(info)
    metadata["status"] = "training"
    write_json(args.output / "run.json", metadata)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                 foreach=config.get("optimizer_foreach"))
    parent = args.resume or args.extend_from
    first = restore_checkpoint(model, optimizer, parent, sampler, noise) if parent else 0
    if first >= config["steps"]:
        raise ValueError("No updates remain in the requested schedule")
    stop = training_boundary(config, first, args.stop_at_step)
    metadata["starting_step"] = (previous_metadata.get("starting_step", first) if args.resume else first)
    write_json(args.output / "run.json", metadata)
    start = time.monotonic()
    events = (args.output / "events.jsonl").open("a", buffering=1)

    def record(value):
        events.write(json.dumps(value) + "\n")
        print(json.dumps({k: v for k, v in value.items() if k != "sample_keys"}), flush=True)

    if args.smoke:
        before = validate(model, processor, validation, config, device)
        record({"event": "validation_before", **before})
    elif not args.resume:
        record({"event": "validation_before", "step": first,
                **validate(model, processor, validation, config, device)})
    checkpoint = None
    record({"event": "training_segment", "from_step": first, "stop_at_step": stop,
            "schedule_end_step": config["steps"]})
    for step in range(first + 1, stop + 1):
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        row = update(model, processor, train, optimizer, config, sampler, noise, device, audit=step == first + 1)
        record({"event": "update", "step": step, "lr": lr, **row})
        if args.smoke and step == first + 2:
            checkpoint = save_checkpoint(model, optimizer, args.output, step, config, sampler, noise, metadata)
        if args.smoke and step == first + 3:
            expected = adapter_state(model)
            restore_checkpoint(model, optimizer, checkpoint, sampler, noise)
            for group in optimizer.param_groups:
                group["lr"] = lr
            repeated = update(model, processor, train, optimizer, config, sampler, noise, device)
            assert repeated["sample_keys"] == row["sample_keys"]
            assert abs(repeated["loss"] - row["loss"]) < 1e-5
            actual = adapter_state(model)
            maximum = max(float((value - actual[name]).abs().max()) for name, value in expected.items())
            assert maximum < 1e-5, maximum
            record({"event": "resume_verified", "max_parameter_abs_error": maximum,
                    "loss_abs_error": abs(repeated["loss"] - row["loss"]),
                    "gradient_norm_abs_error": abs(repeated["grad_norm"] - row["grad_norm"]),
                    "parameter_atol": 1e-5, "bitwise_equal": maximum == 0})
        deadline = time.monotonic() - start > config["max_training_hours"] * 3600
        if step % config["validation_every"] == 0 or step == stop or deadline:
            record({"event": "validation", "step": step,
                    **validate(model, processor, validation, config, device)})
        if step % config["checkpoint_every"] == 0 or step == stop or deadline:
            checkpoint = save_checkpoint(model, optimizer, args.output, step, config, sampler, noise, metadata)
        write_json(args.output / "status.json", {"step": step, "target_steps": config["steps"],
                   "elapsed_seconds": time.monotonic() - start, "last_update_seconds": row["seconds"],
                   "last_loss": row["loss"], "latest_checkpoint": checkpoint.name if checkpoint else None,
                   "status": "deadline_reached" if deadline else "running"})
        if deadline:
            break
    if args.smoke and not args.extend_from:
        # A separate disposable overfit check; never exported as a trained policy.
        item = train[0]
        inputs = to_device(prepare_observation(processor, item["image"], item["task"], config["visual_tokens"]), device)
        target = torch.from_numpy(item["tokens"])[None].to(device)
        valid = torch.from_numpy(item["valid"])[None].to(device)
        from .model import random_actions
        fixed_noise = random_actions(target.shape, config, device, torch.Generator(device=device).manual_seed(123))
        overfit_losses = []
        for _ in range(24):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"]
            cache = encode_prefix(model, inputs)
            loss = denoising_loss(action_logits(model, cache, fixed_noise, config), target, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                          error_if_nonfinite=True)
            optimizer.step()
            overfit_losses.append(float(loss.detach()))
            del cache, loss
        assert np.mean(overfit_losses[-3:]) < np.mean(overfit_losses[:3]) * 0.8, overfit_losses
        record({"event": "disposable_single_frame_overfit", "losses": overfit_losses,
                "trained_policy_claim": False})
    terminal = "awaiting_evaluation" if step < config["steps"] else "completed"
    metadata["status"] = "smoke_passed" if args.smoke else ("deadline_reached" if deadline else terminal)
    metadata["completed_steps"] = step
    metadata["updates_this_run"] = step - metadata["starting_step"]
    write_json(args.output / "run.json", metadata)
    status = json.loads((args.output / "status.json").read_text())
    status["status"] = metadata["status"]
    write_json(args.output / "status.json", status)
    record({"event": metadata["status"], "step": step})
    train.close()
    validation.close()


if __name__ == "__main__":
    main()
