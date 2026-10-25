"""Equal-observation DDP objectives, deterministic sampling and rank-local RNG."""
import hashlib
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .batching import mean_sample_loss, pack_observations
from .data import prepare_observation
from .model import action_logits, encode_prefix, training_corruption


def check_batch_config(config, world_size):
    for key in ("microbatch", "gradient_accumulation", "world_size", "global_batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Expected positive integer {key}")
    if config["world_size"] != world_size:
        raise ValueError("World size differs from the fixed training config")
    expected = world_size * config["microbatch"] * config["gradient_accumulation"]
    if expected != config["global_batch_size"]:
        raise ValueError("Global batch must equal world size * microbatch * accumulation")


def sample_rank_indices(sampler, dataset_size, config, rank, dataset=None):
    """All ranks draw the same global batch, then take disjoint *positions*.

    Sampling is with replacement, as in the original trainer. Coincidental duplicate
    observations are permitted; replicating the complete batch on every GPU is not.
    """
    check_batch_config(config, config["world_size"])
    if not 0 <= rank < config["world_size"]:
        raise ValueError("Rank outside world")
    if config.get("sampling", "uniform_frames") == "uniform_frames":
        indices = sampler.integers(dataset_size, size=config["global_batch_size"])
    else:
        if dataset is None:
            raise ValueError("Task-balanced sampling requires dataset task ranges")
        indices = dataset.sample_indices(sampler, config["global_batch_size"])
    return indices.reshape(config["world_size"], config["gradient_accumulation"],
                           config["microbatch"])[rank]


def prepare_microbatch(model, processor, dataset, config, indices, noise, device):
    observations, targets, noisy, masks, keys = [], [], [], [], []
    valid_count = 0
    for index in indices:
        item = dataset[int(index)]
        observations.append(prepare_observation(processor, item["image"], item["task"],
                                                config["visual_tokens"]))
        target = torch.from_numpy(item["tokens"])[None].to(device)
        valid = torch.from_numpy(item["valid"])[None].to(device)
        corrupted, mask = training_corruption(target, valid, config, noise)
        targets.append(target)
        noisy.append(corrupted)
        masks.append(mask)
        keys.append(item["key"])
        valid_count += int(valid.sum())
    inputs = pack_observations(observations, model.config.text_config.pad_token_id)
    if inputs["input_ids"].shape[1] >= model.config.text_config.sliding_window:
        raise ValueError("Batched prefixes must fit the untruncated cache")
    inputs = {key: value.to(device=device,
              dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    return inputs, torch.cat(noisy), torch.cat(targets), torch.cat(masks), keys, valid_count


class DenoisingObjective(nn.Module):
    """Both encoder and decoder execute inside DDP's forward/no_sync context."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, inputs, noisy, targets, masks):
        cache = encode_prefix(self.model, inputs)
        logits = action_logits(self.model, cache, noisy, self.config,
                               prefix_attention_mask=inputs["attention_mask"])
        return mean_sample_loss(logits, targets, masks)


def backward_microbatch(ddp, arguments, micro, accumulation):
    if not 0 <= micro < accumulation:
        raise ValueError("Invalid accumulation position")
    context = ddp.no_sync() if micro + 1 < accumulation else nullcontext()
    with context:
        loss = ddp(*arguments)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DDP loss")
        (loss / accumulation).backward()
    return float(loss.detach())


def capture_rng(sampler, noise):
    return {"sampler": sampler.bit_generator.state, "noise": noise.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}


def restore_rng(state, sampler, noise):
    sampler.bit_generator.state = state["sampler"]
    noise.set_state(state["noise"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])


def gather_objects(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def assert_adapter_replicas(model):
    """Full trainable-weight hash; used at smoke/checkpoint boundaries, not each step."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    hashes = gather_objects(digest.hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError("DDP adapter replicas diverged")
    return hashes[0]
