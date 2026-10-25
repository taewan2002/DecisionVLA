"""Experimental training batching; the preserved training runs do not import this module."""
import torch
from torch.nn import functional as F

from .data import prepare_observation
from .model import action_logits, encode_prefix, training_corruption


def pack_observations(observations, pad_token_id):
    """Right-pad token fields; concatenate the existing single-image processor outputs."""
    if not observations:
        raise ValueError("An observation batch cannot be empty")
    fields = set(observations[0])
    supported = {"input_ids", "attention_mask", "mm_token_type_ids",
                 "pixel_values", "image_position_ids"}
    if not fields <= supported or not {"input_ids", "attention_mask"} <= fields:
        raise ValueError(f"Unsupported processor fields: {fields}")
    if any(set(item) != fields for item in observations):
        raise ValueError("All observations must have identical processor fields")
    length = max(item["input_ids"].shape[1] for item in observations)
    result = {}
    for key in fields:
        values = []
        for item in observations:
            value = item[key]
            if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                if value.ndim != 2 or value.shape != item["input_ids"].shape:
                    raise ValueError("Expected singleton token fields")
                value = F.pad(value, (0, length - value.shape[1]),
                              value=pad_token_id if key == "input_ids" else 0)
            values.append(value)
        result[key] = torch.cat(values, dim=0)
    return result


def mean_sample_loss(logits, targets, masks):
    """Match sequential accumulation: equal weight per observation, not per token."""
    counts = masks.sum(-1)
    if not (counts > 0).all():
        raise ValueError("Each observation requires a supervised token")
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss.masked_fill(~masks, 0).sum(-1) / counts).mean()


def backward_batch(model, processor, dataset, config, sampler, noise, device, microbatch):
    effective = config["microbatch"] * config["gradient_accumulation"]
    if effective % microbatch or microbatch < 1:
        raise ValueError("microbatch must divide the existing effective batch")
    losses, keys = [], []
    supervised_tokens = valid_tokens = 0
    for _ in range(effective // microbatch):
        processed, targets, noisy_rows, masks = [], [], [], []
        for _ in range(microbatch):
            item = dataset[int(sampler.integers(len(dataset)))]
            keys.append(item["key"])
            processed.append(prepare_observation(processor, item["image"], item["task"],
                                                 config["visual_tokens"]))
            target = torch.from_numpy(item["tokens"])[None].to(device)
            valid = torch.from_numpy(item["valid"])[None].to(device)
            # Preserve original sample order, per-sample noising RNG and empty-mask fallback.
            noisy, mask = training_corruption(target, valid, config, noise)
            targets.append(target)
            noisy_rows.append(noisy)
            masks.append(mask)
            supervised_tokens += int(mask.sum())
            valid_tokens += int(valid.sum())
        inputs = pack_observations(processed, model.config.text_config.pad_token_id)
        if inputs["input_ids"].shape[1] >= model.config.text_config.sliding_window:
            raise ValueError("Padded-prefix batching requires an untruncated cache")
        inputs = {key: value.to(device=device,
                  dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        cache = encode_prefix(model, inputs)
        logits = action_logits(model, cache, torch.cat(noisy_rows), config,
                               prefix_attention_mask=inputs["attention_mask"])
        loss = mean_sample_loss(logits, torch.cat(targets), torch.cat(masks))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite batched loss")
        (loss * microbatch / effective).backward()
        losses.append(float(loss.detach()))
        del cache, logits, loss
    return {"loss": sum(losses) / len(losses), "sample_keys": keys,
            "supervised_tokens": supervised_tokens, "valid_tokens": valid_tokens}
