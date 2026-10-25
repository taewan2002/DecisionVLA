"""Native robot adaptation objective; no automatic training launch.

The 80/20 noise mixture and action-only supervised positions are our adaptation,
not an OpenJEV training recipe. Empty replacement draws contribute zero loss.
"""
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize
from transformers import LogitsProcessorList

from .native_sampling import generation_config


def prepare_robot_example(processor, sample, canvas, visual_tokens=280):
    """One dataset sample, with the SAME typed prefix/label bindings as inference.

    A future trainer must call this helper instead of the historical pilot's
    prepare_observation. This function prepares data and does not run training.
    """
    from .decision_canvas import prepare_decision_observation, robot_option_targets
    from .typed_decisions import robot_schema

    schema = robot_schema()
    inputs = prepare_decision_observation(processor, sample["image"], sample["task"], schema,
                                         canvas, visual_tokens)
    targets = robot_option_targets(torch.as_tensor(sample["tokens"], dtype=torch.long)[None])
    valid = torch.as_tensor(sample["valid"], dtype=torch.bool)[None]
    return inputs, targets, valid


def corrupt_canvas(canvas, clean_bins, valid, generator, full_noise_probability=0.2):
    if valid.shape != clean_bins.shape or valid.dtype != torch.bool or not valid.any(-1).all():
        raise ValueError("Every observation needs at least one valid action position")
    if not 0 <= full_noise_probability <= 1:
        raise ValueError("Invalid full-noise probability")
    clean = canvas.targets(clean_bins)
    rates = torch.rand((len(clean), 1), device=clean.device, generator=generator)
    full = torch.rand(rates.shape, device=clean.device, generator=generator) < full_noise_probability
    rates = torch.where(full, 1.0, rates)
    replaced = (torch.rand(clean_bins.shape, device=clean.device, generator=generator) < rates) & valid
    noise = torch.randint(canvas.vocab_size, clean_bins.shape, device=clean.device, generator=generator)
    corrupted = clean.clone()
    corrupted[:, canvas.slots] = torch.where(replaced, noise, clean[:, canvas.slots])
    # Unavailable future slots cannot reveal repeated target actions to valid slots.
    corrupted[:, canvas.slots] = torch.where(valid, corrupted[:, canvas.slots], noise)
    return corrupted, clean, replaced, rates


def selected_ce(logits, clean, slots, selected):
    rows, columns = selected.nonzero(as_tuple=True)
    if not len(rows):
        return logits[..., 0].sum() * 0
    positions = torch.as_tensor(slots, device=logits.device)[columns]
    losses = F.cross_entropy(logits[rows, positions].float(), clean[rows, positions], reduction="none")
    sums = torch.zeros(len(logits), dtype=losses.dtype, device=logits.device).scatter_add(0, rows, losses)
    return (sums / selected.sum(-1).clamp_min(1)).mean()


def encode_native_prefix(model, inputs, cfg):
    ids = inputs["input_ids"]
    batch, length = ids.shape
    attention = inputs.get("attention_mask", torch.ones_like(ids)).bool()
    positions = torch.arange(length, dtype=torch.int32, device=ids.device)[None]
    extras = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
    cache = model._prepare_cache_for_generation(cfg, batch, length)
    ids, masks = model._prepare_encoder_inputs(ids, attention, positions, cache,
                                               True, model.config.canvas_length, batch, **extras)
    cache = model.model.encoder(input_ids=ids, attention_mask=masks, past_key_values=cache,
                               position_ids=positions, **extras).past_key_values
    return cache, F.pad(attention, (0, model.config.canvas_length), value=True), positions.shape[-1]


def native_training_loss(model, inputs, clean_bins, valid, canvas, generator, *,
                         full_noise_probability=0.2, self_condition_probability=0.5):
    if not 0 <= self_condition_probability <= 1:
        raise ValueError("Invalid self-conditioning probability")
    noisy, clean, selected, rates = corrupt_canvas(canvas, clean_bins, valid, generator, full_noise_probability)
    # This SFT proposal conditions on a first-pass prediction processed at the
    # native initial temperature (remaining == budget). It is not a sampled rollout.
    cfg = generation_config(model, canvas, 4)
    with parametrize.cached():
        cache, attention, prefix_length = encode_native_prefix(model, inputs, cfg)
        kwargs = {"decoder_input_ids": noisy, "past_key_values": cache,
                  "decoder_attention_mask": attention,
                  "decoder_position_ids": (prefix_length + torch.arange(len(canvas.template), device=noisy.device))[None]}
        use_conditioning = bool(torch.rand((), device=noisy.device, generator=generator) < self_condition_probability)
        conditioning = None
        if use_conditioning:
            with torch.no_grad():
                first = model(**kwargs).logits
                processors = model._prepare_logits_processor(cfg, LogitsProcessorList())
                conditioning = processors(inputs["input_ids"], canvas.pin_logits(first),
                                          cur_step=cfg.max_denoising_steps).to(model.dtype).detach()
        logits = model(**kwargs, self_conditioning_logits=conditioning).logits
        loss = selected_ce(logits, clean, canvas.slots, selected)
    return loss, {"selected_count": selected.sum(-1).detach(), "noise_rate": rates.detach(),
                  "self_conditioning": use_conditioning, "normalizer": "per_observation_then_batch"}
