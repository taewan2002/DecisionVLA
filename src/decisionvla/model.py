"""Historical action-only pilot on DiffusionGemma; not the proposed native sampler.

The replacement design is documented in docs/architecture.md.
"""
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from .adapters import install_adapters


def load_model(path, config, device="cuda"):
    if config["self_conditioning"]:
        raise ValueError("Pilot v1 trains and samples without self-conditioning")
    processor = AutoProcessor.from_pretrained(path)
    start = config["action_token_start"]
    for token in range(start, start + config["bins"]):
        text = processor.tokenizer.convert_ids_to_tokens(token)
        if not text.startswith("<unused") or token in processor.tokenizer.all_special_ids:
            raise ValueError(f"Action vocabulary is not unused: {token} {text}")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="grouped_mm",
    ).to(device)
    info = install_adapters(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, processor, info


def encode_prefix(model, inputs):
    encoder = model.model.encoder
    # Explicit masks preserve bidirectional vision attention, including on the pinned
    # Transformers version whose encoder forward discards the mask builder's return.
    embeddings = encoder.language_model.embed_tokens(inputs["input_ids"])
    masks = encoder.create_masks_for_generate(
        config=model.config.text_config, inputs_embeds=embeddings,
        attention_mask=inputs.get("attention_mask"), past_key_values=None,
        position_ids=torch.arange(embeddings.shape[1], device=embeddings.device)[None],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    call = dict(inputs)
    call["attention_mask"] = masks
    return encoder(**call).past_key_values


def action_logits(model, cache, bins, config, *, prefix_attention_mask=None):
    count = config["horizon"] * config["action_dim"]
    if bins.shape[-1] != count:
        raise ValueError("Wrong action chunk length")
    canvas = torch.full((bins.shape[0], config["canvas_length"]), model.config.text_config.pad_token_id,
                        dtype=torch.long, device=bins.device)
    canvas[:, :count] = bins + config["action_token_start"]
    prefix = cache.get_seq_length()
    attention = torch.ones((bins.shape[0], prefix + config["canvas_length"]),
                           dtype=torch.bool, device=bins.device)
    attention[:, prefix + count:] = False
    decoder_kwargs = {}
    if prefix_attention_mask is not None:
        if prefix_attention_mask.shape != (bins.shape[0], prefix):
            raise ValueError("Prefix padding mask does not match the cache")
        attention[:, :prefix] = prefix_attention_mask.bool()
        decoder_kwargs["decoder_position_ids"] = (
            prefix_attention_mask.sum(-1, keepdim=True)
            + torch.arange(config["canvas_length"], device=bins.device)[None]
        )
    hidden = model.model.decoder(decoder_input_ids=canvas, past_key_values=cache,
                                 decoder_attention_mask=attention,
                                 **decoder_kwargs).last_hidden_state[:, :count]
    weight = model.model.decoder.embed_tokens.action_weight()
    logits = F.linear(hidden, weight).float()
    cap = model.final_logit_softcapping
    logits = torch.tanh(logits / cap) * cap
    # Each seventh component is a binary gripper; continuous axes use all bins.
    allowed = torch.ones((count, config["bins"]), dtype=torch.bool, device=bins.device)
    allowed[6::7, 1:-1] = False
    return logits.masked_fill(~allowed[None], -torch.inf)


def random_actions(shape, config, device, generator):
    result = torch.randint(config["bins"], shape, device=device, generator=generator)
    result[..., 6::7] = torch.randint(2, result[..., 6::7].shape, device=device,
                                     generator=generator) * (config["bins"] - 1)
    return result


def corrupt_actions(clean, config, generator, *, return_mask=False):
    noise_rate = torch.rand((clean.shape[0], 1), device=clean.device, generator=generator)
    fully_noisy = torch.rand(noise_rate.shape, device=clean.device, generator=generator)
    noise_rate = torch.where(fully_noisy < config["full_noise_probability"], 1.0, noise_rate)
    replace = torch.rand(clean.shape, device=clean.device, generator=generator) < noise_rate
    noise = random_actions(clean.shape, config, clean.device, generator)
    noisy = torch.where(replace, noise, clean)
    return (noisy, noise_rate, replace) if return_mask else (noisy, noise_rate)


def denoising_loss(logits, target, valid):
    loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    return loss[valid].mean()


def training_corruption(clean, valid, config, generator):
    """Choose supervision explicitly; legacy configs retain their original RNG path.

    Resampled-only supervision removes the reward for reproducing untouched labels.
    Coincidental matches MUST remain supervised: excluding them would teach a binary
    gripper to always flip its input, since every supervised label would be opposite.
    """
    scope = config.get("denoising_loss_scope", "all_valid")
    if scope not in {"all_valid", "resampled_only"}:
        raise ValueError(f"Unknown denoising loss scope: {scope}")
    if not valid.any():
        raise ValueError("A training chunk must contain a valid target")
    if scope == "all_valid":
        noisy, _ = corrupt_actions(clean, config, generator)
        return noisy, valid
    noisy, _, replaced = corrupt_actions(clean, config, generator, return_mask=True)
    mask = valid & replaced
    if not mask.any():
        # Rare near-zero-noise draws must still yield a finite, nonempty objective.
        locations = valid.nonzero(as_tuple=False)
        choice = torch.randint(len(locations), (), device=clean.device, generator=generator)
        batch, token = locations[choice]
        noisy = noisy.clone()
        support = 2 if int(token) % 7 == 6 else config["bins"]
        value = torch.randint(support, (), device=clean.device, generator=generator)
        noisy[batch, token] = value * (config["bins"] - 1) if support == 2 else value
        mask = mask.clone()
        mask[batch, token] = True
    return noisy, mask


@torch.no_grad()
def sample_actions(model, cache, config, steps, generator, *, return_probabilities=False, step_observer=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    shape = (1, config["horizon"] * config["action_dim"])
    device = next(model.parameters()).device
    noisy = random_actions(shape, config, device, generator)
    for step in range(steps):
        logits = action_logits(model, cache, noisy, config)
        if step_observer is not None:
            # Isolated snapshots: diagnostics cannot mutate the canvas or sampling logits.
            # The callback must not consume the sampler's generator.
            step_observer(step, noisy.detach().clone(), logits.softmax(-1).detach())
        if step == steps - 1:
            ids = logits.argmax(-1)
            # Observe the existing final readout; no extra forward or RNG draw.
            return (ids, logits.softmax(-1)) if return_probabilities else ids
        clean = torch.multinomial(logits.softmax(-1).reshape(-1, config["bins"]), 1,
                                  generator=generator).reshape(shape)
        # A simple uniform re-noising schedule, not vLLM's native entropy sampler.
        probability = 1 - (step + 1) / steps
        replace = torch.rand(shape, device=device, generator=generator) < probability
        noisy = torch.where(replace, random_actions(shape, config, device, generator), clean)
