# Scoped sampler adaptation: Copyright (c) 2026 Open Jev contributors (MIT).
# Modifications: DecisionVLA contributors. See third_party/open-jev-LICENSE.
"""Native generate(), with only template initialization and exact pass stopping.

Same scoped overrides as JoshuaSP/open-jev's constrained_sampler. No copied
denoising loop: HF owns proposals, entropy acceptance, re-noising and conditioning.
"""
import copy
import dataclasses
import json
import threading
import time
from contextlib import contextmanager

import torch
from torch.nn.utils import parametrize
from transformers import LogitsProcessorList

from .action_canvas import CanvasProcessor

_GENERATION_LOCK = threading.RLock()


@contextmanager
def profile_stages(model, device, measurements):
    """Optional stage timing, without synchronizing between denoising passes."""
    events = []

    def stamp():
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            return event
        return time.perf_counter()

    def begin_encoder(module, args):
        events.append(["prefix_seconds", stamp(), None])

    def end_encoder(module, args, output):
        events[-1][2] = stamp()

    owned = model.__dict__.get("_denoising_step")
    original = model._denoising_step

    def denoise(*args, **kwargs):
        start = stamp()
        output = original(*args, **kwargs)
        events.append(["denoising_readout_seconds", start, stamp()])
        return output

    hooks = [model.model.encoder.register_forward_pre_hook(begin_encoder),
             model.model.encoder.register_forward_hook(end_encoder)]
    model._denoising_step = denoise
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        if owned is None:
            delattr(model, "_denoising_step")
        else:
            model._denoising_step = owned
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for key, start, end in events:
        seconds = start.elapsed_time(end) / 1000 if device.type == "cuda" else end - start
        measurements[key] = measurements.get(key, 0.0) + seconds


@contextmanager
def native_canvas_sampler(model, canvas, trace=None):
    names = ("_prepare_sampler", "_prepare_diffusion_stopping_criteria")
    originals = {name: getattr(model, name) for name in names}
    owned = {name: model.__dict__.get(name) for name in names}

    def prepare(config):
        sampler = originals["_prepare_sampler"](config)
        sampler.initialize_canvas = canvas.initialize_canvas
        if trace is not None:
            accept, renoise = sampler.accept_canvas, sampler.renoise_canvas

            def observed_accept(current, proposal, logits, step):
                result = accept(current, proposal, logits, step)
                trace.append({"remaining": int(step), "input": current.detach().clone(),
                              "proposal": proposal.detach().clone(), "processed_logits": logits.detach().clone(),
                              "accepted": sampler.accepted_token_mask.detach().clone()})
                return result

            def observed_renoise(accepted, step):
                result = renoise(accepted, step)
                trace[-1]["next_canvas"] = result.detach().clone()
                return result

            sampler.accept_canvas, sampler.renoise_canvas = observed_accept, observed_renoise
        return sampler

    model._prepare_sampler = prepare
    model._prepare_diffusion_stopping_criteria = lambda config: None
    try:
        yield
    finally:
        for name in names:
            if owned[name] is None:
                delattr(model, name)
            else:
                setattr(model, name, owned[name])


def generation_config(model, canvas, passes):
    if type(passes) is not int or passes not in (1, 2, 4):
        raise ValueError("Native comparison supports exactly 1, 2 or 4 passes")
    cfg = copy.deepcopy(model.generation_config)
    cfg.max_new_tokens = len(canvas.template)
    cfg.max_length = None
    cfg.max_denoising_steps = passes
    cfg.cache_implementation = "dynamic"
    cfg.cache_config = None
    cfg.validate()
    if cfg.sampler_config is None or cfg.t_min is None or cfg.t_max is None:
        raise ValueError("The base generation config must resolve sampler and temperature")
    return cfg


def serialized_generation_config(config):
    # HF's to_dict leaves the sampler dataclass nested in the result.
    # Keep a real structured value for RPC/audits; never stringify unknown objects.
    def encode(value):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        raise TypeError(f"Unsupported generation setting: {type(value).__name__}")
    return json.loads(json.dumps(config.to_dict(), default=encode, allow_nan=False))


@torch.inference_mode()
def generate_actions(model, inputs, canvas, passes, seed, *, trajectory=False, trace=None, timing=None):
    from contextlib import nullcontext

    cfg = generation_config(model, canvas, passes)
    processor = CanvasProcessor(canvas, passes, trajectory)
    device = inputs["input_ids"].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Native generate uses the global RNG. Isolate and restore it; the policy
    # serializes calls, so concurrent requests cannot corrupt scoped overrides.
    with _GENERATION_LOCK, torch.random.fork_rng(devices=devices), parametrize.cached():
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        with native_canvas_sampler(model, canvas, trace), (
            profile_stages(model, device, timing) if timing is not None else nullcontext()
        ):
            model.generate(**inputs, generation_config=cfg, logits_processor=LogitsProcessorList([processor]))
    if processor.steps != passes or processor.final is None:
        raise RuntimeError(f"Expected {passes} native passes, observed {processor.steps}")
    return {"readout": processor.final, "trajectory": processor.readouts,
            "generation_config": serialized_generation_config(cfg), "passes": processor.steps,
            "canvas_sha256": canvas.sha256}
