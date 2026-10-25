"""Run in requirements-native.txt's isolated runtime; no model download required."""
import copy

import pytest
import torch
import transformers

if transformers.__version__ != "5.11.0":
    pytest.skip("Native reference requires its isolated Transformers 5.11.0 runtime", allow_module_level=True)

from torch.nn import functional as F
from torch.nn.utils import parametrize
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

from decisionvla.action_canvas import ActionCanvas
from decisionvla.native_adapters import install_native_adapters
from decisionvla.native_runtime import check_runtime
from decisionvla.native_sampling import generate_actions, generation_config, native_canvas_sampler
from decisionvla.native_training import corrupt_canvas, native_training_loss, selected_ce


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = DiffusionGemmaConfig(text_config=dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        num_global_key_value_heads=1, layer_types=["sliding_attention", "full_attention"],
        sliding_window=64, num_experts=4, top_k_experts=2, moe_intermediate_size=16),
        vision_config=dict(model_type="gemma4_vision", hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=2, head_dim=16, patch_size=14),
        canvas_length=16, boi_token_id=4, eoi_token_id=5, image_token_id=6)
    model = DiffusionGemmaForBlockDiffusion(config).eval()
    model.set_experts_implementation("eager")
    model.generation_config = DiffusionGemmaGenerationConfig(
        max_denoising_steps=4, sampler_config=EntropyBoundSamplerConfig(0.1), t_min=0.4, t_max=0.8,
        eos_token_id=1, pad_token_id=0, cache_implementation="dynamic")
    canvas = ActionCanvas((7, 48, 48, 48, 8, 48, 48, 48, 9, 1, 1, 1, 1, 1, 1, 1),
                          (1, 2, 3, 5, 6, 7), 64, 48, 16, action_dim=3)
    inputs = {"input_ids": torch.tensor([[2, 13, 14]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    return model, canvas, inputs


def adapter_config():
    return dict(lora_rank=4, lora_alpha=8, lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_mlp_targets=["gate_proj", "up_proj", "down_proj"], lora_experts=True,
                action_token_start=48, bins=16, train_action_embeddings=True)


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_native_generate_oracle_and_conditioning(tiny, passes):
    model, canvas, inputs = tiny
    check_runtime()
    conditions = []
    hook = model.model.decoder.register_forward_pre_hook(lambda m, a, kw: conditions.append(
        None if kw["self_conditioning_logits"] is None else kw["self_conditioning_logits"].clone()), with_kwargs=True)
    before = torch.random.get_rng_state().clone()
    trace = []
    result = generate_actions(model, inputs, canvas, passes, 31, trajectory=True, trace=trace)
    hook.remove()
    assert torch.equal(before, torch.random.get_rng_state())
    assert conditions[0] is None and len(trace) == passes
    for i in range(1, passes):
        torch.testing.assert_close(conditions[i], trace[i - 1]["processed_logits"].to(model.dtype), rtol=0, atol=0)
    # Independent dense constraint processor, native generate + original sampler.
    mask = torch.zeros(len(canvas.template), canvas.vocab_size, dtype=torch.bool)
    mask[torch.arange(len(canvas.template)), torch.tensor(canvas.template)] = True
    mask[list(canvas.slots)] = True
    logits_seen = []
    oracle_trace = []

    class OracleProcessor:
        def __call__(self, ids, scores):
            logits_seen.append(scores.detach().clone())
            return scores.masked_fill(~mask[None], -torch.inf)

    original_prepare = model._prepare_sampler
    original_stop = model._prepare_diffusion_stopping_criteria

    def prepare(cfg):
        sampler = original_prepare(cfg)
        initialize = sampler.initialize_canvas
        accept, renoise = sampler.accept_canvas, sampler.renoise_canvas

        def init(batch_size, device):
            noise = initialize(batch_size, device)
            fixed = mask.sum(-1) == 1
            return torch.where(fixed[None], torch.tensor(canvas.template)[None], noise)
        sampler.initialize_canvas = init

        def observed_accept(current, proposal, logits, step):
            accepted = accept(current, proposal, logits, step)
            oracle_trace.append({"input": current.clone(), "proposal": proposal.clone(),
                                 "processed_logits": logits.clone(), "accepted": sampler.accepted_token_mask.clone()})
            return accepted

        def observed_renoise(accepted, step):
            next_canvas = renoise(accepted, step)
            oracle_trace[-1]["next_canvas"] = next_canvas.clone()
            return next_canvas

        sampler.accept_canvas, sampler.renoise_canvas = observed_accept, observed_renoise
        return sampler

    model._prepare_sampler = prepare
    model._prepare_diffusion_stopping_criteria = lambda cfg: None
    try:
        torch.manual_seed(31)
        with torch.inference_mode():
            model.generate(**inputs, generation_config=generation_config(model, canvas, passes),
                           logits_processor=LogitsProcessorList([OracleProcessor()]))
    finally:
        model._prepare_sampler, model._prepare_diffusion_stopping_criteria = original_prepare, original_stop
    assert len(logits_seen) == passes
    for i, raw in enumerate(logits_seen):
        for name, value in oracle_trace[i].items():
            torch.testing.assert_close(trace[i][name], value, rtol=0, atol=0)
        expected = canvas.readout(raw)
        for name in expected:
            torch.testing.assert_close(result["trajectory"][i][name], expected[name], rtol=0, atol=0)
        # Upstream computes this in tensor float32, not Python float64.
        temp = 0.4 + 0.4 * (torch.tensor(passes - i, dtype=torch.int32) / passes)
        torch.testing.assert_close(trace[i]["processed_logits"], raw.masked_fill(~mask[None], -torch.inf) / temp,
                                   rtol=0, atol=0)
        probs = trace[i]["processed_logits"].softmax(-1)
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        sorted_entropy, order = entropy.sort(-1)
        accepted = torch.zeros_like(entropy, dtype=torch.bool).scatter(-1, order,
            sorted_entropy.cumsum(-1) - sorted_entropy <= 0.1)
        assert torch.equal(trace[i]["accepted"], accepted)


def test_sampler_restores_on_error(tiny):
    model, canvas, _ = tiny
    before = model._prepare_sampler
    with pytest.raises(RuntimeError), native_canvas_sampler(model, canvas):
        raise RuntimeError("intentional")
    assert model._prepare_sampler == before and "_prepare_sampler" not in model.__dict__


def test_timing_and_optional_observation_do_not_change_actions(tiny):
    model, canvas, inputs = tiny
    original = generate_actions(model, inputs, canvas, 4, 21)
    timing = {}
    observed = generate_actions(model, inputs, canvas, 4, 21, trajectory=True, trace=[], timing=timing)
    for name in original["readout"]:
        torch.testing.assert_close(original["readout"][name], observed["readout"][name], rtol=0, atol=0)
    assert set(timing) == {"prefix_seconds", "denoising_readout_seconds"}
    assert all(value > 0 for value in timing.values())


def test_full_vocabulary_and_terminal_mass(tiny):
    _, canvas, _ = tiny
    logits = torch.zeros(1, 16, 64)
    logits[..., 20] = 20  # forbidden action dominates intermediate distribution
    processed = canvas.pin_logits(logits)
    assert (processed[:, canvas.slots].argmax(-1) == 20).all()
    readout = canvas.readout(logits)
    assert (readout["allowed_mass"] < 1e-6).all()
    assert (readout["probabilities"][:, 2::3, 1:-1] == 0).all()
    torch.testing.assert_close(readout["probabilities"].sum(-1), torch.ones(1, 6))


def test_tied_rows_all_three_paths_and_native_zero_adapter(tiny):
    model, canvas, inputs = tiny
    original = copy.deepcopy(model)
    install_native_adapters(model, adapter_config())
    a = generate_actions(original, inputs, canvas, 4, 33)
    b = generate_actions(model, inputs, canvas, 4, 33)
    for key in a["readout"]:
        torch.testing.assert_close(a["readout"][key], b["readout"][key], rtol=0, atol=0)
    embedding = model.model.decoder.embed_tokens
    delta = embedding.parametrizations.weight[0].delta
    with torch.no_grad():
        delta.normal_(std=0.1)
    with parametrize.cached():
        weight = original.model.decoder.embed_tokens.weight.clone()
        weight[48:] += delta
        ids = torch.tensor([[13, 48, 63]])
        torch.testing.assert_close(embedding(ids), F.embedding(ids, weight) * embedding.embed_scale, rtol=0, atol=0)
        hidden = torch.randn(1, 6, 32)
        torch.testing.assert_close(model.lm_head(hidden), F.linear(hidden, weight), rtol=0, atol=0)
        probabilities = torch.randn(1, 6, 64).softmax(-1)
        torch.testing.assert_close(probabilities @ embedding.weight, probabilities @ weight, rtol=0, atol=0)
        assert model.model.encoder.language_model.embed_tokens is embedding
        (model.lm_head(hidden).sum() + embedding(ids).sum() + (probabilities @ embedding.weight).sum()).backward()
    assert delta.grad.abs().sum() > 0
    assert embedding.parametrizations.weight.original.grad is None
    assert all(not p.requires_grad for n, p in model.named_parameters() if "router" in n or "vision" in n)


def test_corruption_masks_loss_and_gradients(tiny):
    model, canvas, inputs = tiny
    bins = torch.tensor([[2, 4, 0, 8, 3, 15]])
    valid = torch.tensor([[True, True, True, False, False, False]])
    noisy, clean, selected, _ = corrupt_canvas(canvas, bins, valid, torch.Generator().manual_seed(42), 1)
    assert torch.equal(selected, valid)
    fixed = [p for p in range(16) if p not in canvas.slots]
    assert torch.equal(noisy[:, fixed], clean[:, fixed])
    # Accidental matches remain selected. No noisy != clean mask is used by CE.
    clean_logits = torch.randn(1, 16, 64, requires_grad=True)
    loss = selected_ce(clean_logits, clean, canvas.slots, selected)
    expected = F.cross_entropy(clean_logits[:, canvas.slots[:3]].reshape(-1, 64), clean[:, canvas.slots[:3]].flatten())
    torch.testing.assert_close(loss, expected)
    install_native_adapters(model, adapter_config())
    for probability in (0.0, 1.0):
        model.zero_grad(set_to_none=True)
        loss, audit = native_training_loss(model, inputs, bins, valid, canvas, torch.Generator().manual_seed(8),
                                          full_noise_probability=1, self_condition_probability=probability)
        loss.backward()
        assert torch.isfinite(loss) and audit["self_conditioning"] == bool(probability)
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        assert any("lora_B" in n and g is not None and g.abs().sum() > 0 for n, g in grads.items())
        assert all(g is None or torch.isfinite(g).all() for g in grads.values())


def test_adapter_optimizer_rng_resume_is_exact(tiny, tmp_path):
    from decisionvla.checkpoints import load_adapter, save_adapter

    model, canvas, inputs = tiny
    install_native_adapters(model, adapter_config())
    initial = copy.deepcopy(model)
    bins = torch.tensor([[2, 4, 0, 8, 3, 15]])
    valid = torch.ones_like(bins, dtype=torch.bool)

    def setup(m):
        return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, foreach=False)

    def update(m, optimizer, generator):
        optimizer.zero_grad(set_to_none=True)
        loss, audit = native_training_loss(m, inputs, bins, valid, canvas, generator)
        loss.backward()
        optimizer.step()
        return loss.detach().clone(), audit

    optimizer = setup(model)
    generator = torch.Generator().manual_seed(100)
    for _ in range(2):
        update(model, optimizer, generator)
    save_adapter(model, tmp_path)
    torch.save({"optimizer": optimizer.state_dict(), "rng": generator.get_state()}, tmp_path / "resume.pt")
    expected_loss, _ = update(model, optimizer, generator)
    load_adapter(initial, tmp_path)
    resumed_optimizer = setup(initial)
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_rng = torch.Generator().set_state(state["rng"])
    actual_loss, _ = update(initial, resumed_optimizer, resumed_rng)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for a, b in zip(model.parameters(), initial.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), resumed_rng.get_state())
