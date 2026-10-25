import copy
import json
import math
from pathlib import Path

import pytest

from decisionvla.continuation import learning_rate_at, validate_extension, validate_resume_config


def test_kernel_migration_cannot_change_training_budget_data_or_optimizer():
    config = json.loads(Path("configs/libero_spatial_r32_b64_100k.json").read_text())
    parent = {**config, "step": 38, "format": "decisionvla-shared-lora-v2"}
    new = {**config, "expert_kernel": "triton_sm120_backward"}
    assert validate_resume_config(parent, config) == {}
    with pytest.raises(ValueError):
        validate_resume_config(parent, new)
    assert validate_resume_config(parent, new, allow_kernel_migration=True) == {
        "expert_kernel": {"before": "torch_grouped_mm", "after": "triton_sm120_backward"}}
    for key, value in [("learning_rate", 0.1), ("steps", 50000), ("microbatch", 8),
                       ("seed", 43), ("action_offset", 0), ("denoising_loss_scope", "all_valid")]:
        with pytest.raises(ValueError):
            validate_resume_config(parent, {**new, key: value}, allow_kernel_migration=True)
    with pytest.raises(ValueError):
        validate_resume_config({**parent, "unexpected": True}, new, allow_kernel_migration=True)


def test_original_schedule_is_unchanged_and_extension_uses_local_steps():
    original = json.loads(Path("configs/libero_spatial_r32_all.json").read_text())
    for step in [1, 100, 1000, 2000]:
        warmup = min(step / 100, 1)
        progress = max(0, step - 100) / 1900
        expected = 1e-4 * warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        assert learning_rate_at(step, original) == expected
    extended = {**original, "steps": 10000, "schedule_start_step": 2000, "learning_rate": 5e-5}
    assert learning_rate_at(2001, extended) == pytest.approx(5e-7)
    assert learning_rate_at(2100, extended) == pytest.approx(5e-5)
    assert learning_rate_at(10000, extended) == pytest.approx(5e-6)
    extended["warmup_start_learning_rate"] = 1e-5
    assert learning_rate_at(2001, extended) == pytest.approx(1.04e-5)
    with pytest.raises(ValueError):
        learning_rate_at(2000, extended)


def test_extension_rejects_silent_architecture_or_data_changes():
    previous = {**json.loads(Path("configs/libero_spatial_r32_all.json").read_text()), "step": 2000,
                "format": "decisionvla-shared-lora-v2"}
    old = copy.deepcopy(previous)
    config = {k: v for k, v in previous.items() if k not in {"step", "format"}}
    config.update(steps=10000, schedule_start_step=2000, learning_rate=5e-5)
    assert "schedule_start_step" in validate_extension(previous, config)
    for key, value in [("lora_rank", 64), ("action_offset", 0), ("gradient_accumulation", 16),
                       ("dataset_revision", "other")]:
        with pytest.raises(ValueError, match="architecture/data semantics"):
            validate_extension(previous, {**config, key: value})
    with pytest.raises(ValueError, match="explicitly"):
        validate_extension(previous, {**config, "schedule_start_step": 0})
    assert previous == old


def test_continuation_training_rollout_and_analysis_agree_on_checkpoint_and_objective():
    folder = Path("configs")
    train = json.loads((folder / "libero_spatial_r32_corr_10k.json").read_text())
    rollout = json.loads((folder / "eval_libero_spatial_confidence_10k.json").read_text())
    analysis = json.loads((folder / "confidence_analysis_10k.json").read_text())
    assert train["steps"] == rollout["expected_checkpoint_step"] == analysis["checkpoint_step"] == 10000
    assert train["denoising_loss_scope"] == "resampled_only"
    assert train["lora_rank"] == 32 and train["gradient_accumulation"] == 8
    assert len(rollout["task_ids"]) * len(rollout["initial_state_ids"]) == analysis["episodes"] == 500


def test_disposable_ddp_checkpoint_cannot_seed_production(tmp_path):
    from decisionvla.train_distributed import restore_checkpoint

    config = {"steps": 100000}
    (tmp_path / "adapter_config.json").write_text(json.dumps({**config, "step": 43}))
    (tmp_path / "training_metadata.json").write_text('{"diagnostic_only": true}')
    with pytest.raises(ValueError, match="Disposable"):
        restore_checkpoint(None, None, tmp_path, config)
