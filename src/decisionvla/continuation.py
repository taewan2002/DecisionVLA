"""Explicit schedule extension rules, separate from exact same-config resume."""
import math


def validate_resume_config(previous, config, *, allow_kernel_migration=False):
    """Only an explicit kernel migration may alter an otherwise identical run."""
    old = {k: v for k, v in previous.items() if k not in {"step", "format"}}
    old.setdefault("expert_kernel", "torch_grouped_mm")
    current = {"expert_kernel": "torch_grouped_mm", **config}
    changed = {key: {"before": old.get(key), "after": current.get(key)}
               for key in old.keys() | current.keys() if old.get(key) != current.get(key)}
    permitted = {
        ("torch_grouped_mm", "triton_sm120"),
        ("torch_grouped_mm", "triton_sm120_lora"),
        ("torch_grouped_mm", "triton_sm120_backward"),
        ("triton_sm120_backward", "cublas_sm120_forward"),
    }
    if changed:
        if (not allow_kernel_migration or set(changed) != {"expert_kernel"}
                or (changed["expert_kernel"]["before"], changed["expert_kernel"]["after"]) not in permitted):
            raise ValueError("Resume requires the identical config; only explicit validated kernel migration is allowed")
    return changed


def training_boundary(config, first, stop_at_step=None):
    """A process boundary does not change the global optimizer/LR schedule."""
    stop = config["steps"] if stop_at_step is None else stop_at_step
    if not first < stop <= config["steps"]:
        raise ValueError("Training boundary must follow the restored step and not exceed the schedule")
    return stop

EXTENDABLE = {"steps", "learning_rate", "warmup_steps", "schedule_start_step",
              "warmup_start_learning_rate",
              "max_training_hours", "validation_every", "validation_samples", "checkpoint_every",
              "denoising_loss_scope"}


def validate_extension(previous, config):
    parent_step = previous["step"]
    if not (config["steps"] > parent_step and config.get("schedule_start_step") == parent_step):
        raise ValueError("Extension must explicitly start its new schedule at the parent step")
    old = {k: v for k, v in previous.items() if k not in {"step", "format"}}
    changed = {k: {"before": old.get(k), "after": config.get(k)}
               for k in old.keys() | config.keys() if old.get(k) != config.get(k)}
    if changed.keys() - EXTENDABLE:
        raise ValueError(f"Extension cannot change architecture/data semantics: {sorted(changed.keys() - EXTENDABLE)}")
    return changed


def learning_rate_at(step, config):
    local_step = step - config.get("schedule_start_step", 0)
    budget = config["steps"] - config.get("schedule_start_step", 0)
    if not 1 <= local_step <= budget:
        raise ValueError("Step outside the declared schedule")
    warmup = min(local_step / max(1, config["warmup_steps"]), 1.0)
    progress = max(0, local_step - config["warmup_steps"]) / max(1, budget - config["warmup_steps"])
    initial = config.get("warmup_start_learning_rate", 0.0)
    warm_rate = initial + (config["learning_rate"] - initial) * warmup
    return warm_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
