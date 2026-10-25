"""Single-token robot labels; unrestricted intermediate vocabulary, terminal readout.

The fixed token template is audited against the pinned base tokenizer. Never
tokenize the printable unused-token aliases: they do not round-trip as one ID.
"""
import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files

import torch


@dataclass(frozen=True)
class ActionCanvas:
    template: tuple[int, ...]
    slots: tuple[int, ...]
    vocab_size: int
    action_start: int
    bins: int
    action_dim: int = 7

    def __post_init__(self):
        if (not self.template or not self.slots or len(set(self.slots)) != len(self.slots)
                or min(self.slots) < 0 or max(self.slots) >= len(self.template)
                or min(self.template) < 0 or max(self.template) >= self.vocab_size
                or self.bins < 2 or self.action_start < 0
                or self.action_start + self.bins > self.vocab_size
                or len(self.slots) % self.action_dim):
            raise ValueError("Invalid action canvas")

    @classmethod
    def robot(cls, eos=1):
        spec = json.loads(files("decisionvla").joinpath("native_canvas.json").read_text())
        template = spec["template_token_ids"]
        return cls(tuple(template + [eos] * (256 - len(template))), tuple(spec["slots"]),
                   spec["vocabulary_size"], spec["action_token_ids"][0], 256)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    def tensors(self, device):
        template = torch.tensor(self.template, device=device)
        variable = torch.zeros(len(self.template), dtype=torch.bool, device=device)
        variable[list(self.slots)] = True
        return template, variable

    def initialize_canvas(self, batch_size, device):
        # Draw the SAME full-canvas RNG stream as the reference before pinning syntax.
        noise = torch.randint(self.vocab_size, (batch_size, len(self.template)), device=device)
        template, variable = self.tensors(device)
        return torch.where(variable[None], noise, template[None])

    def pin_logits(self, scores):
        if scores.ndim != 3 or scores.shape[1:] != (len(self.template), self.vocab_size):
            raise ValueError("Expected native full-canvas/full-vocabulary logits")
        template, variable = self.tensors(scores.device)
        fixed = ~variable
        selected = scores[:, fixed].gather(-1, template[fixed][None, :, None].expand(scores.shape[0], -1, -1))
        result = scores.clone()
        result[:, fixed] = -torch.inf
        rows = result[:, fixed].scatter(-1, template[fixed][None, :, None].expand(scores.shape[0], -1, -1), selected)
        result[:, fixed] = rows
        return result

    def targets(self, bins):
        if bins.ndim != 2 or bins.shape[1] != len(self.slots):
            raise ValueError("Expected a batch of flattened action bins")
        if torch.any((bins < 0) | (bins >= self.bins)):
            raise ValueError("Action bin outside vocabulary")
        gripper = bins[:, self.action_dim - 1::self.action_dim]
        if torch.any((gripper != 0) & (gripper != self.bins - 1)):
            raise ValueError("Gripper must use endpoint bins")
        template, _ = self.tensors(bins.device)
        result = template.expand(bins.shape[0], -1).clone()
        result[:, self.slots] = bins + self.action_start
        return result

    def readout(self, scores):
        # scores are native softcapped logits BEFORE sampler temperature/layout.
        values = scores[:, self.slots, :]
        allowed = values[..., self.action_start:self.action_start + self.bins].float().clone()
        allowed[:, self.action_dim - 1::self.action_dim, 1:-1] = -torch.inf
        log_mass = allowed.logsumexp(-1) - values.float().logsumexp(-1)
        return {"bins": allowed.argmax(-1), "probabilities": allowed.softmax(-1),
                "allowed_mass": log_mass.exp(), "allowed_log_mass": log_mass}

    def audit_tokenizer(self, tokenizer):
        raw = list(self.template[:243])
        labels = [tokenizer.convert_ids_to_tokens(self.action_start + i) for i in range(self.bins)]
        if any(not s.startswith("<unused") for s in labels):
            raise ValueError("Expected unused native action labels")
        if set(range(self.action_start, self.action_start + self.bins)) & set(tokenizer.all_special_ids):
            raise ValueError("Action labels cannot be special tokens")
        decoded = json.loads(tokenizer.decode(raw))
        if decoded != {"actions": [[labels[0]] * self.action_dim for _ in range(8)]}:
            raise ValueError("Pinned tokenizer/template mismatch")
        return {"canvas_sha256": self.sha256, "slots": len(self.slots), "template_json_verified": True}


class CanvasProcessor:
    """Observe raw readouts without changing the native proposal probabilities."""

    def __init__(self, canvas, passes, trajectory=False):
        self.canvas, self.passes, self.trajectory = canvas, passes, trajectory
        self.steps = 0
        self.readouts = []
        self.final = None

    def __call__(self, input_ids, scores):
        self.steps += 1
        if self.trajectory or self.steps == self.passes:
            self.final = {k: v.detach() for k, v in self.canvas.readout(scores).items()}
            if self.trajectory:
                self.readouts.append({k: v.clone() for k, v in self.final.items()})
        return self.canvas.pin_logits(scores)
