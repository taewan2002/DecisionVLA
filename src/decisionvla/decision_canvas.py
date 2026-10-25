"""Compile typed questions into native single-token answer slots."""
import json
from dataclasses import dataclass

import numpy as np
import torch

from .action_canvas import ActionCanvas
from .typed_decisions import robot_schema


@dataclass(frozen=True)
class DecisionCanvas(ActionCanvas):
    label_ids: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if len(self.label_ids) != len(self.slots) or any(
                not ids or len(ids) > self.bins or len(set(ids)) != len(ids)
                or min(ids) < self.action_start or max(ids) >= self.action_start + self.bins
                for ids in self.label_ids):
            raise ValueError("Each question requires distinct labels in the adapted token bank")

    @classmethod
    def compile(cls, schema, tokenizer, *, eos=1, vocab_size=262144, action_start=261888):
        is_robot = schema == robot_schema()
        labels = tuple((action_start + 255, action_start) if is_robot and i % 7 == 6
                       else tuple(range(action_start, action_start + len(q.values)))
                       for i, q in enumerate(schema.questions))
        tokens = tokenizer.encode('{"answers":["', add_special_tokens=False)
        slots = []
        for i, ids in enumerate(labels):
            if i:
                tokens += tokenizer.encode('\",\"', add_special_tokens=False)
            slots.append(len(tokens))
            tokens.append(ids[0])
        tokens += tokenizer.encode('\"]}', add_special_tokens=False)
        if len(tokens) > 256:
            raise ValueError("Questions do not fit the 256-token native canvas")
        expected = {"answers": [tokenizer.convert_ids_to_tokens(ids[0]) for ids in labels]}
        if tokenizer.decode(tokens) != json.dumps(expected, separators=(",", ":")):
            raise ValueError("Tokenizer cannot encode this fixed answer layout")
        if set(sum(labels, ())) & set(tokenizer.all_special_ids):
            raise ValueError("Answer labels cannot be special tokens")
        return cls(tuple(tokens + [eos] * (256 - len(tokens))), tuple(slots), vocab_size,
                   action_start, 256, action_dim=1, label_ids=labels)

    def readout(self, scores):
        values = scores[:, self.slots, :].float()
        indices = torch.tensor([list(ids) + [ids[0]] * (self.bins - len(ids)) for ids in self.label_ids],
                               device=scores.device)
        mask = torch.arange(self.bins, device=scores.device)[None] < torch.tensor(
            [len(ids) for ids in self.label_ids], device=scores.device)[:, None]
        allowed = values.gather(-1, indices[None].expand(len(scores), -1, -1)).masked_fill(~mask, -torch.inf)
        log_mass = allowed.logsumexp(-1) - values.logsumexp(-1)
        return {"bins": allowed.argmax(-1), "probabilities": allowed.softmax(-1),
                "allowed_mass": log_mass.exp(), "allowed_log_mass": log_mass}

    def targets(self, options):
        if options.ndim != 2 or options.shape[1] != len(self.slots) or options.dtype != torch.long:
            raise ValueError("Expected int64 option indices [batch, questions]")
        counts = torch.tensor([len(ids) for ids in self.label_ids], device=options.device)
        if torch.any((options < 0) | (options >= counts)):
            raise ValueError("Target option outside question support")
        labels = torch.tensor([list(ids) + [ids[0]] * (self.bins - len(ids)) for ids in self.label_ids],
                              device=options.device)
        template, _ = self.tensors(options.device)
        result = template.expand(len(options), -1).clone()
        result[:, self.slots] = labels[None].expand(len(options), -1, -1).gather(-1, options[..., None])[..., 0]
        return result


def prepare_decision_observation(processor, image, instruction, schema, canvas, visual_tokens=280):
    """Shared training/inference prefix; question meaning reaches the model."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": schema.prompt(instruction, canvas.label_ids)},
    ]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt",
                                         processor_kwargs={"images_kwargs": {"max_soft_tokens": visual_tokens}})


def robot_option_targets(action_bins):
    if action_bins.shape[-1] != 56:
        raise ValueError("Expected flattened 8x7 robot targets")
    result = action_bins.clone()
    grip = result[..., 6::7]
    if torch.any((grip != 0) & (grip != 255)):
        raise ValueError("Gripper targets must be endpoint bins")
    result[..., 6::7] = (grip == 0).long()
    return result


def robot_action_probabilities(probabilities):
    p = np.asarray(probabilities).reshape(8, 7, 256)
    result = p.copy()
    result[:, 6] = 0
    result[:, 6, 255], result[:, 6, 0] = p[:, 6, 0], p[:, 6, 1]
    return result
