"""Typed questions and direct probability readouts, without generated-text parsing.

Inspired by the Decisions interface, not an OpenAI-compatible service. Confidence
here means entropy concentration, not OpenAI's proprietary confidence calculation.
"""
import hashlib
import json
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

SCHEMA = "decisionvla.decisions.v1"
ROBOT_PROFILE = "libero.actions.v1"
AXES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def distribution(probabilities, count):
    p = np.asarray(probabilities, dtype=np.float64)
    if (count < 1 or p.shape != (count,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.isclose(p.sum(), 1, atol=2e-6, rtol=0)):
        raise ValueError("Expected a normalized finite distribution matching the question")
    return p


def entropy_confidence(p):
    if len(p) == 1:
        return 1.0
    nonzero = p[p > 0]
    return float(np.clip(1 + (nonzero * np.log(nonzero)).sum() / np.log(len(p)), 0, 1))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Question:
    type: str
    instructions: str
    name: str | None
    values: tuple
    descriptions: tuple[str, ...]

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("A question must be an object")
        kind = raw.get("type")
        kind = "predicate" if kind == "noul" else kind
        fields = {"type", "instructions", "name"}
        if kind == "choice":
            fields.add("choices")
        elif kind == "score":
            fields.add("levels")
        elif kind != "predicate":
            raise ValueError("Supported types: predicate (alias noul), choice, score")
        if set(raw) - fields or not _text(raw.get("instructions")):
            raise ValueError("Questions require text instructions and only documented fields")
        name = raw.get("name")
        if "name" in raw and not _text(name):
            raise ValueError("Question name must be a nonempty string")
        if kind == "predicate":
            values, descriptions = (True, False), ("Condition is true", "Condition is false")
        else:
            options = raw.get("choices" if kind == "choice" else "levels")
            key = "value" if kind == "choice" else "label"
            limit = 256 if kind == "choice" else 10
            if not isinstance(options, list) or not 2 <= len(options) <= limit:
                raise ValueError(f"{kind} requires 2–{limit} options")
            if any(not isinstance(o, dict) or set(o) - {key, "description"} or key not in o
                   or ("description" in o and not isinstance(o["description"], str)) for o in options):
                raise ValueError("Invalid option fields")
            values = tuple(o[key] for o in options)
            if kind == "choice":
                if any(not (_text(v) or type(v) is bool) for v in values):
                    raise ValueError("Choice values must be nonempty strings or booleans")
            elif any(not _text(v) for v in values):
                raise ValueError("Score labels must be nonempty strings")
            if len({canonical(v) for v in values}) != len(values):
                raise ValueError("Duplicate options")
            descriptions = tuple(o.get("description", "") for o in options)
        return cls(kind, raw["instructions"], name, tuple(values), tuple(descriptions))

    def as_dict(self):
        result = {"type": self.type, "instructions": self.instructions}
        if self.name is not None:
            result["name"] = self.name
        if self.type != "predicate":
            key = "value" if self.type == "choice" else "label"
            result["choices" if self.type == "choice" else "levels"] = [
                {key: v, "description": d} for v, d in zip(self.values, self.descriptions, strict=True)]
        return result

    def read(self, probabilities):
        p = distribution(probabilities, len(self.values))
        result = {"type": self.type}
        if self.name is not None:
            result["name"] = self.name
        if self.type == "predicate":
            result["probability"] = float(p[0])
        else:
            if self.type == "choice":
                result["choice"] = self.values[int(p.argmax())]
                result["probabilities"] = [{"value": v, "probability": float(x)}
                                           for v, x in zip(self.values, p, strict=True)]
            else:
                result["score"] = float(np.arange(len(p)) @ p)
                result["probabilities"] = [{"value": i, "label": label, "probability": float(p[i])}
                                           for i, label in enumerate(self.values)]
            result["confidence"] = entropy_confidence(p)
        return result


@dataclass(frozen=True)
class DecisionSchema:
    questions: tuple[Question, ...]

    @classmethod
    def parse(cls, questions):
        if not isinstance(questions, list) or not 1 <= len(questions) <= 56:
            raise ValueError("DecisionVLA supports 1–56 questions per canvas")
        parsed = tuple(Question.parse(q) for q in questions)
        names = [q.name for q in parsed if q.name is not None]
        if len(set(names)) != len(names):
            raise ValueError("Question names must be unique")
        return cls(parsed)

    @cached_property
    def sha256(self):
        return hashlib.sha256(canonical(self.as_list()).encode()).hexdigest()

    def as_list(self):
        return [q.as_dict() for q in self.questions]

    def read(self, probabilities):
        if len(probabilities) != len(self.questions):
            raise ValueError("One distribution is required per question")
        return [q.read(p) for q, p in zip(self.questions, probabilities, strict=True)]

    def prompt(self, instruction, label_ids):
        """Deduplicate identical legends; preserve question and option order."""
        if not _text(instruction) or len(label_ids) != len(self.questions):
            raise ValueError("Expected shared instruction and matching label bindings")
        if self == robot_schema():
            # Exact arithmetic expansion of the fixed 56 questions, not 48
            # repeated 256-option legends in every observation's prefix.
            return ("Use the shared image and task to decide the next robot action chunk. "
                    "Each canvas slot is one typed answer; do not generate explanations.\n"
                    + canonical({"task": instruction, "profile": ROBOT_PROFILE,
                        "order": "step 0 through 7; within each step dx,dy,dz,droll,dpitch,dyaw,gripper_positive",
                        "continuous": {"type": "choice", "value": "bin i, integer 0 through 255",
                            "control": "-1 + 2*i/255 (normalized LIBERO delta control)",
                            "label_token_id": f"{label_ids[0][0]} + i"},
                        "gripper_positive": {"type": "predicate", "condition": "command is +1 rather than -1",
                            "true_token_id": label_ids[6][0], "false_token_id": label_ids[6][1]},
                        "execution": "modal continuous bins; gripper +1 if P(true)>0.5, otherwise -1"}))
        legends, questions, seen = [], [], {}
        for question, ids in zip(self.questions, label_ids, strict=True):
            if len(ids) != len(question.values):
                raise ValueError("Label count must match question support")
            options = [{"token_id": token, "value": value, "description": description}
                       for token, value, description in zip(ids, question.values, question.descriptions, strict=True)]
            key = canonical(options)
            if key not in seen:
                seen[key] = len(legends)
                legends.append(options)
            questions.append({"name": question.name, "type": question.type,
                              "instructions": question.instructions, "legend": seen[key]})
        return ("Evaluate each typed question using the shared image and task. "
                "Output one answer label per corresponding canvas slot, in question order. "
                "Labels are vocabulary token IDs; do not generate explanations.\n"
                + canonical({"task": instruction, "legends": legends, "questions": questions}))


@lru_cache(maxsize=1)
def robot_schema():
    """56 supervised command questions; no synthetic progress or success target."""
    choices = [{"value": str(i), "description": f"Normalized control {2 * i / 255 - 1:.8f}"}
               for i in range(256)]
    questions = []
    for step in range(8):
        for axis in AXES:
            questions.append({"type": "choice", "name": f"action.{step}.{axis}",
                              "instructions": f"Select the {axis} control bin at future step {step}.",
                              "choices": choices})
        questions.append({"type": "predicate", "name": f"action.{step}.gripper_positive",
                          "instructions": f"Should the gripper command at future step {step} be +1 rather than -1?"})
    return DecisionSchema.parse(questions)


def read_decision(question, probabilities):
    return Question.parse(question).read(probabilities)


def action_decisions(probabilities):
    """Canonical answers from a question-conditioned robot distribution."""
    from .confidence import summarize

    p = np.asarray(probabilities, dtype=np.float64)
    summarize(p)
    distributions = [row if i % 7 != 6 else row[[255, 0]] for i, row in enumerate(p.reshape(56, 256))]
    return {"schema": SCHEMA, "profile": ROBOT_PROFILE, "calibrated": False,
            "confidence_definition": "1 - entropy / log(support_size)",
            "answers": robot_schema().read(distributions)}
