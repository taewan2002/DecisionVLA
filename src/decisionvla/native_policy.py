"""Native DecisionVLA policy, separate from the historical action-only pilot."""
import json
import threading
import time
from pathlib import Path

import torch

from .checkpoints import adapter_files, file_sha256, load_adapter
from .codec import ActionCodec
from .confidence import summarize
from .decision_canvas import DecisionCanvas, prepare_decision_observation, robot_action_probabilities
from .native_adapters import install_native_adapters
from .native_runtime import ARCHITECTURE, check_runtime, validate_config
from .native_sampling import generate_actions
from .typed_decisions import ROBOT_PROFILE, SCHEMA, DecisionSchema, action_decisions, robot_schema


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class NativeDecisionVLA:
    def __init__(self, model, processor, config, *, checkpoint=None):
        validate_config(config)
        self.runtime = check_runtime()
        self.model, self.processor, self.config = model.eval(), processor, config
        self.device = model.device
        eos = model.generation_config.eos_token_id
        self.eos = eos[0] if isinstance(eos, list) else eos
        self.questions = robot_schema()
        self.canvas = DecisionCanvas.compile(self.questions, processor.tokenizer, eos=self.eos)
        if config.get("decision_canvas_sha256") != self.canvas.sha256:
            raise ValueError("Checkpoint decision canvas does not match the pinned tokenizer")
        self.lock = threading.Lock()
        self.metadata = {"kind": ARCHITECTURE, "step": config.get("step"), "config": config,
                         "checkpoint_sha256": checkpoint, "runtime": self.runtime,
                         "canvas_sha256": self.canvas.sha256,
                         "trained": checkpoint is not None, "supported_passes": [1, 2, 4],
                         "decision_schema": SCHEMA, "decision_profile": ROBOT_PROFILE,
                         "question_schema_sha256": self.questions.sha256,
                         "question_types": ["predicate", "choice", "score"],
                         "custom_questions": "explicit_experimental_opt_in_only", "calibrated": False}

    @classmethod
    def load(cls, adapter, *, device="cuda"):
        """Require a native checkpoint and all payload hashes, rejecting legacy weights."""
        check_runtime()
        path = Path(adapter)
        config = json.loads((path / "adapter_config.json").read_text())
        validate_config(config)
        if type(config.get("step")) is not int or config["step"] < 1:
            raise ValueError("Evaluation requires a trained native checkpoint with a positive step")
        expected = json.loads((path / "sha256.json").read_text())
        payload = adapter_files(path) + [path / "adapter_config.json"]
        if (path / "adapter.safetensors.index.json").exists():
            payload.append(path / "adapter.safetensors.index.json")
        hashes = {p.name: file_sha256(p) for p in payload}
        if any(expected.get(name) != sha for name, sha in hashes.items()):
            raise ValueError("Native checkpoint SHA mismatch")
        model, processor, _ = load_fresh_model(config, device)
        load_adapter(model, path)
        return cls(model, processor, config, checkpoint=hashes)

    def predict(self, image, instruction, steps=4, seed=42):
        return self.predict_with_confidence(image, instruction, steps, seed)["actions"]

    @torch.inference_mode()
    def predict_with_confidence(self, image, instruction, steps=4, seed=42, *, trajectory=False, typed=False):
        return self._infer(image, instruction, self.questions, self.canvas, steps, seed,
                           trajectory=trajectory, typed=typed)

    def decide(self, image, instruction, *, questions=None, steps=4, seed=42, experimental=False):
        """Typed decision request. Custom semantics are not learned by LIBERO BC.

        Without questions, use the checkpoint's fixed robot profile. An explicit
        experimental flag permits custom question-conditioned inference, but never
        returns robot controls or claims validated semantic capabilities for it.
        """
        if type(experimental) is not bool:
            raise ValueError("experimental must be boolean")
        schema = self.questions if questions is None else DecisionSchema.parse(questions)
        if schema != self.questions and not experimental:
            raise ValueError("Custom questions are unvalidated; require experimental=True")
        canvas = self.canvas if schema == self.questions else DecisionCanvas.compile(
            schema, self.processor.tokenizer, eos=self.eos)
        return self._infer(image, instruction, schema, canvas, steps, seed, typed=True)

    @torch.inference_mode()
    def _infer(self, image, instruction, schema, canvas, steps, seed, *, trajectory=False, typed=False):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Expected a nonnegative int64 seed")
        with self.lock:
            synchronize(self.device)
            start = time.perf_counter()
            inputs = prepare_decision_observation(self.processor, image, instruction, schema, canvas,
                                                 self.config["visual_tokens"])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            synchronize(self.device)
            prepared = time.perf_counter()
            timing = {}
            result = generate_actions(self.model, inputs, canvas, steps, seed,
                                      trajectory=trajectory, timing=timing)
            synchronize(self.device)
            generated = time.perf_counter()
            readout = result["readout"]
            p = readout["probabilities"][0].cpu().numpy()
            robot = schema == self.questions
            response = {}
            if robot:
                action_p = robot_action_probabilities(p)
                response["actions"] = ActionCodec(256).decode(action_p.argmax(-1))
                diagnostics = summarize(action_p)
                shape = (8, 7)
            else:
                diagnostics, shape = {}, (len(schema.questions),)
            diagnostics.update(allowed_mass=readout["allowed_mass"].cpu().reshape(shape).tolist(),
                allowed_log_mass=readout["allowed_log_mass"].cpu().reshape(shape).tolist(),
                probability_source="native_softcapped_logits_before_layout_and_temperature",
                architecture=ARCHITECTURE, passes=steps, seed=seed, canvas_sha256=canvas.sha256,
                question_schema_sha256=schema.sha256, prefix_tokens=int(inputs["input_ids"].shape[-1]),
                generation_config=result["generation_config"])
            if trajectory:
                diagnostics["trajectory"] = [{k: v.cpu().tolist() for k, v in r.items()}
                                             for r in result["trajectory"]]
            if typed:
                if robot:
                    decisions = action_decisions(action_p)
                else:
                    decisions = {"schema": SCHEMA, "profile": "custom.experimental", "calibrated": False,
                        "confidence_definition": "1 - entropy / log(support_size)",
                        "answers": schema.read([row[:len(q.values)] for row, q in zip(p, schema.questions, strict=True)])}
                decisions.update(question_schema_sha256=schema.sha256,
                                 semantic_validation="pending" if robot else "unvalidated_custom_questions")
                response["decisions"] = decisions
            end = time.perf_counter()
            return {**response, "diagnostics": diagnostics,
                    "timing": {**timing, "preprocess_seconds": prepared - start,
                               "native_generate_seconds": generated - prepared,
                               "readout_cpu_seconds": end - generated, "total_seconds": end - start}}


def load_fresh_model(config, device="cuda"):
    """Fresh base + new adapters only. Does not start or resume training."""
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    validate_config(config)
    check_runtime()
    processor = AutoProcessor.from_pretrained(config["base_model"], revision=config["base_revision"])
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        config["base_model"], revision=config["base_revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="eager").to(device)
    resolved_device = model.device
    devices = [resolved_device.index] if resolved_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(config["seed"])
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(config["seed"])
        info = install_native_adapters(model, config)
    return model, processor, info
