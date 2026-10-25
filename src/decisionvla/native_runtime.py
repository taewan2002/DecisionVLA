"""Fail closed on a different native sampler/model implementation."""
import hashlib
import inspect
from importlib.metadata import version

BASE_MODEL = "google/diffusiongemma-26B-A4B-it"
BASE_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
ARCHITECTURE = "decisionvla.native-decisions.v1"
SOURCE_HASHES = {
    "generation_diffusion_gemma": "b814a6fc41492794c1f50ba71c750206684e25dfdd53932e0842675871fef0de",
    "modeling_diffusion_gemma": "3f1a24f517b51470113c5181a53c3ddb07c05d01b8780a3a6d030497a231e2ec",
}


def check_runtime():
    from transformers.models.diffusion_gemma import generation_diffusion_gemma, modeling_diffusion_gemma

    actual = {}
    for module in (generation_diffusion_gemma, modeling_diffusion_gemma):
        with open(inspect.getfile(module), "rb") as stream:
            actual[module.__name__.split(".")[-1]] = hashlib.file_digest(stream, "sha256").hexdigest()
    runtime = {p: version(p) for p in ("torch", "transformers", "torchvision")}
    if (runtime["transformers"] != "5.11.0" or runtime["torch"].split("+")[0] != "2.14.0"
            or actual != SOURCE_HASHES):
        raise RuntimeError("Use the isolated requirements-native.txt runtime; pilot runtime is incompatible")
    return {"versions": runtime, "source_sha256": actual}


def validate_config(config):
    from .typed_decisions import ROBOT_PROFILE, SCHEMA, robot_schema

    required = {"architecture": ARCHITECTURE, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                "self_conditioning": True, "horizon": 8, "action_dim": 7, "bins": 256,
                "action_token_start": 261888, "canvas_length": 256,
                "decision_schema": SCHEMA, "decision_profile": ROBOT_PROFILE,
                "question_schema_sha256": robot_schema().sha256}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Not a native DecisionVLA config; legacy adapters cannot be imported")
