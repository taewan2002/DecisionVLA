"""Python client for the local native service; no OpenAI credentials or HTTP API."""
import base64
import io

from .rpc import request
from .typed_decisions import SCHEMA


class DecisionClient:
    def __init__(self, socket_path):
        self.path = socket_path
        self.metadata = request(self.path, {"op": "metadata"})["metadata"]
        if self.metadata.get("decision_schema") != SCHEMA:
            raise ValueError("Server does not implement the typed decision contract")

    def create(self, image, instruction, *, questions=None, steps=4, seed=42, experimental=False):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return request(self.path, {"op": "decide", "image_png": base64.b64encode(buffer.getvalue()).decode(),
                                  "instruction": instruction, "questions": questions, "steps": steps,
                                  "seed": seed, "experimental": experimental})
