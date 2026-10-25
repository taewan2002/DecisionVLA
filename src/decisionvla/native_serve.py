"""Local native decision/policy service; immutable checkpoint, 1/2/4 passes."""
import argparse
import base64
import io
import json
import socket
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .confidence import SCHEMA
from .rpc import receive, send


def handle_request(policy, data):
    if not isinstance(data, dict):
        raise ValueError("Expected a request object")
    if data.get("op") == "metadata":
        return {"ok": True, "metadata": {**policy.metadata, "confidence_schema": SCHEMA}}
    op = data.get("op")
    if op not in {"predict", "decide"}:
        raise ValueError("Expected predict, decide or metadata")
    fields = {"op", "image_png", "instruction", "steps", "seed", "diagnostics", "typed_decisions", "verify_parity"}
    if op == "decide":
        fields |= {"questions", "experimental"}
    if set(data) - fields:
        raise ValueError("Unknown request fields")
    steps, seed = data.get("steps", 4), data.get("seed", 42)
    if type(steps) is not int or steps not in (1, 2, 4) or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Expected 1/2/4 passes and a nonnegative int64 seed")
    for name in ("diagnostics", "typed_decisions", "verify_parity", "experimental"):
        if name in data and type(data[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if not isinstance(data.get("instruction"), str) or not data["instruction"].strip():
        raise ValueError("Expected a nonempty instruction")
    with Image.open(io.BytesIO(base64.b64decode(data["image_png"], validate=True))) as source:
        if source.size != (128, 128):
            raise ValueError("Expected 128x128 policy image")
        image = source.convert("RGB")
    start = time.perf_counter()
    if op == "decide":
        if data.get("verify_parity"):
            raise ValueError("Use predict for the evaluator's action-parity preflight")
        result = policy.decide(image, data["instruction"], questions=data.get("questions"), steps=steps,
                               seed=seed, experimental=data.get("experimental", False))
    else:
        result = policy.predict_with_confidence(image, data["instruction"], steps, seed,
                                                typed=data.get("typed_decisions", False))
    response = {"ok": True, "policy_seconds": time.perf_counter() - start, "timing": result["timing"]}
    if "actions" in result:
        response["actions"] = result["actions"].tolist()
    if "decisions" in result:
        response.update(result["decisions"])
    if data.get("diagnostics") or op == "decide":
        response["diagnostics"] = result["diagnostics"]
    if data.get("verify_parity"):
        original = policy.predict(image, data["instruction"], steps, seed)
        np.testing.assert_array_equal(result["actions"], original)
        response["action_parity_verified"] = True
    return response


def main():
    from .native_policy import NativeDecisionVLA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.socket.exists():
        raise FileExistsError(args.socket)
    policy = NativeDecisionVLA.load(args.adapter, device=args.device)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        args.socket.chmod(0o600)
        server.listen(2)
        print(json.dumps({"event": "native_policy_ready", "metadata": policy.metadata}), flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(180)
                try:
                    send(connection, handle_request(policy, receive(connection)))
                except Exception as error:  # noqa: BLE001 - errors never become successful decisions
                    send(connection, {"ok": False, "error": repr(error)})


if __name__ == "__main__":
    main()
