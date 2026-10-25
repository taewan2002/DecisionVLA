"""Serve a frozen BF16 checkpoint locally; no network port is exposed."""
import argparse
import base64
import io
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import ActionCodec
from .confidence import SCHEMA, summarize
from .rpc import receive, send, validate_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=2000)
    parser.add_argument("--socket-owner", help="Optional numeric UID:GID for a root policy / host client")
    parser.add_argument("--stub", action="store_true", help="CPU transport smoke only; never a policy evaluation")
    args = parser.parse_args()
    owner = tuple(map(int, args.socket_owner.split(":"))) if args.socket_owner else None
    if owner is not None and (len(owner) != 2 or min(owner) < 0):
        parser.error("--socket-owner must be nonnegative UID:GID")
    if args.stub:
        policy = None
        metadata = {"kind": "stub_smoke_only", "step": None}
    else:
        from .checkpoints import adapter_files, file_sha256
        from .policy import DecisionVLA

        config = json.loads((args.adapter / "adapter_config.json").read_text())
        assert config["step"] == args.expected_step
        assert config["horizon"] == 8
        suites = config.get("suites", [config.get("suite")])
        assert set(suites) <= {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        assert config["rotate_image_180"] and config["camera"] == "agentview_rgb"
        files = adapter_files(args.adapter) + [args.adapter / "adapter_config.json"]
        index = args.adapter / "adapter.safetensors.index.json"
        if index.exists():
            files.append(index)
        expected = json.loads((args.adapter / "sha256.json").read_text())
        hashes = {p.name: file_sha256(p) for p in files}
        assert all(expected.get(name) == sha for name, sha in hashes.items())
        policy = DecisionVLA(args.model, args.adapter)
        metadata = {"kind": "decisionvla_bf16_reference", "step": config["step"], "config": config,
                    "checkpoint_sha256": hashes}
    metadata["confidence_schema"] = SCHEMA
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise FileExistsError(args.socket)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(args.socket))
        args.socket.chmod(0o600)
        if owner is not None:
            os.chown(args.socket, *owner)
        listener.listen(2)
        print(json.dumps({"event": "policy_ready", "metadata": metadata}), flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(180)
                try:
                    data = receive(connection)
                    if data["op"] == "metadata":
                        response = {"ok": True, "metadata": metadata}
                    elif data["op"] == "predict":
                        image = Image.open(io.BytesIO(base64.b64decode(data["image_png"], validate=True)))
                        if image.size != (128, 128):
                            raise ValueError("Evaluation requires the training camera resolution, 128 x 128")
                        image = image.convert("RGB")
                        if data["steps"] != 4:
                            raise ValueError("This evaluation fixes four denoising steps")
                        start = time.perf_counter()
                        diagnostics = None
                        if policy is None:
                            bins = np.full((8, 7), 128)
                            bins[:, 6] = 0
                            actions = ActionCodec(256).decode(bins)
                            if data.get("diagnostics"):
                                diagnostics = summarize(np.eye(256)[bins])
                        elif data.get("diagnostics"):
                            prediction = policy.predict_with_confidence(
                                image, data["instruction"], data["steps"], data["seed"])
                            actions, diagnostics = prediction["actions"], prediction["diagnostics"]
                        else:
                            actions = policy.predict(image, data["instruction"], data["steps"], data["seed"])
                        actions = validate_actions(actions)
                        response = {"ok": True, "actions": actions.tolist(),
                                    "policy_seconds": time.perf_counter() - start}
                        if diagnostics is not None:
                            response["diagnostics"] = diagnostics
                        if data.get("verify_parity"):
                            if policy is None:
                                raise ValueError("A stub cannot verify real-policy parity")
                            original = policy.predict(image, data["instruction"], data["steps"], data["seed"])
                            np.testing.assert_array_equal(actions, original)
                            response["action_parity_verified"] = True
                    else:
                        raise ValueError("Unknown policy operation")
                    send(connection, response)
                except Exception as error:  # noqa: BLE001 - report each request failure to the client
                    traceback.print_exc()
                    send(connection, {"ok": False, "error": repr(error)})


if __name__ == "__main__":
    main()
