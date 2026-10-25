"""Local Unix-socket transport between the isolated policy and simulator runtimes."""
import base64
import io
import json
import socket
import struct

import numpy as np

MAX_MESSAGE = 2 * 1024 * 1024


def receive_exact(connection, length):
    parts = []
    while length:
        part = connection.recv(length)
        if not part:
            raise EOFError("Policy connection closed")
        parts.append(part)
        length -= len(part)
    return b"".join(parts)


def receive(connection):
    length, = struct.unpack("!I", receive_exact(connection, 4))
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("Invalid policy message length")
    return json.loads(receive_exact(connection, length))


def send(connection, value):
    payload = json.dumps(value, allow_nan=False).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("Policy message too large")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def request(path, value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(180)
        connection.connect(str(path))
        send(connection, value)
        reply = receive(connection)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "Unknown policy error"))
    return reply


def validate_actions(actions, horizon=8):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
        raise ValueError("Expected a finite [8, 7] action chunk")
    if np.any(np.abs(actions) > 1.00001) or not np.isin(actions[:, 6], [-1, 1]).all():
        raise ValueError("Invalid native LIBERO control range or gripper sign")
    return actions


class PolicyClient:
    def __init__(self, path):
        self.path = path
        self.metadata = request(path, {"op": "metadata"})["metadata"]

    def predict(self, image, instruction, steps, seed):
        response = self._predict(image, instruction, steps, seed, False)
        return validate_actions(response["actions"]), response["policy_seconds"]

    def predict_with_confidence(self, image, instruction, steps, seed, verify_parity=False):
        from .confidence import validate_diagnostics

        response = self._predict(image, instruction, steps, seed, True, verify_parity)
        actions = validate_actions(response["actions"])
        if verify_parity and response.get("action_parity_verified") is not True:
            raise ValueError("Policy did not verify action parity")
        return actions, response["policy_seconds"], validate_diagnostics(response["diagnostics"], actions)

    def _predict(self, image, instruction, steps, seed, diagnostics, verify_parity=False):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        response = request(self.path, {"op": "predict", "image_png": base64.b64encode(buffer.getvalue()).decode(),
                                       "instruction": instruction, "steps": steps, "seed": seed,
                                       "diagnostics": diagnostics, "verify_parity": verify_parity})
        return response
