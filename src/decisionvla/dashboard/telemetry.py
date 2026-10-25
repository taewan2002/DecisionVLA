"""Record existing predictions without changing the sampler or controlling a robot."""
import base64
import copy
import datetime as dt
import io
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

from ..confidence import validate_diagnostics
from ..rpc import validate_actions

SCHEMA = "decisionvla.observatory.v1"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


class SessionWriter:
    """Single-writer session; frames become visible only after an atomic index update."""

    def __init__(self, directory, metadata, *, history=600):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self.metadata = {**metadata, "schema": SCHEMA}
        self.rows, self.count, self.history = [], 0, history
        atomic_json(self.directory / "session.json", self.metadata)
        atomic_json(self.directory / "index.json", {"frames": [], "total": 0, "dropped": 0, "errors": 0})

    def write(self, image, instruction, prediction, seconds, *, seed=42, steps=4, context=None,
              dropped=0, errors=0):
        actions = validate_actions(prediction["actions"])
        diagnostics = validate_diagnostics(prediction["diagnostics"], actions)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Prediction latency must be positive and finite")
        if not isinstance(instruction, str) or len(instruction) > 16000:
            raise ValueError("Invalid instruction")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        event = {"id": self.count, "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
                 "instruction": instruction, "seed": int(seed), "denoising_steps": int(steps),
                 "actions": actions.tolist(), "diagnostics": diagnostics,
                 "policy_seconds": float(seconds), "context": context or {},
                 "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}
        atomic_json(self.directory / "frames" / f"{self.count:06d}.json", event)
        self.rows.append({k: event[k] for k in ["id", "recorded_at", "policy_seconds", "context"]} |
                         {"confidence": diagnostics["chunk_confidence"]})
        self.rows = self.rows[-self.history:]
        self.count += 1
        atomic_json(self.directory / "index.json",
                    {"frames": self.rows, "total": self.count, "dropped": dropped, "errors": errors})
        return event


class Recorder:
    """Bounded asynchronous CPU telemetry. Full queues drop records, never policy calls."""

    def __init__(self, directory, metadata, *, queue_size=4):
        self.writer = SessionWriter(directory, metadata)
        self.pending = queue.Queue(maxsize=queue_size)
        self.dropped = self.errors = 0
        self.last_error = None
        self.closed = False
        self.thread = threading.Thread(target=self._work, daemon=True, name="decisionvla-telemetry")
        self.thread.start()

    def submit(self, image, instruction, prediction, seconds, **kwargs):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        # Snapshot mutable results; no GPU tensors, forward calls or RNG operations.
        try:
            item = (image.copy(), instruction, copy.deepcopy(prediction), seconds, kwargs)
            self.pending.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _work(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                image, instruction, prediction, seconds, kwargs = item
                self.writer.write(image, instruction, prediction, seconds, **kwargs,
                                  dropped=self.dropped, errors=self.errors)
            except Exception as error:  # noqa: BLE001 - telemetry must not interrupt policy execution
                self.errors += 1
                self.last_error = repr(error)
                try:
                    atomic_json(self.writer.directory / "recorder-status.json",
                                {"closed": False, "dropped": self.dropped, "errors": self.errors,
                                 "last_error": self.last_error})
                except OSError:
                    pass  # The in-memory error remains available if the disk is unavailable.
            finally:
                self.pending.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join()
            atomic_json(self.writer.directory / "recorder-status.json",
                        {"closed": True, "dropped": self.dropped, "errors": self.errors,
                         "last_error": self.last_error})

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ObservedPolicy:
    """Wrap direct DecisionVLA or PolicyClient; return its original result unchanged."""

    def __init__(self, policy, recorder):
        self.policy, self.recorder = policy, recorder

    def predict_with_confidence(self, image, instruction, steps=4, seed=42, **kwargs):
        start = time.perf_counter()
        result = self.policy.predict_with_confidence(image, instruction, steps, seed, **kwargs)
        seconds = time.perf_counter() - start
        if isinstance(result, dict):
            prediction = result
        else:
            actions, seconds, diagnostics = result
            prediction = {"actions": actions, "diagnostics": diagnostics}
        self.recorder.submit(image, instruction, prediction, seconds, steps=steps, seed=seed)
        return result
