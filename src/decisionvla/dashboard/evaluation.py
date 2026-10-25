"""Optional LIBERO telemetry; outcomes come from the simulator, never confidence."""
import datetime as dt
import functools
import json
import logging
from pathlib import Path

from .telemetry import Recorder, atomic_json

LOGGER = logging.getLogger(__name__)


def best_effort(method):
    """A telemetry failure must not change an episode's actions or outcome."""
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - observation must not stop evaluation
            self.recorder.errors += 1
            self.recorder.last_error = repr(error)
            LOGGER.warning("Evaluation telemetry failed: %s", error)
            return None
    return call


class EvaluationRecorder:
    """One serial evaluation worker, with a separate, non-resumable viewer session."""

    def __init__(self, directory, metadata, *, expected=None, queue_size=4):
        self.recorder = Recorder(directory, {**metadata, "source": "live_inference"}, queue_size=queue_size)
        self.directory = Path(directory)
        self.episodes = {}
        self.expected, self.status, self.active = expected, "running", None
        try:
            self._save()
        except OSError:
            self.recorder.close()
            raise

    def _save(self):
        atomic_json(self.directory / "evaluation.json", {
            "status": self.status, "expected": self.expected, "active": self.active,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(), "episodes": list(self.episodes.values()),
        })

    @best_effort
    def begin(self, key, *, suite, task_id, initial_id, instruction, max_steps):
        if key in self.episodes:
            raise ValueError("Duplicate episode in this viewer session")
        self.active = key
        self.episodes[key] = {"episode": key, "suite": suite, "task_id": task_id,
                              "initial_state_id": initial_id, "instruction": instruction,
                              "status": "running", "success": None, "first_confidence": None,
                              "policy_queries": 0, "policy_steps": 0, "max_steps": max_steps}
        self._save()

    @best_effort
    def query(self, image, instruction, chunk, stats, seconds, *, query, control_step, seed, steps):
        row = self.episodes[self.active]
        if query == 0:
            row["first_confidence"] = float(stats["chunk_confidence"])
        row.update(policy_queries=query + 1, policy_steps=control_step)
        context = {k: row[k] for k in ["episode", "suite", "task_id", "initial_state_id", "max_steps"]}
        context.update(query=query, control_step=control_step)
        self.recorder.submit(image, instruction, {"actions": chunk, "diagnostics": stats}, seconds,
                             seed=seed, steps=steps, context=context)

    @best_effort
    def finish(self, result):
        row = self.episodes[self.active]
        row.update(status="completed", success=bool(result["success"]),
                   policy_queries=result["policy_queries"], policy_steps=result["policy_steps"])
        self.active = None
        self._save()

    @best_effort
    def restore(self, key, result, trace, config):
        """Count verified reused episodes without inventing missing query images."""
        self.begin(key, suite=config["suite"], task_id=result["task_id"],
                   initial_id=result["initial_state_id"], instruction=result["instruction"],
                   max_steps=config["max_policy_steps"])
        if self.active in self.episodes and "chunk_confidence" in trace:
            self.episodes[self.active]["first_confidence"] = float(trace["chunk_confidence"][0])
        self.finish(result)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.status = "error" if exc_type else "completed"
        if self.active in self.episodes:
            self.episodes[self.active].update(status="error", success=None)
        try:
            self._save()
        except OSError as error:
            LOGGER.warning("Could not save evaluation status: %s", error)
        finally:
            try:
                self.recorder.close()
            except OSError as error:
                LOGGER.warning("Could not close telemetry status: %s", error)


def session_metadata(policy, *, benchmark, selection=None):
    """Use recorded execution metadata; never assume 5090/NVFP4 from the viewer."""
    return {"benchmark": benchmark, "checkpoint_step": policy.get("step"),
            "device": policy.get("device", "See policy metadata"),
            "precision": policy.get("precision", "Not recorded"),
            "execution": policy.get("kind", "Not recorded"),
            "selection": selection, "policy_metadata": policy,
            "latency_scope": "Policy call only; excludes RPC, simulator and telemetry writes"}


def evaluation_view(directory, metadata, index, outcomes):
    """All-episode totals, with an explicit bounded list for interactive inspection."""
    path = Path(directory) / "evaluation.json"
    if path.exists():
        evaluation = json.loads(path.read_text())
        episodes = evaluation["episodes"]
        scope = "worker"
    else:
        # Older verified replay sessions already contain actual episode outcomes.
        grouped = {}
        for frame in index["frames"]:
            context = frame["context"]
            key = context.get("episode")
            if not key:
                continue
            result = outcomes.get(key, {})
            success = context.get("recorded_success", result.get("success"))
            row = grouped.setdefault(key, {"episode": key, "suite": context.get("suite", "LIBERO"),
                "status": "completed" if isinstance(success, bool) else "unknown", "success": success,
                "first_confidence": None, "policy_queries": 0, "policy_steps": None})
            row["policy_queries"] += 1
            if context.get("query") == 0:
                row["first_confidence"] = frame["confidence"]
        episodes = list(grouped.values())
        evaluation = {"status": "recorded", "expected": None, "active": None}
        scope = "recorded_subset"
    completed = [r for r in episodes if r["status"] == "completed" and isinstance(r.get("success"), bool)]
    successes = sum(r["success"] for r in completed)
    suites = []
    for suite in sorted({r["suite"] for r in episodes}):
        selected = [r for r in completed if r["suite"] == suite]
        suites.append({"suite": suite, "completed": len(selected),
                       "successes": sum(r["success"] for r in selected)})
    return {"status": evaluation["status"], "expected": evaluation["expected"],
            "active": evaluation["active"], "scope": scope, "completed": len(completed),
            "successes": successes, "failures": len(completed) - successes,
            "errors": sum(r["status"] == "error" for r in episodes), "suites": suites,
            "episodes": episodes[-200:], "listed_episodes": min(200, len(episodes)),
            "total_episodes": len(episodes), "selection": metadata.get("selection")}
