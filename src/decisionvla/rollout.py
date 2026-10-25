"""LIBERO-Spatial closed-loop evaluation in a CPU/OSMesa simulator runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .confidence import METRICS, SCHEMA
from .rpc import PolicyClient, validate_actions


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def camera_image(observation):
    raw = np.asarray(observation["agentview_image"])
    if raw.shape != (128, 128, 3) or raw.dtype != np.uint8:
        raise ValueError("Unexpected simulator camera shape or dtype")
    return Image.fromarray(raw[::-1, ::-1].copy())


def run_episode(env, initial_state, policy, instruction, config, task_id, initial_id, writer=None,
                *, observer=None):
    """Record only policy actions in the horizon; settling is a separate phase."""
    if observer is not None:
        if not config.get("record_confidence"):
            raise ValueError("Dashboard observation requires an existing confidence-enabled protocol")
        observer.begin(f'{config["suite"]}/task-{task_id:02d}-init-{initial_id:02d}',
                       suite=config["suite"], task_id=task_id, initial_id=initial_id,
                       instruction=instruction, max_steps=config["max_policy_steps"])
    env.seed(config["environment_seed"])
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(config["settle_steps"]):
        observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        if done or env.check_success():
            raise RuntimeError("Initial state terminated or succeeded during settling")
    actions, chunks, query_steps, query_seeds, image_hashes, latencies, compute = [], [], [], [], [], [], []
    states = [np.asarray(env.get_sim_state()).copy()]
    diagnostics = []
    success, terminated = False, False
    queued = []
    if writer is not None:
        writer.append_data(np.asarray(camera_image(observation)))
    for step in range(config["max_policy_steps"]):
        if not queued:
            image = camera_image(observation)
            seed = config["policy_seed"] + task_id * 100000 + initial_id * 1000 + len(chunks)
            start = time.perf_counter()
            if config.get("record_confidence"):
                chunk, seconds, stats = policy.predict_with_confidence(
                    image, instruction, config["denoising_steps"], seed,
                    verify_parity=bool(config.get("check_repeat_prediction") and not chunks))
                diagnostics.append(stats)
            else:
                chunk, seconds = policy.predict(image, instruction, config["denoising_steps"], seed)
            chunk = validate_actions(chunk)
            latencies.append(time.perf_counter() - start)
            compute.append(seconds)
            if config.get("check_repeat_prediction") and not chunks:
                repeated, _ = policy.predict(image, instruction, config["denoising_steps"], seed)
                np.testing.assert_array_equal(chunk, validate_actions(repeated))
            chunks.append(chunk.copy())
            query_steps.append(step)
            query_seeds.append(seed)
            image_hashes.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
            if observer is not None:
                observer.query(image, instruction, chunk, stats, seconds, query=len(chunks) - 1,
                               control_step=step, seed=seed, steps=config["denoising_steps"])
            queued = list(chunk[:config["execute_chunk_steps"]])
        action = queued.pop(0)
        observation, _, done, _ = env.step(action.tolist())
        actions.append(action.copy())
        states.append(np.asarray(env.get_sim_state()).copy())
        if writer is not None:
            writer.append_data(np.asarray(camera_image(observation)))
        # done may also signal a simulator time limit. Success is explicitly checked.
        success = bool(env.check_success())
        terminated = bool(done)
        if success or terminated:
            break
    trace = {"actions": np.asarray(actions), "states": np.asarray(states), "chunks": np.asarray(chunks),
             "query_steps": np.asarray(query_steps), "query_seeds": np.asarray(query_seeds, dtype=np.int64),
             "image_sha256": np.asarray(image_hashes), "query_seconds": np.asarray(latencies),
             "policy_seconds": np.asarray(compute), "initial_state": np.asarray(initial_state)}
    if diagnostics:
        trace["probabilities"] = np.asarray([d["probabilities"] for d in diagnostics], dtype=np.float32)
        trace["action_bins"] = np.asarray([d["action_bins"] for d in diagnostics], dtype=np.int16)
        for key in METRICS:
            trace[key] = np.asarray([d["metrics"][key] for d in diagnostics], dtype=np.float32)
        trace["chunk_confidence"] = np.asarray([d["chunk_confidence"] for d in diagnostics])
    result = {"task_id": task_id, "initial_state_id": initial_id, "instruction": instruction,
              "success": success, "policy_steps": len(actions), "settle_steps": config["settle_steps"],
              "policy_queries": len(chunks), "environment_terminated": terminated,
              "repeated_prediction_verified": bool(config.get("check_repeat_prediction")),
              "confidence_schema": SCHEMA if diagnostics else None,
              "confidence_action_parity_verified": bool(diagnostics and config.get("check_repeat_prediction")),
              "initial_state_sha256": hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest()}
    if observer is not None:
        observer.finish(result)
    return result, trace


def aggregate(results, task_ids):
    tasks = []
    for task_id in task_ids:
        selected = [r for r in results if r["task_id"] == task_id]
        count = len(selected)
        successes = sum(r["success"] for r in selected)
        tasks.append({"task_id": task_id, "episodes": count, "successes": successes,
                      "success_rate": successes / count if count else None})
    successes = sum(r["success"] for r in results)
    return {"episodes": len(results), "successes": successes,
            "success_rate": successes / len(results) if results else None, "tasks": tasks,
            "task_macro_success_rate": float(np.mean([r["success_rate"] for r in tasks]))
            if all(r["episodes"] for r in tasks) else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_libero_spatial.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dashboard-session", type=Path,
                        help="New optional telemetry directory, separate from evaluation evidence")
    parser.add_argument("--smoke", action="store_true", help="10 tasks, one initial state, four controls; no result claim")
    parser.add_argument("--preflight", action="store_true", help="One task, eight controls, real checkpoint; excluded")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config["suite"] == "libero_spatial" and config["camera_resolution"] == 128
    assert 1 <= config["execute_chunk_steps"] <= 8
    kind = "closed_loop_evaluation"
    if args.smoke:
        config = {**config, "initial_state_ids": [0], "max_policy_steps": 4, "save_video": False}
        kind = "simulator_smoke_not_policy_result"
    if args.preflight:
        assert not args.smoke
        config = {**config, "task_ids": [0], "initial_state_ids": [0], "max_policy_steps": 8,
                  "check_repeat_prediction": True}
        kind = "real_checkpoint_preflight_excluded_from_score"
    args.output.mkdir(parents=True, exist_ok=True)
    policy = PolicyClient(args.socket)
    if config.get("record_confidence"):
        assert policy.metadata["confidence_schema"] == SCHEMA
    if not args.smoke:
        assert policy.metadata["kind"] == "decisionvla_bf16_reference"
        assert policy.metadata["step"] == config["expected_checkpoint_step"]
    import imageio.v2 as imageio
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    repository = Path(get_libero_path("benchmark_root")).parents[1]
    # The pinned simulator image installs LIBERO as root; evaluation runs as the
    # host UID so the private socket and output files keep consistent ownership.
    revision = subprocess.check_output(["git", "-c", "safe.directory=" + str(repository),
                                        "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert revision == config["libero_revision"], "Simulator source revision differs from the plan"
    suite = benchmark.get_benchmark_dict()[config["suite"]](task_order_index=0)
    states, task_metadata = {}, []
    for task_id in config["task_ids"]:
        task = suite.get_task(task_id)
        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        # Pinned upstream LIBERO assets are trusted local inputs, not user-supplied pickles.
        states[task_id] = torch.load(path, map_location="cpu", weights_only=False)
        assert max(config["initial_state_ids"]) < len(states[task_id])
        bddl = Path(suite.get_task_bddl_file_path(task_id))
        task_metadata.append({"task_id": task_id, "name": task.name, "language": task.language,
                              "init_states_sha256": digest(path), "bddl_sha256": digest(bddl)})
    contract = {"kind": kind, "config": config, "policy": policy.metadata, "tasks": task_metadata,
                "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "runtime": {p: importlib.metadata.version(p) for p in
                            ["torch", "numpy", "mujoco", "robosuite", "Pillow", "libero"]}}
    contract_path = args.output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Evaluation contract changed; use a new output directory")
    else:
        write_json(contract_path, contract)
    results, all_latency, all_compute = [], [], []
    start = time.monotonic()
    expected = len(config["task_ids"]) * len(config["initial_state_ids"])
    from .dashboard.evaluation import EvaluationRecorder, session_metadata

    if args.dashboard_session and not config.get("record_confidence"):
        parser.error("--dashboard-session requires record_confidence in the existing protocol")
    metadata = session_metadata(policy.metadata, benchmark=config["suite"], selection=kind)
    with (EvaluationRecorder(args.dashboard_session, metadata, expected=expected)
          if args.dashboard_session else nullcontext()) as observer:
        try:
            for task_id in config["task_ids"]:
                task = suite.get_task(task_id)
                env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                                         camera_heights=128, camera_widths=128, camera_names=["agentview"],
                                         control_freq=20, render_gpu_device_id=-1)
                try:
                    for initial_id in config["initial_state_ids"]:
                        name = f"task-{task_id:02d}-init-{initial_id:02d}"
                        destination = args.output / name
                        if destination.exists():
                            result = json.loads((destination / "result.json").read_text())
                            assert result["task_id"] == task_id and result["initial_state_id"] == initial_id
                            for file, sha in result["artifacts_sha256"].items():
                                assert Path(file).name == file and digest(destination / file) == sha
                            trace = np.load(destination / "trace.npz", allow_pickle=False)
                            if observer is not None:
                                observer.restore(f'{config["suite"]}/{name}', result, trace, config)
                        else:
                            if time.monotonic() - start > config["max_evaluation_hours"] * 3600:
                                raise TimeoutError("Evaluation deadline reached; completed episodes preserved")
                            temporary = args.output / (name + ".partial")
                            temporary.mkdir(exist_ok=False)
                            writer = (imageio.get_writer(temporary / "rollout.mp4", fps=20, codec="libx264")
                                      if config["save_video"] else None)
                            try:
                                result, trace = run_episode(env, states[task_id][initial_id], policy, task.language,
                                                            config, task_id, initial_id, writer, observer=observer)
                            finally:
                                if writer is not None:
                                    writer.close()
                            np.savez_compressed(temporary / "trace.npz", **trace)
                            result["artifacts_sha256"] = {p.name: digest(p) for p in temporary.iterdir() if p.is_file()}
                            write_json(temporary / "result.json", result)
                            temporary.rename(destination)
                        results.append(result)
                        all_latency.extend(trace["query_seconds"].tolist())
                        all_compute.extend(trace["policy_seconds"].tolist())
                        if hasattr(trace, "close"):
                            trace.close()
                        status = {"status": "running", "kind": kind, "completed_episodes": len(results),
                                  "expected_episodes": expected, "elapsed_seconds": time.monotonic() - start}
                        write_json(args.output / "status.json", status)
                        print(json.dumps({"event": "episode", **result}), flush=True)
                finally:
                    env.close()
            summary = {"status": "completed", "kind": kind, **aggregate(results, config["task_ids"]),
                       "expected_episodes": expected, "errors": 0, "elapsed_seconds": time.monotonic() - start,
                       "query_seconds_p50_p95": np.percentile(all_latency, [50, 95]).tolist(),
                       "policy_seconds_p50_p95": np.percentile(all_compute, [50, 95]).tolist()}
            assert len(results) == expected
            write_json(args.output / "summary.json", summary)
            write_json(args.output / "status.json", {**status, "status": "completed"})
            print(json.dumps(summary), flush=True)
        except Exception as error:
            write_json(args.output / "status.json", {"status": "error", "error": repr(error), "kind": kind,
                                                    "completed_episodes": len(results), "expected_episodes": expected})
            raise


if __name__ == "__main__":
    main()
