"""Assist a serial evaluator with complete, ordered tasks and one shared policy server.

Never split an environment's initialization sequence. Partial tasks remain private;
only a complete audited task can be published, well ahead of the serial evaluator.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from decisionvla.confidence import METRICS, SCHEMA, validate_diagnostics
from decisionvla.rollout import digest, run_episode, write_json
from decisionvla.rpc import PolicyClient


def publication_safe(config, leader_status, task_id, margin=5):
    """The serial evaluator must still have at least five earlier episodes to finish."""
    if leader_status.get("status") != "running":
        return False
    first_index = config["task_ids"].index(task_id) * len(config["initial_state_ids"])
    return leader_status["completed_episodes"] <= first_index - margin


def audit_episode(folder):
    result = json.loads((folder / "result.json").read_text())
    for name, sha in result["artifacts_sha256"].items():
        assert Path(name).name == name and digest(folder / name) == sha
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        assert trace["probabilities"].shape == (result["policy_queries"], 8, 7, 256)
        assert trace["query_steps"].tolist() == list(range(0, result["policy_steps"], 8))
        np.testing.assert_array_equal(trace["actions"], trace["chunks"].reshape(-1, 7)[:result["policy_steps"]])
        assert len(trace["states"]) == result["policy_steps"] + 1
        for q in range(result["policy_queries"]):
            validate_diagnostics({"schema": SCHEMA, "calibrated": False,
                                  "conditioning": "final_denoising_step_given_sampled_canvas",
                                  "probabilities": trace["probabilities"][q],
                                  "action_bins": trace["action_bins"][q],
                                  "metrics": {k: trace[k][q] for k in METRICS},
                                  "chunk_confidence": trace["chunk_confidence"][q]}, trace["chunks"][q])
    return {p.name: digest(p) for p in folder.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--parity-only", action="store_true", help="Replay first two episodes; do not publish")
    parser.add_argument("--gate", type=Path, help="Passed real-policy parity report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.results / "contract.json").read_text())
    contract_sha = digest(args.results / "contract.json")
    config = contract["config"]
    assert config["initial_state_ids"] == list(range(50))
    assert config["record_confidence"] and config["save_video"]
    import decisionvla.rollout as rollout_module

    source = Path(rollout_module.__file__).parent
    assert {p.name: digest(p) for p in sorted(source.glob("*.py"))} == contract["code_sha256"]
    policy = PolicyClient(args.socket)
    assert policy.metadata == contract["policy"]
    if not args.parity_only:
        gate = json.loads(args.gate.read_text())
        assert gate["status"] == "parity_passed" and gate["contract_sha256"] == contract_sha
        assert publication_safe(config, json.loads((args.results / "status.json").read_text()), args.task_id)
        assert not list(args.results.glob(f"task-{args.task_id:02d}-init-*")), "Task has already started"
    import imageio.v2 as imageio
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[config["suite"]](task_order_index=0)
    task = suite.get_task(args.task_id)
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    metadata = next(t for t in contract["tasks"] if t["task_id"] == args.task_id)
    assert digest(states_path) == metadata["init_states_sha256"]
    assert digest(Path(suite.get_task_bddl_file_path(args.task_id))) == metadata["bddl_sha256"]
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
                            camera_heights=128, camera_widths=128, camera_names=["agentview"],
                            control_freq=20, render_gpu_device_id=-1)
    start = time.time()
    state = {"status": "running", "task_id": args.task_id, "contract_sha256": contract_sha,
             "started_unix": start, "completed_episodes": 0, "policy_server_shared": True,
             "worker_sha256": digest(Path(__file__)), "parity_only": args.parity_only}
    write_json(args.output / "status.json", state)
    artifacts = {}
    try:
        for initial in config["initial_state_ids"][:2] if args.parity_only else config["initial_state_ids"]:
            name = f"task-{args.task_id:02d}-init-{initial:02d}"
            folder = args.output / name
            folder.mkdir()
            writer = imageio.get_writer(folder / "rollout.mp4", fps=20, codec="libx264")
            try:
                result, trace = run_episode(env, states[initial], policy, task.language, config,
                                            args.task_id, initial, writer)
            finally:
                writer.close()
            np.savez_compressed(folder / "trace.npz", **trace)
            result["artifacts_sha256"] = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
            write_json(folder / "result.json", result)
            artifacts[name] = audit_episode(folder)
            if args.parity_only:
                original = json.loads((args.results / name / "result.json").read_text())
                assert {k: v for k, v in result.items() if k != "artifacts_sha256"} == {
                    k: v for k, v in original.items() if k != "artifacts_sha256"}
                with np.load(args.results / name / "trace.npz", allow_pickle=False) as expected:
                    for key in expected.files:
                        if key not in ("query_seconds", "policy_seconds"):
                            np.testing.assert_array_equal(trace[key], expected[key], err_msg=key)
                state["all_observations_states_actions_probabilities_exact"] = True
            state.update(completed_episodes=initial + 1, elapsed_seconds=time.time() - start)
            write_json(args.output / "status.json", state)
        assert digest(args.results / "contract.json") == contract_sha
        if args.parity_only:
            state["status"] = "parity_passed"
        else:
            assert len(artifacts) == 50
            write_json(args.output / "audited-artifacts.json", artifacts)
            leader = json.loads((args.results / "status.json").read_text())
            if not publication_safe(config, leader, args.task_id):
                state.update(status="completed_unpublished", reason="serial evaluator too close")
            else:
                assert not list(args.results.glob(f"task-{args.task_id:02d}-init-*"))
                write_json(args.output / "publication.json", {"status": "publishing", "leader": leader,
                                                              "episodes": list(artifacts)})
                published = []
                try:
                    for name in artifacts:
                        # Both directories are on the same filesystem. No overwrite is permitted.
                        assert not (args.results / name).exists()
                        os.rename(args.output / name, args.results / name)
                        published.append(name)
                except BaseException:
                    for name in reversed(published):
                        os.rename(args.results / name, args.output / name)
                    raise
                write_json(args.output / "publication.json", {"status": "published", "leader": leader,
                                                              "episodes": published})
                state["status"] = "published"
    except BaseException as error:
        state.update(status="error", error=repr(error))
        raise
    finally:
        env.close()
        state["elapsed_seconds"] = time.time() - start
        write_json(args.output / "status.json", state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
