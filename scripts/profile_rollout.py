"""Profile CPU rollout replay without issuing policy queries or changing evaluation artifacts."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from decisionvla.confidence import METRICS
from decisionvla.rollout import run_episode, write_json


class Timed:
    def __init__(self, target):
        self.target = target
        self.seconds = {}
        self.calls = {}

    def __getattr__(self, name):
        value = getattr(self.target, name)
        if not callable(value):
            return value

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return value(*args, **kwargs)
            finally:
                self.seconds[name] = self.seconds.get(name, 0) + time.perf_counter() - start
                self.calls[name] = self.calls.get(name, 0) + 1
        return measured


class ReplayPolicy:
    def __init__(self, trace):
        self.trace = trace
        self.index = 0
        self.image_hashes = []

    def predict_with_confidence(self, image, instruction, steps, seed, verify_parity=False):
        i = self.index
        assert not verify_parity and steps == 4
        assert seed == self.trace["query_seeds"][i]
        actual = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        self.image_hashes.append(actual)
        self.index += 1
        stats = {"probabilities": self.trace["probabilities"][i],
                 "action_bins": self.trace["action_bins"][i],
                 "metrics": {k: self.trace[k][i] for k in METRICS},
                 "chunk_confidence": self.trace["chunk_confidence"][i]}
        return self.trace["chunks"][i], 0.0, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--warmup-episode", help="Replay a saved episode first to test environment reset history")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.results / "contract.json").read_text())
    config = contract["config"]
    episode = args.results / args.episode
    expected = json.loads((episode / "result.json").read_text())
    with np.load(episode / "trace.npz", allow_pickle=False) as archive:
        original = {k: archive[k] for k in archive.files}
    import imageio.v2 as imageio
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[config["suite"]](task_order_index=0)
    # Match the evaluator's Torch initialization before constructing MuJoCo.
    torch.empty(0)
    start = time.perf_counter()
    env = Timed(OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(expected["task_id"]),
                                  camera_heights=128, camera_widths=128, camera_names=["agentview"],
                                  control_freq=20, render_gpu_device_id=-1))
    report = {"kind": "CPU replay, no policy inference, excluded from evaluation",
              "episode": args.episode, "environment": {k: os.getenv(k) for k in
                  ("LP_NUM_THREADS", "OMP_NUM_THREADS", "MUJOCO_GL", "PYOPENGL_PLATFORM")},
              "environment_construction_seconds": time.perf_counter() - start}
    if args.warmup_episode:
        warm_folder = args.results / args.warmup_episode
        warm_result = json.loads((warm_folder / "result.json").read_text())
        assert warm_result["task_id"] == expected["task_id"]
        with np.load(warm_folder / "trace.npz", allow_pickle=False) as archive:
            warm_trace = {k: archive[k] for k in archive.files}
        _, warm_actual = run_episode(env, warm_trace["initial_state"], ReplayPolicy(warm_trace),
                                     warm_result["instruction"], config, warm_result["task_id"],
                                     warm_result["initial_state_id"])
        report["warmup"] = {"episode": args.warmup_episode,
                            "state_difference": float(np.max(np.abs(warm_actual["states"] - warm_trace["states"]))),
                            "images_exact": bool(np.array_equal(warm_actual["image_sha256"],
                                                                 warm_trace["image_sha256"]))}
    writer = Timed(imageio.get_writer(args.output / "replay.mp4", fps=20, codec="libx264"))
    policy = ReplayPolicy(original)
    try:
        start = time.perf_counter()
        result, trace = run_episode(env, original["initial_state"], policy, expected["instruction"], config,
                                    expected["task_id"], expected["initial_state_id"], writer)
        writer.close()
        report["rollout_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        np.savez_compressed(args.output / "replay.npz", **trace)
        report.update(compression_seconds=time.perf_counter() - start,
                      maximum_state_difference=float(np.max(np.abs(trace["states"] - original["states"]))),
                      initial_settled_state_difference=float(np.max(np.abs(trace["states"][0] - original["states"][0]))),
                      observation_hashes_exact=bool(np.array_equal(trace["image_sha256"], original["image_sha256"])),
                      observation_hash_matches=int(np.sum(trace["image_sha256"] == original["image_sha256"])),
                      first_observation_hash=policy.image_hashes[0])
        for key in ("actions", "chunks", "query_steps", "query_seeds", "image_sha256", "probabilities"):
            np.testing.assert_array_equal(trace[key], original[key], err_msg=key)
        np.testing.assert_allclose(trace["states"], original["states"], rtol=0, atol=1e-12)
        assert result["success"] == expected["success"]
        assert result["policy_steps"] == expected["policy_steps"]
        assert policy.index == expected["policy_queries"]
        report.update(status="passed", queries=policy.index, policy_steps=result["policy_steps"],
                      observation_hashes_exact=True, actions_and_probabilities_exact=True)
    except BaseException as error:
        report.update(status="error", error=repr(error), verified_queries=policy.index)
        raise
    finally:
        writer.close()
        env.close()
        report.update(environment_seconds=env.seconds, environment_calls=env.calls, writer_seconds=writer.seconds)
        write_json(args.output / "profile.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
