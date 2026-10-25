"""Explicit paired development evaluation; never resumes or changes pilot runs.

Run in the existing simulator environment, talking to native_serve's separate
policy runtime. The frozen protocol covers 40 tasks x 5 initial states x 3 budgets.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import load_benchmark
from .confidence import SCHEMA
from .native_compare import paired_success, sha256
from .rollout import run_episode, write_json
from .rpc import PolicyClient


class RecordingClient:
    def __init__(self, policy):
        self.policy = policy
        self.readouts = []

    def predict_with_confidence(self, *args, **kwargs):
        result = self.policy.predict_with_confidence(*args, **kwargs)
        self.readouts.append(result[2])
        return result

    def predict(self, *args, **kwargs):
        return self.policy.predict(*args, **kwargs)


def run(args):
    from libero.libero.envs import OffScreenRenderEnv, bddl_utils

    plan = json.loads(args.plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    if protocol["passes"] != [1, 2, 4] or protocol["benchmark"] != "original":
        raise ValueError("Development comparison requires native 1/2/4 on original LIBERO")
    policy = PolicyClient(args.socket)
    if (policy.metadata.get("kind") != "decisionvla.native-decisions.v1"
            or policy.metadata.get("trained") is not True
            or policy.metadata.get("confidence_schema") != SCHEMA):
        raise ValueError("Expected a trained native checkpoint, not a legacy/base policy")
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"protocol": protocol, "protocol_sha256": sha256(args.protocol),
        "plan_sha256": sha256(args.plan), "policy": policy.metadata,
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "scope": "development comparison on reused original initial states; Plus excluded"}
    write_json(args.output / "contract.json", contract)
    rows = []
    for suite_index, suite_name in enumerate(plan["suites"]):
        suite, _ = load_benchmark(plan, "original", suite_name)
        for task_id in range(suite.n_tasks):
            path = suite.get_task_bddl_file_path(task_id)
            instruction = bddl_utils.get_problem_info(str(path))["language_instruction"]
            initial_states = suite.get_task_init_states(task_id)
            env = OffScreenRenderEnv(bddl_file_name=path, camera_heights=128, camera_widths=128,
                                    camera_names=["agentview"], control_freq=20, render_gpu_device_id=-1)
            try:
                for initial_id in protocol["initial_state_ids"]:
                    # Rotate pass order to reduce systematic warm/runtime ordering.
                    passes = [1, 2, 4]
                    offset = (suite_index * 10 + task_id + initial_id) % 3
                    for budget in passes[offset:] + passes[:offset]:
                        destination = args.output / f"pass-{budget}" / suite_name / f"task-{task_id:02d}-init-{initial_id:02d}"
                        destination.mkdir(parents=True, exist_ok=False)
                        cfg = {**plan["policy"], **protocol["policy"], "suite": suite_name,
                               "denoising_steps": budget, "record_confidence": True,
                               "check_repeat_prediction": True,
                               "max_policy_steps": plan["max_policy_steps"][suite_name]}
                        recorder = RecordingClient(policy)
                        result, trace = run_episode(env, initial_states[initial_id], recorder, instruction,
                                                    cfg, task_id, initial_id)
                        trace["allowed_mass"] = np.asarray([d["allowed_mass"] for d in recorder.readouts])
                        trace["allowed_log_mass"] = np.asarray([d["allowed_log_mass"] for d in recorder.readouts])
                        np.savez_compressed(destination / "trace.npz", **trace)
                        result.update(suite=suite_name, passes=budget, policy_seed=cfg["policy_seed"],
                                      checkpoint_sha256=policy.metadata["checkpoint_sha256"],
                                      bddl_sha256=sha256(path), task_name=suite.get_task(task_id).name,
                                      trace_sha256=sha256(destination / "trace.npz"))
                        write_json(destination / "result.json", result)
                        write_json(destination / "generation.json", recorder.readouts[0]["generation_config"])
                        rows.append(result)
                        write_json(args.output / "status.json", {"status": "running", "episodes": len(rows),
                                   "expected": len(plan["suites"]) * 10 * len(protocol["initial_state_ids"]) * 3})
            finally:
                env.close()
    write_json(args.output / "rows.json", rows)
    if contract["source_sha256"] != {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}:
        raise RuntimeError("Evaluation source changed during the run; preserve artifacts and audit")
    write_json(args.output / "report.json", paired_success(rows))
    write_json(args.output / "status.json", {"status": "completed", "episodes": len(rows)})
    write_json(args.output / "sha256.json", {str(p.relative_to(args.output)): sha256(p)
               for p in args.output.rglob("*") if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/eval_libero40_plus_100k.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/eval_native_passes.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
