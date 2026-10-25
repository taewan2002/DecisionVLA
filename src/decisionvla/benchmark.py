"""Pinned four-suite LIBERO / LIBERO Plus workers, with per-task atomic evidence.

A worker owns whole tasks. Incomplete original tasks are never resumed by
skipping initial states; retain the failed directory and use a new attempt.
Plus uses the upstream environment wrapper, including its sensor corruption.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from .analyze_confidence import association, audit_results
from .confidence import SCHEMA
from .rollout import aggregate, digest, run_episode, write_json
from .rpc import PolicyClient


def base_task_index(name, originals):
    matches = [(len(t), i) for i, t in enumerate(originals) if name == t or name.startswith(t + "_")]
    if not matches:
        raise ValueError(f"Unmapped original task: {name}")
    return max(matches)[1]


def physical_bddl(path):
    # Plus encodes camera/robot/noise perturbations in the virtual filename.
    # Pass that virtual path to the environment; hash its underlying BDDL.
    return Path(str(path).split("_view_")[0] + ".bddl") if "_view_" in str(path) else Path(path)


def load_benchmark(plan, kind, suite_name):
    from libero.libero import benchmark, get_libero_path
    repository = Path(get_libero_path("benchmark_root")).parents[1]
    revision = subprocess.check_output(["git", "-c", "safe.directory=" + str(repository),
        "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert revision == plan[kind]["revision"], "Unexpected simulator revision"
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    expected = plan[kind]["task_counts"][suite_name]
    assert suite.n_tasks == expected
    if kind == "original":
        assert suite.get_task_names() == plan["original_tasks"][suite_name]
        categories = {}
    else:
        path = Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"
        assert digest(path) == plan["plus"]["classification_sha256"]
        rows = json.loads(path.read_text())[suite_name]
        assert len(rows) == expected
        categories = {}
        for i, row in enumerate(rows):
            assert row["id"] == i + 1 and row["name"] == suite.get_task(i).name
            categories[i] = row
    return suite, categories


def task_config(plan, kind, suite, task_id, step, preflight):
    return {**plan["policy"], "suite": suite, "task_ids": [task_id],
        "initial_state_ids": [0] if kind == "plus" or preflight else list(range(50)),
        "max_policy_steps": 8 if preflight else plan["max_policy_steps"][suite],
        "expected_checkpoint_step": step, "check_repeat_prediction": preflight,
        "record_confidence": True}


def run_worker(args, plan):
    import imageio.v2 as imageio
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs import bddl_utils
    policy = PolicyClient(args.socket)
    assert policy.metadata["step"] == args.step
    assert policy.metadata["kind"] == "decisionvla_bf16_reference"
    assert policy.metadata["confidence_schema"] == SCHEMA
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = {p: importlib.metadata.version(p) for p in ["torch", "numpy", "mujoco", "robosuite", "Pillow", "libero"]}
    all_rows, completed = [], 0
    expected = sum(plan[args.kind]["task_counts"].values())
    selected_global = 0
    for suite_index, suite_name in enumerate(plan["suites"]):
        suite, categories = load_benchmark(plan, args.kind, suite_name)
        chosen = list(range(suite.n_tasks))
        if args.preflight:
            if args.kind == "plus":
                chosen = [next(i for i, row in categories.items() if row["category"] == category)
                          for category in sorted({row["category"] for row in categories.values()})]
            else:
                chosen = [0]
        for task_id in chosen:
            owner = selected_global % args.shards
            selected_global += 1
            if owner != args.shard:
                continue
            task = suite.get_task(task_id)
            cfg = task_config(plan, args.kind, suite_name, task_id, args.step, args.preflight)
            destination = args.output / f"{suite_name}-task-{task_id:04d}"
            if destination.exists():
                raise FileExistsError(f"Whole-task output already exists: {destination}")
            destination.mkdir()
            random.seed(cfg["environment_seed"])
            np.random.seed(cfg["environment_seed"])
            virtual_path = suite.get_task_bddl_file_path(task_id)
            bddl = physical_bddl(virtual_path)
            # The BDDL is the semantic instruction; virtual perturbation suffixes
            # are metadata, never appended to the command shown to the policy.
            instruction = bddl_utils.get_problem_info(str(bddl))["language_instruction"]
            initial_states = suite.get_task_init_states(task_id)
            initial = np.asarray(initial_states)
            assert np.isfinite(initial).all() and len(initial) > max(cfg["initial_state_ids"])
            base = base_task_index(task.name, plan["original_tasks"][suite_name])
            contract = {"kind": "closed_loop_evaluation" if not args.preflight else "real_checkpoint_preflight_excluded_from_score",
                "config": cfg, "policy": policy.metadata, "runtime": runtime,
                "benchmark": args.kind, "plan_sha256": digest(args.plan),
                "task": {"name": task.name, "base_task_id": suite_index * 10 + base,
                    "instruction": instruction, "upstream_task_language": task.language,
                    "bddl_sha256": digest(bddl),
                    "initial_states_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                    "perturbation": categories.get(task_id)},
                "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
            write_json(destination / "contract.json", contract)
            env = OffScreenRenderEnv(bddl_file_name=virtual_path, camera_heights=128,
                camera_widths=128, camera_names=["agentview"], control_freq=20, render_gpu_device_id=-1)
            results = []
            try:
                for initial_id in cfg["initial_state_ids"]:
                    episode = destination / f"task-{task_id:02d}-init-{initial_id:02d}"
                    temporary = episode.with_name(episode.name + ".partial")
                    temporary.mkdir()
                    writer = imageio.get_writer(temporary / "rollout.mp4", fps=20, codec="libx264") if (
                        args.kind == "original" and plan["save_original_videos"] and not args.preflight) else None
                    try:
                        result, trace = run_episode(env, initial_states[initial_id], policy,
                            instruction, cfg, task_id, initial_id, writer)
                    finally:
                        if writer is not None:
                            writer.close()
                    np.savez_compressed(temporary / "trace.npz", **trace)
                    result["artifacts_sha256"] = {p.name: digest(p) for p in temporary.iterdir() if p.is_file()}
                    write_json(temporary / "result.json", result)
                    temporary.replace(episode)
                    results.append(result)
                    completed += 1
                    write_json(args.output / "status.json", {"status": "running", "episodes": completed,
                        "suite": suite_name, "task": task_id, "initial": initial_id})
            finally:
                env.close()
            summary = {"status": "completed", **aggregate(results, [task_id])}
            write_json(destination / "summary.json", summary)
            if not args.preflight:
                rows, hashes = audit_results(destination, {"checkpoint_step": args.step,
                    "suite": suite_name, "episodes": len(results), "max_policy_steps": cfg["max_policy_steps"]})
                for row in rows:
                    row.update(variant_task_id=row["task_id"], task_id=suite_index * 10 + base,
                        suite=suite_name, category=categories.get(task_id, {}).get("category", "original"))
                all_rows.extend(rows)
                write_json(destination / "audit.json", {"status": "passed", "rows": rows, "sha256": hashes})
            print(json.dumps({"event": "task_completed", "suite": suite_name, "task": task_id,
                "episodes": len(results), "successes": summary["successes"]}), flush=True)
    write_json(args.output / "worker-complete.json", {"status": "passed", "rows": all_rows,
        "episodes": completed, "shard": args.shard, "shards": args.shards, "preflight": args.preflight,
        "total_benchmark_tasks": expected, "plan_sha256": digest(args.plan), "checkpoint_step": args.step})


def collect(args, plan):
    rows, manifests = [], {}
    for shard in range(args.shards):
        root = args.output / f"shard-{shard}"
        path = root / "worker-complete.json"
        report = json.loads(path.read_text())
        assert report["status"] == "passed" and not report["preflight"]
        assert report["shard"] == shard and report["shards"] == args.shards
        assert report["checkpoint_step"] == args.step and report["plan_sha256"] == digest(args.plan)
        recomputed = []
        for taskdir in sorted(p for p in root.iterdir() if p.is_dir()):
            contract = json.loads((taskdir / "contract.json").read_text())
            audit = json.loads((taskdir / "audit.json").read_text())
            for name, sha in audit["sha256"].items():
                assert digest(taskdir / name) == sha
            assert contract["benchmark"] == args.kind and contract["plan_sha256"] == digest(args.plan)
            cfg = contract["config"]
            checked, hashes = audit_results(taskdir, {"checkpoint_step": args.step,
                "suite": cfg["suite"], "episodes": len(cfg["initial_state_ids"]),
                "max_policy_steps": plan["max_policy_steps"][cfg["suite"]]})
            assert hashes == audit["sha256"]
            base = base_task_index(contract["task"]["name"], plan["original_tasks"][cfg["suite"]])
            base += plan["suites"].index(cfg["suite"]) * 10
            assert base == contract["task"]["base_task_id"]
            perturbation = contract["task"]["perturbation"]
            for row in checked:
                row.update(variant_task_id=row["task_id"], task_id=base, suite=cfg["suite"],
                    category=perturbation["category"] if perturbation else "original")
            assert checked == audit["rows"]
            recomputed.extend(checked)
            manifests[str(taskdir.relative_to(args.output))] = digest(taskdir / "audit.json")
        def key(r):
            return r["suite"], r["variant_task_id"], r["initial_state_id"]
        assert sorted(recomputed, key=key) == sorted(report["rows"], key=key)
        rows.extend(recomputed)
    expected = 2000 if args.kind == "original" else 10030
    assert len(rows) == expected
    keys = [(r["suite"], r["variant_task_id"], r["initial_state_id"]) for r in rows]
    assert len(set(keys)) == expected
    for suite in plan["suites"]:
        expected_keys = {(suite, t, i) for t in range(plan[args.kind]["task_counts"][suite])
                         for i in ([0] if args.kind == "plus" else range(50))}
        assert {k for k in keys if k[0] == suite} == expected_keys
    result = association(rows, iterations=2000, seed=plan["analysis_seed"])
    result.update(status="completed", benchmark=args.kind, checkpoint_step=args.step,
        episode_weighted_success_rate=result["successes"] / expected,
        original_task_macro_success_rate=float(np.mean([
            np.mean([1-r["failure"] for r in rows if r["task_id"] == task]) for task in range(40)])),
        plan_sha256=digest(args.plan), association_group="original task (40 groups)",
        per_variant_auroc="undefined: one episode per variant" if args.kind == "plus" else None,
        suite_success={s: float(np.mean([1-r["failure"] for r in rows if r["suite"] == s])) for s in plan["suites"]},
        category_success={s: float(np.mean([1-r["failure"] for r in rows if r["category"] == s]))
                          for s in sorted({r["category"] for r in rows})})
    write_json(args.output / "report.json", result)
    write_json(args.output / "verified-task-audits.json", manifests)
    with (args.output / "episodes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--kind", choices=["original", "plus"], required=True)
    p.add_argument("--socket", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, default=100000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    assert 0 <= args.shard < args.shards
    plan = json.loads(args.plan.read_text())
    if args.collect:
        collect(args, plan)
    else:
        run_worker(args, plan)


if __name__ == "__main__":
    main()
