"""Paired native 1/2/4-pass latency and recorded-observation action comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid latency samples")
    p50, p95 = np.quantile(values, [.5, .95])
    return {"count": len(values), "p50_seconds": float(p50), "p95_seconds": float(p95),
            "query_hz_at_p50": float(1 / p50) if p50 else None}


def paired_success(rows, *, bootstrap=20000, seed=9001):
    """Task-stratified paired resampling; never count missing/errors as failures."""
    conditions = {p: {} for p in (1, 2, 4)}
    for row in rows:
        if row.get("error") or type(row.get("success")) is not bool:
            raise ValueError("Incomplete/error episode is not a failure label")
        key = (row["suite"], row["task_id"], row["initial_state_id"])
        target = conditions[row["passes"]]
        if key in target:
            raise ValueError("Duplicate episode key")
        target[key] = row
    keys = sorted(conditions[4])
    if not keys or any(set(c) != set(keys) for c in conditions.values()):
        raise ValueError("All three pass budgets must cover identical episode keys")
    for key in keys:
        reference = conditions[4][key]
        for condition in conditions.values():
            if any(condition[key][field] != reference[field] for field in
                   ("initial_state_sha256", "policy_seed", "checkpoint_sha256")):
                raise ValueError("Unpaired initial state, seed or checkpoint")
    tasks = sorted({key[:2] for key in keys})
    groups = [np.array([i for i, k in enumerate(keys) if k[:2] == t]) for t in tasks]
    outcomes = {p: np.array([conditions[p][k]["success"] for k in keys], float) for p in conditions}
    rng = np.random.default_rng(seed)
    # Same resamples for both contrasts, preserving task balance and pairing.
    samples = [rng.choice(g, (bootstrap, len(g)), replace=True) for g in groups]
    report = {"episodes_per_pass": len(keys), "tasks": len(tasks), "bootstrap": bootstrap,
              "ci_scope": "paired within-task bootstrap, conditional on these tasks and policy seed",
              "reference_passes": 4, "noninferiority_margin": -0.05, "conditions": {}}
    for p, y in outcomes.items():
        difference = y - outcomes[4]
        deltas = np.mean([difference[s].mean(-1) for s in samples], axis=0)
        low, high = np.quantile(deltas, [.025, .975])
        report["conditions"][str(p)] = {"successes": int(y.sum()), "success_rate": float(y.mean()),
            "task_macro_success_rate": float(np.mean([y[g].mean() for g in groups])),
            "difference_vs_4": float(np.mean([difference[g].mean() for g in groups])),
            "paired_ci95": [float(low), float(high)],
            "ci_degenerate": bool(low == high),
            "passes_quality_screen": bool(low > -0.05 and low != high and outcomes[4].sum() > 0)}
    report["degenerate_ci_note"] = (
        "A degenerate empirical bootstrap interval is not evidence of equivalence; quality screen stays unresolved")
    report["selection"] = "Report alongside separately measured latency; no automatic policy switch"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True,
                        help="JSON list: image path, instruction, optional target_actions [8,7]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("At least one warmup and repeat are required")
    from PIL import Image

    from .native_policy import NativeDecisionVLA

    observations = json.loads(args.observations.read_text())
    if not observations:
        raise ValueError("No observations")
    args.output.mkdir(parents=True, exist_ok=False)
    policy = NativeDecisionVLA.load(args.adapter, device=args.device)
    rows, snapshots, inputs = [], {}, []
    for index, row in enumerate(observations):
        path = (args.observations.parent / row["image"]).resolve()
        image = Image.open(path).convert("RGB")
        if image.size != (128, 128):
            raise ValueError("Use the audited 128x128, already rotated policy input")
        inputs.append((image, row["instruction"]))
        row["image_sha256"] = sha256(path)
    contract = {"kind": "native_paired_observation_latency", "policy": policy.metadata,
        "observations_sha256": sha256(args.observations), "inputs": observations,
        "seed": args.seed, "passes": [1, 2, 4], "warmup_per_case_pass": args.warmup,
        "repeats": args.repeats, "telemetry": False, "simulation": False,
        "timing_scope": "preprocess, native encoder+denoising+terminal readout, CPU result; no network/robot",
        "quality_scope": "demonstration error if labeled; not closed-loop success",
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    for i, (image, instruction) in enumerate(inputs):
        for p in (1, 2, 4):
            for _ in range(args.warmup):
                policy.predict_with_confidence(image, instruction, p, args.seed + i)
        for repeat in range(args.repeats):
            order = [1, 2, 4]
            offset = (i + repeat) % 3
            for p in order[offset:] + order[:offset]:
                result = policy.predict_with_confidence(image, instruction, p, args.seed + i)
                record = {"case": i, "repeat": repeat, "passes": p, "seed": args.seed + i,
                          **result["timing"]}
                if repeat == 0:
                    stats = result["diagnostics"]
                    for name, value in {"actions": result["actions"], "probabilities": stats["probabilities"],
                                        "allowed_mass": stats["allowed_mass"]}.items():
                        snapshots[f"case_{i}_pass_{p}_{name}"] = np.asarray(value)
                    (args.output / f"case-{i}-pass-{p}-generation.json").write_text(
                        json.dumps(stats["generation_config"], indent=2) + "\n")
                rows.append(record)
                with (args.output / "latency.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
    report = {"conditions": {}, "action_comparisons": []}
    for p in (1, 2, 4):
        report["conditions"][str(p)] = {key: percentiles([r[key] for r in rows if r["passes"] == p])
            for key in ("preprocess_seconds", "prefix_seconds", "denoising_readout_seconds",
                        "native_generate_seconds", "readout_cpu_seconds", "total_seconds")}
        for i, row in enumerate(observations):
            actual = snapshots[f"case_{i}_pass_{p}_actions"]
            reference = snapshots[f"case_{i}_pass_4_actions"]
            comparison = {"case": i, "passes": p, "mae_vs_4": float(np.abs(actual - reference).mean()),
                          "changed_controls_vs_4": int(np.count_nonzero(actual != reference))}
            if "target_actions" in row:
                from .rpc import validate_actions
                target = validate_actions(row["target_actions"])
                comparison["demonstration_mae"] = float(np.abs(actual - target).mean())
            report["action_comparisons"].append(comparison)
    np.savez_compressed(args.output / "predictions.npz", **snapshots)
    report["artifacts_sha256"] = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
