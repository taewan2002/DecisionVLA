"""Audit complete rollout artifacts and measure predeclared episode-level association."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .confidence import METRICS, SCHEMA, validate_diagnostics
from .rollout import digest, write_json


def pair_counts(labels, scores):
    """Failure-positive concordance, half credit for ties; no single-class fabrication."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid binary endpoint/score vectors")
    positive, negative = scores[labels], scores[~labels]
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    return float(np.sum(left) + 0.5 * np.sum(right - left)), positive.size * negative.size


def auc(labels, scores):
    numerator, denominator = pair_counts(labels, scores)
    return numerator / denominator if denominator else None


def within_task_auc(labels, scores, tasks):
    counts = [pair_counts(labels[tasks == t], scores[tasks == t]) for t in np.unique(tasks)]
    pairs = sum(d for _, d in counts)
    return sum(n for n, _ in counts) / pairs if pairs else None


def association(rows, iterations=2000, seed=20261024):
    labels = np.array([r["failure"] for r in rows], dtype=bool)
    scores = np.array([r["first_query_uncertainty"] for r in rows])
    tasks = np.array([r["task_id"] for r in rows])
    groups = [np.flatnonzero(tasks == t) for t in np.unique(tasks)]
    primary = within_task_auc(labels, scores, tasks)
    pooled = auc(labels, scores)
    rng = np.random.default_rng(seed)
    boot, pooled_boot = [], []
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        value = within_task_auc(labels[indices], scores[indices], tasks[indices])
        if value is not None:
            boot.append(value)
        value = auc(labels[indices], scores[indices])
        if value is not None:
            pooled_boot.append(value)
    null = []
    if primary is not None:
        for _ in range(iterations):
            permuted = labels.copy()
            for group in groups:
                permuted[group] = rng.permutation(labels[group])
            null.append(within_task_auc(permuted, scores, tasks))
    task_results = []
    for group in groups:
        numerator, pairs = pair_counts(labels[group], scores[group])
        task_results.append({"task_id": int(tasks[group[0]]), "episodes": len(group),
                             "failures": int(labels[group].sum()), "eligible_pairs": pairs,
                             "auroc": numerator / pairs if pairs else None})
    # Fixed [0,1] bins, never optimized on failure labels. Empirical rates are
    # descriptive associations, not a calibration curve of success probabilities.
    bins = []
    confidence = 1 - scores
    for index in range(5):
        lo, hi = index / 5, (index + 1) / 5
        selected = (confidence >= lo) & ((confidence < hi) if index < 4 else (confidence <= hi))
        bins.append({"confidence_lower_inclusive": lo, "confidence_upper": hi,
                     "episodes": int(selected.sum()),
                     "failure_rate": float(labels[selected].mean()) if selected.any() else None})
    return {"episodes": len(rows), "failures": int(labels.sum()), "successes": int((~labels).sum()),
            "primary_within_task_auroc": primary,
            "primary_ci95": np.percentile(boot, [2.5, 97.5]).tolist() if boot else None,
            "primary_bootstrap_valid": len(boot),
            "primary_permutation_p_one_sided": (1 + sum(v >= primary for v in null)) / (1 + len(null))
            if null else None,
            "pooled_auroc": pooled,
            "pooled_ci95": np.percentile(pooled_boot, [2.5, 97.5]).tolist() if pooled_boot else None,
            "secondary_pooled_entropy_auroc": auc(labels, [r["first_query_entropy"] for r in rows]),
            "secondary_pooled_low_probability_auroc": auc(labels, [r["first_query_low_probability"] for r in rows]),
            "tasks": task_results, "confidence_bins_descriptive": bins,
            "interpretation": "undefined: no within-task outcome variation" if primary is None else
            "association only; raw confidence is not success probability, and control was unchanged"}


def audit_results(directory, protocol):
    directory = Path(directory)
    contract = json.loads((directory / "contract.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    config = contract["config"]
    assert contract["kind"] == "closed_loop_evaluation" and config["record_confidence"]
    assert contract["policy"]["step"] == protocol["checkpoint_step"]
    assert config["suite"] == protocol["suite"]
    assert config["max_policy_steps"] == protocol.get("max_policy_steps", 220)
    assert summary["status"] == "completed" and summary["episodes"] == protocol["episodes"]
    rows, hashes = [], {}
    for task in config["task_ids"]:
        for initial in config["initial_state_ids"]:
            episode = directory / f"task-{task:02d}-init-{initial:02d}"
            result = json.loads((episode / "result.json").read_text())
            assert (result["task_id"], result["initial_state_id"]) == (task, initial)
            assert result["confidence_schema"] == SCHEMA and type(result["success"]) is bool
            for name, expected in result["artifacts_sha256"].items():
                assert Path(name).name == name and digest(episode / name) == expected
                hashes[str((episode / name).relative_to(directory))] = expected
            hashes[str((episode / "result.json").relative_to(directory))] = digest(episode / "result.json")
            with np.load(episode / "trace.npz", allow_pickle=False) as trace:
                queries = result["policy_queries"]
                assert trace["probabilities"].shape == (queries, 8, 7, 256)
                assert trace["query_steps"].tolist() == list(range(0, result["policy_steps"], 8))
                np.testing.assert_array_equal(trace["actions"], trace["chunks"].reshape(-1, 7)[:result["policy_steps"]])
                assert len(trace["states"]) == result["policy_steps"] + 1
                for q in range(queries):
                    validate_diagnostics({"schema": SCHEMA, "calibrated": False,
                                          "conditioning": "final_denoising_step_given_sampled_canvas",
                                          "probabilities": trace["probabilities"][q],
                                          "action_bins": trace["action_bins"][q],
                                          "metrics": {k: trace[k][q] for k in METRICS},
                                          "chunk_confidence": trace["chunk_confidence"][q]}, trace["chunks"][q])
                rows.append({"task_id": task, "initial_state_id": initial, "failure": int(not result["success"]),
                             "policy_steps": result["policy_steps"], "queries": queries,
                             "first_query_uncertainty": float(1 - trace["chunk_confidence"][0]),
                             "first_query_entropy": float(trace["normalized_entropy"][0].mean()),
                             "first_query_low_probability": float(1 - trace["selected_probability"][0].mean())})
    assert len(rows) == protocol["episodes"]
    assert sum(not r["failure"] for r in rows) == summary["successes"]
    return rows, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/confidence_analysis.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    rows, hashes = audit_results(args.results, protocol)
    report = association(rows, seed=protocol["seed"])
    report.update(protocol=protocol, protocol_sha256=digest(args.protocol),
                  contract_sha256=digest(args.results / "contract.json"), status="completed")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report)
    write_json(args.output / "verified-artifacts.json", hashes)
    with (args.output / "episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
