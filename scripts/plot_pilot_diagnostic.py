"""Render the fixed 2,000-update training and task-0 post-hoc diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root
    output = root / "diagnostic-task0"
    training_file = root / "training-diagnostic.json"
    geometry_file = output / "geometry.json"
    training = json.loads(training_file.read_text())
    geometry = json.loads(geometry_file.read_text())
    rows = geometry["rows"]
    assert geometry["task_id"] == 0 and len(rows) == 50
    assert sorted(r["initial_id"] for r in rows) == list(range(50))
    assert sum(geometry["categories"].values()) == len(rows)
    failure = [r for r in rows if not r["success"]]
    success = [r for r in rows if r["success"]]
    assert failure and success, "This pilot plot requires both outcome classes"
    f = np.asarray([r["first_chunk_confidence"] for r in failure])
    s = np.asarray([r["first_chunk_confidence"] for r in success])
    # Lower confidence predicts failure. Count ties as one half.
    auc = float(np.mean((f[:, None] < s[None, :]) + 0.5 * (f[:, None] == s[None, :])))
    axes_confidence = np.asarray([r["first_confidence_by_axis"] for r in failure])
    np.testing.assert_allclose(axes_confidence.mean(1), f, atol=1e-7, rtol=0)
    with np.load(output / "geometry-traces.npz", allow_pickle=False) as traces:
        states = 0
        for row in rows:
            prefix = f"init_{row['initial_id']:02d}_"
            assert len(traces[prefix + "height_m"]) == row["policy_steps"] + 1
            assert len(traces[prefix + "confidence"]) == len(traces[prefix + "query_steps"])
            states += len(traces[prefix + "height_m"])
    summary = {
        "scope": "post-hoc first task only; not the final 500-episode analysis",
        "episodes": len(rows), "successes": len(success), "states_restored": states,
        "first_query_failure_auroc": auc,
        "first_confidence_failure_mean": float(f.mean()),
        "first_confidence_success_mean": float(s.mean()),
        "failure_translation_confidence_mean": float(axes_confidence[:, :3].mean()),
        "failure_rotation_confidence_mean": float(axes_confidence[:, 3:6].mean()),
        "failure_gripper_confidence_mean": float(axes_confidence[:, 6].mean()),
        "unique_training_anchor_fraction": training["unique_anchor_count"] / training["train_anchors_available"],
        "supervised_action_step_coverage": training["unique_supervised_action_steps"] / training["train_anchors_available"],
        "continuous_mae_reduction_vs_zero": 1 - training["fixed_validation_last"]["continuous_action_mae"]
        / training["zero_continuous_action_validation_mae"],
        "new_policy_calls": 0, "new_simulation_steps": 0,
        "limitations": ["Only one success: discrimination estimate is unstable",
                        "Geometry categories are post-hoc contact/lift proxies",
                        "Training validation reuses only 20 held-out observations",
                        "Confidence is not a calibrated probability of task success"],
        "input_sha256": {"training-diagnostic.json": sha256(training_file),
                         "geometry.json": sha256(geometry_file),
                         "geometry-traces.npz": sha256(output / "geometry-traces.npz")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), layout="constrained")
    fig.suptitle("DecisionVLA · 2,000-update pilot diagnostic", fontsize=19, fontweight="bold")
    ax = axes[0, 0]
    curve = [r for r in training["validation_curve"] if r.get("step", 0) >= 100]
    ax.plot([r["step"] for r in curve], [r["continuous_action_mae"] for r in curve],
            "o-", color="#315a9c", markersize=4, label="Policy validation MAE")
    ax.axhline(training["zero_continuous_action_validation_mae"], color="#bd682a", ls="--",
               label="Zero continuous-control baseline")
    ax.set(title="A. Some learning, substantial residual error", xlabel="Training updates (from 100)",
           ylabel="Normalized continuous-action MAE", ylim=(0.14, 0.32))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.98, 0.46, "20 fixed held-out observations\n0.164 vs 0.186 baseline",
            transform=ax.transAxes, ha="right", fontsize=9)
    ax = axes[0, 1]
    ax.scatter([r["initial_id"] for r in failure], f, color="#c37a38", s=26, label="Failure (49)")
    ax.scatter([r["initial_id"] for r in success], s, marker="*", color="#237c76", s=180,
               edgecolor="black", linewidth=0.5, label="Success (1)", zorder=4)
    ax.set(title="B. High first-query confidence also on failures", xlabel="Task 0 initial-state index",
           ylabel="Raw chunk confidence", ylim=(0.88, 1.005))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.03, 0.08, f"Failure AUROC = {auc:.3f}\nOnly one success; unstable estimate",
            transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    categories = ["no_sustained_bilateral_contact", "contact_without_3cm_lift", "lift_without_task_success", "success"]
    labels = ["No sustained two-finger contact", "Contact, no sustained >3 cm lift",
              "Contact + lift, task unsuccessful", "Task success"]
    counts = [geometry["categories"].get(k, 0) for k in categories]
    bars = ax.barh(labels, counts, color=["#d39460", "#b97b43", "#965b31", "#237c76"], height=0.6)
    ax.bar_label(bars, padding=4)
    ax.invert_yaxis()
    ax.set(title="C. Most failures lack sustained target contact", xlabel="Episodes (task 0 only)", xlim=(0, 43))
    ax.tick_params(axis="y", labelsize=8)
    ax.text(0.98, 0.02, "Post-hoc geometric proxies; not causal attribution",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555555")
    ax = axes[1, 1]
    means = axes_confidence.mean(0)
    ax.bar(["x", "y", "z", "rx", "ry", "rz", "gripper"], means,
           color=["#527cb5"] * 3 + ["#8999ae"] * 3 + ["#237c76"])
    for i, value in enumerate(means):
        ax.text(i, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    ax.set(title="D. Certain rotations / gripper raise the average", ylabel="Mean first-query confidence (49 failures)",
           ylim=(0, 1.12), yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.text(0.5, 0.04, "Distribution concentration, not probability of completing the task",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    fig.savefig(output / "pilot-diagnostic.png", dpi=160)
    fig.savefig(output / "pilot-diagnostic.svg")
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
