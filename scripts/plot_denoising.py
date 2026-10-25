"""Plot measured per-pass confidence, error and input copying from a probe report."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    steps = sorted(map(int, report["steps"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout="constrained")
    fig.suptitle("Checkpoint 2,000 · Denoising becomes more certain without becoming more accurate",
                 fontsize=13, fontweight="bold")
    for ax, key, title, limits in zip(axes, ["confidence", "continuous_mae", "input_copy_fraction"],
                                    ["Raw action confidence", "Continuous-action MAE", "Input-token copy fraction"],
                                    [(0, 1.05), (0, 0.19), (0, 1.05)]):
        values = [report["steps"][str(s)][key] for s in steps]
        ax.plot(steps, values, "o-", color="#b96e32" if key == "continuous_mae" else "#315a9c", lw=2)
        for step, value in zip(steps, values):
            ax.annotate(f"{value:.3f}", (step, value), xytext=(0, 7), textcoords="offset points", ha="center")
        ax.set(title=title, xlabel="Decoder pass", xticks=steps, ylim=limits)
        ax.grid(axis="y", alpha=0.15)
    fig.supxlabel("20 held-out observations / 10 known tasks / 3 seeds · per-pass argmax diagnostics · no new training",
                  fontsize=9)
    fig.savefig(args.output.with_suffix(".png"), dpi=160, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
