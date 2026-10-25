"""Recompute denoising diagnostics from raw probabilities/canvases without a GPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from decisionvla.codec import ActionCodec
from decisionvla.confidence import summarize
from decisionvla.rollout import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results / "report.json").read_text())
    rows = [json.loads(line) for line in (args.results / "events.jsonl").read_text().splitlines()]
    plan = json.loads((args.results / "plan.json").read_text())
    codec, count = ActionCodec(), 0
    for observation, key in enumerate(plan["keys"]):
        filename = f"observation-{observation:02d}.npz"
        assert digest(args.results / filename) == report["trace_sha256"][filename]
        selected = [r for r in rows if r["observation"] == observation]
        assert len(selected) == 14 and all(r["key"] == key for r in selected)
        with np.load(args.results / filename, allow_pickle=False) as trace:
            target = trace["target_tokens"].reshape(8, 7)
            valid = trace["valid"].reshape(8, 7)
            for row in selected:
                prefix = f"seed_{row['seed']}_pass_{row['pass']}" if row["kind"] == "sample" else row["kind"]
                p = trace[prefix + "_probabilities"].reshape(8, 7, 256)
                canvas = trace[prefix + "_canvas"].reshape(8, 7)
                stats = summarize(p)
                ids = p.argmax(-1)
                np.testing.assert_array_equal(ids, row["prediction_bins"])
                actions = codec.decode(ids)
                wrong = valid & (ids != target)
                values = {"confidence": stats["chunk_confidence"],
                          "continuous_mae": np.abs(actions[:, :6] - trace["target_actions"][:, :6])[valid[:, :6]].mean(),
                          "translation_mae": np.abs(actions[:, :3] - trace["target_actions"][:, :3])[valid[:, :3]].mean(),
                          "token_accuracy": (ids == target)[valid].mean(),
                          "input_copy_fraction": (ids == canvas)[valid].mean(),
                          "gripper_accuracy": (ids[:, 6] == target[:, 6])[valid[:, 6]].mean(),
                          "entropy": np.mean(stats["metrics"]["normalized_entropy"]),
                          "peak_probability": np.mean(stats["metrics"]["selected_probability"])}
                if wrong.any():
                    values["wrong_token_confidence"] = np.asarray(stats["metrics"]["ordered_confidence"])[wrong].mean()
                else:
                    assert row["wrong_token_confidence"] is None
                for name, value in values.items():
                    np.testing.assert_allclose(value, row[name], atol=1e-7, rtol=0)
                count += 1
    paired = []
    for observation in range(len(plan["keys"])):
        a = [r for r in rows if r["kind"] == "sample" and r["observation"] == observation and r["pass"] == 1]
        b = [r for r in rows if r["kind"] == "sample" and r["observation"] == observation and r["pass"] == 4]
        paired.append({"observation": observation, "key": plan["keys"][observation],
                       **{key: float(np.mean([r[key] for r in b]) - np.mean([r[key] for r in a]))
                          for key in ["confidence", "continuous_mae", "translation_mae", "token_accuracy"]}})
    audit = {"status": "passed", "observations": len(plan["keys"]), "verified_distributions": count,
             "report_sha256": digest(args.results / "report.json"),
             "events_sha256": digest(args.results / "events.jsonl"), "plan_sha256": digest(args.results / "plan.json"),
             "confidence_increased_observations": sum(r["confidence"] > 0 for r in paired),
             "mae_increased_observations": sum(r["continuous_mae"] > 0 for r in paired),
             "paired_final_minus_first": paired,
             "scope": "post-hoc known-task demonstration diagnostic, not closed-loop success or calibration"}
    write_json(args.results / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "paired_final_minus_first"}))


if __name__ == "__main__":
    main()
