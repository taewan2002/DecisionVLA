"""Independent CPU recomputation of saved probabilities, actions and timing statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path("/output")
report = json.loads((root / "report.json").read_text())
assert report["status"] == "completed"
ref = np.load(root / "bf16_stream.npz", allow_pickle=False)
quant = np.load(root / "resident_w4.npz", allow_pickle=False)
results = {}
for phase, data, n in [("bf16_stream", ref, 10), ("resident_w4", quant, 20)]:
    raw = json.loads((root / f"{phase}.json").read_text())
    values = np.asarray([row["seconds"] for row in raw["samples"]])
    assert len(values) == n and (values > 0).all()
    assert all(not row["warmup"] for row in raw["samples"])
    calc = {
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "mean_seconds": float(values.mean()),
    }
    for k, v in calc.items():
        assert np.isclose(raw["summary"][k], v, rtol=0, atol=1e-12)
    for i in range(n):
        p = data[f"probabilities_{i}"]
        a = data[f"actions_{i}"]
        assert p.shape == (8, 7, 256) and a.shape == (8, 7)
        assert np.isfinite(p).all() and (p >= 0).all() and np.isfinite(a).all()
        assert np.allclose(p.sum(-1), 1, rtol=0, atol=2e-6)
        assert (p[:, 6, 1:-1] == 0).all()
        decoded = (2 * p.argmax(-1) / 255 - 1).astype(np.float32)
        assert np.allclose(a, decoded, atol=2e-7, rtol=0)
        assert hashlib.sha256(a.tobytes()).hexdigest() == raw["samples"][i]["actions_sha256"]
    results[phase] = calc
rows = []
for i in range(10):
    a, b = ref[f"actions_{i}"], quant[f"actions_{i}"]
    p, q = ref[f"probabilities_{i}"], quant[f"probabilities_{i}"]
    assert np.array_equal(b, quant[f"actions_{i + 10}"]) and np.array_equal(
        q, quant[f"probabilities_{i + 10}"]
    )
    row = {
        "continuous_action_mae": float(np.abs(a[:, :6] - b[:, :6]).mean()),
        "action_max_abs": float(np.abs(a - b).max()),
        "gripper_agreement": float((a[:, 6] == b[:, 6]).mean()),
        "probability_mean_tv": float(np.abs(p - q).sum(-1).mean() / 2),
    }
    for k, v in row.items():
        assert np.isclose(v, report["paired_differences"][i][k], rtol=0, atol=1e-12)
    rows.append(row)
summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
summary["max_action_difference"] = max(r["action_max_abs"] for r in rows)
result = {
    "status": "passed",
    "requests_checked": 30,
    "distributions_checked": 30 * 8 * 7,
    "latency_recomputed": results,
    "paired_difference_summary": summary,
    "repeated_quantized_actions_and_probabilities_exact": True,
    "p50_speedup_same_hardware": results["bf16_stream"]["p50_seconds"]
    / results["resident_w4"]["p50_seconds"],
    "quality_equivalence": "not_established; this audit checks numerical record consistency, not quantization quality",
    "closed_loop_success": "not_evaluated",
    "files_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.suffix in (".json", ".npz", ".jsonl") and p.name != "audit.json"
    },
}
(root / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2))
