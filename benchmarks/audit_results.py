"""Independently recompute percentiles and check recorded workload invariants.

Usage: python benchmarks/audit_results.py reports/base-latency-2026-10-24
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (x - lo) * (values[hi] - values[lo])


def main():
    root = Path(sys.argv[1])
    summaries = {}
    for path in sorted(root.glob("*/results.json")):
        report = json.loads(path.read_text())
        assert report["status"] == "completed", path
        assert not report["weights_trained_for_actions"]
        assert report["engine_args"]["enable_prefix_caching"] is False
        assert report["engine_args"]["mm_processor_cache_gb"] == 0
        assert report["engine_args"]["async_scheduling"] is False
        assert report["engine_args"]["kv_cache_dtype"] == "bfloat16"
        seen_inputs = set()
        cases = []
        for case in report["cases"]:
            samples = case["samples"]
            assert len(samples) == report["repeats"] == 30
            n = report["warmups"] + report["repeats"]
            assert case["worker"]["execute_count"] == n * (case["steps"] + 2)
            assert case["worker"]["scheduled_token_histogram"]["256"] == n * (case["steps"] + 1)
            assert case["worker"]["confidence_threshold"] == -1.0
            values = [s["ms"] for s in samples]
            assert all(math.isfinite(v) and v > 0 for v in values)
            assert abs(percentile(values, 0.50) - case["p50_ms"]) < 1e-8
            assert abs(percentile(values, 0.95) - case["p95_ms"]) < 1e-8
            assert all(s["output_tokens"] == 56 for s in samples)
            for sample in samples:
                assert sample["image_sha256"] not in seen_inputs
                seen_inputs.add(sample["image_sha256"])
            cases.append({k: case[k] for k in ["views", "steps", "p50_ms", "p95_ms"]})
        assert {(c["views"], c["steps"]) for c in cases} == {(v,s) for v in (1,2) for s in (1,2,4,8)}
        telemetry = path.parent / "gpu-telemetry.csv"
        with telemetry.open() as f:
            rows = list(csv.DictReader(f))
        peak = max(float(r[" memory.used [MiB]"].strip().split()[0]) for r in rows)
        summaries[path.parent.name] = {
            "cases": cases, "timed_requests": len(seen_inputs),
            "observed_device_memory_max_GiB_200ms_sampling": peak / 1024,
            "peak_torch_allocated_GiB": max(c["worker"]["peak_allocated_GiB"] for c in report["cases"]),
            "peak_torch_reserved_GiB": max(c["worker"]["peak_reserved_GiB"] for c in report["cases"]),
            "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "telemetry_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
        }
    assert summaries, "No completed reports found"
    if "nvfp4" in summaries and "bf16" in summaries:
        a = json.loads((root / "nvfp4/results.json").read_text())
        b = json.loads((root / "bf16/results.json").read_text())
        assert a["torch"] == b["torch"] and a["vllm"] == b["vllm"]
        for x,y in zip(a["cases"], b["cases"]):
            assert (x["views"],x["steps"]) == (y["views"],y["steps"])
            assert [s["image_sha256"] for s in x["samples"]] == [s["image_sha256"] for s in y["samples"]]
    audit = {"status": "passed", "method": "independent standard-library recomputation",
             "claims_excluded": ["VLA success", "trained-model latency", "RTX 5090 speed"],
             "results": summaries}
    (root / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
