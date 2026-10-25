"""Compare exact-shape forward dispatch on privately captured expert operands.

This may run beside training under an allocator cap; all timings are explicitly
contended microbenchmarks, never estimates of full training or inference speed.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

from decisionvla._cublas_forward import linear_forward as cublas_forward
from decisionvla._triton_gemm import linear_forward as triton_forward
from transformers.integrations.moe import _grouped_linear


def difference(a, b):
    a, b = a.float(), b.float()
    return {"relative_l2": float((a-b).norm()/a.norm().clamp_min(1e-20)),
            "max_abs": float((a-b).abs().max()),
            "different_fraction": float((a != b).float().mean())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.035)
    methods = {"torch": _grouped_linear, "cached_cublas": cublas_forward, "triton": triton_forward}
    rng, rows = random.Random(20261025), []
    for path in sorted(args.operands.glob("private-linear-*.pt")):
        data = torch.load(path, weights_only=True, map_location="cpu")
        x, w, offsets = [data[name].cuda() for name in ("x", "weight", "offsets")]
        del data
        expected = _grouped_linear(x, w, offsets)
        actual = cublas_forward(x, w, offsets)
        assert torch.equal(expected, actual)
        triton = triton_forward(x, w, offsets)
        before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            full_precision = _grouped_linear(x, w, offsets)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before
        row = {"file": path.name, "weight_shape": list(w.shape), "input_shape": list(x.shape),
               "cublas_exact": True, "triton_vs_original": difference(expected, triton),
               "no_reduction_vs_original": difference(expected, full_precision),
               "triton_vs_no_reduction": difference(full_precision, triton)}
        # Compute FP64 reference for up to 8 actual rows/64 columns in the busiest expert.
        ends = offsets.cpu().tolist()
        counts = [end-start for start, end in zip([0]+ends[:-1], ends, strict=True)]
        e = max(range(len(counts)), key=counts.__getitem__)
        start = 0 if e == 0 else ends[e-1]
        end, columns = min(start+8, ends[e]), min(64, w.shape[1])
        exact = (x[start:end].double() @ w[e, :columns].double().T).bfloat16()
        row["fp64_subset"] = {"expert": e, "rows": end-start, "columns": columns,
            "torch": difference(exact, expected[start:end, :columns]),
            "triton": difference(exact, triton[start:end, :columns]),
            "no_reduction": difference(exact, full_precision[start:end, :columns])}
        del actual, triton, full_precision, exact
        timings = {name: [] for name in methods}
        for fn in methods.values():
            for _ in range(2):
                fn(x, w, offsets)
        for _ in range(12):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                methods[name](x, w, offsets)
                torch.cuda.synchronize()
                timings[name].append(time.perf_counter()-start_time)
        row["wall_seconds"] = timings
        row["medians"] = {name: statistics.median(t) for name, t in timings.items()}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "wall_seconds"}), flush=True)
        del x, w, offsets, expected
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps({"status": "completed", "rows": rows,
        "contended_microbenchmark_only": True, "production_training_unchanged": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}, indent=2)+"\n")


if __name__ == "__main__":
    main()
