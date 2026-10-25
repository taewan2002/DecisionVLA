# Live training bottleneck diagnosis — 2026-10-25

The current SM120 runtime uses the per-expert fallback inside PyTorch's
`torch._grouped_mm`. This is a concrete optimization target; the current
approximately 21-second update is not a demonstrated hardware throughput limit.
No training source, objective, batch, learning rate, controller, or checkpoint
was changed by this investigation.

## Measurements

Run: `libero-spatial-r32-b64-100k-v4`, fresh rank32, global batch64,
microbatch4 × accumulation8 × two GPUs. Runtime: PyTorch2.9.1+cu130,
revision `5811a8d7da873dd699ff6687092c225caffcf1bb`.

45 one-second telemetry samples, 00:39:23–00:40:07 KST:

| Metric | GPU0 | GPU1 |
|---|---:|---:|
| Mean GPU utilization | 41.2% | 38.9% |
| Mean power / configured limit | 234.5 / 600W | 240.6 / 600W |
| Device memory used | 87,467MiB | 87,404MiB |
| Mean SM clock | 2,784MHz | 2,832MHz |
| Mean temperature | 60.1°C | 62.7°C |
| Process CPU cores used | 1.028 | 1.027 |

The container has a 12-core CPU quota and had zero recorded CPU throttling.
The subsequent performance snapshot showed neither power-cap nor thermal
slowdown active. GPU utilization measures time with a kernel executing,
**not** the fraction of CUDA/Tensor Cores occupied or achieved FLOPS.
See [NVIDIA's definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization).

At 00:43:56 KST, production reached step27 with finite loss4.57649.
The median of updates2–27 was20.8675s. Earlier matched-shape disposable
preflight updates were9.7101–10.2898s before the strict deterministic settings
and20.4497–21.1877s afterward. This comparison changes the deterministic
algorithm, cuDNN and cuBLAS configuration together; it does not isolate their
individual contributions. Earlier non-deterministic trials failed the resume
comparison and are not production training.

## Where execution waits

Each rank was sampled for45s at50Hz with nonblocking Python stack reads.
Only main-thread samples form the denominator (2,185 /2,186 samples).

| Main-thread location | Rank0 | Rank1 |
|---|---:|---:|
| Data preparation | 0.50% | 1.19% |
| Encoder forward | 23.34% | 22.19% |
| Decoder forward | 12.59% | 13.08% |
| Backward engine | 61.69% | 63.08% |

These are sampled host call locations, not GPU kernel durations. Backward
includes checkpoint recomputation and may include communication waits;
NCCL time has not been separated. Nonblocking sampling reported64 /63 read
errors. Additional autograd-thread samples are excluded from the table to
avoid double counting wall time. Data loading was not the leading sampled
phase; increasing DataLoader workers is therefore a lower priority.

### Confirmed grouped GEMM fallback

The device reports compute capability12.0 (SM120). At the installed PyTorch
revision, `_grouped_mm_cuda` selects its fused BF16 fast path only for the
SM90/SM100 device check. Otherwise it copies expert offsets to CPU and loops
over groups with `mm_out`.

An isolated synthetic BF16 forward probe in the **same container/runtime on
GPU0** confirmed **one `aten::_grouped_mm` →128 `aten::mm` calls**, for128
groups. The constant-input result was correct. The probe loaded no model,
performed no training update, and used34.5MiB peak PyTorch allocation
(CUDA context overhead is not included). Its timings are concurrent with
production and must not be treated as standalone benchmarks. The recorded
operator counts, not elapsed time, establish the dispatch behavior.

The Transformers Python dispatcher reaches `torch._grouped_mm`; this is the
fallback **inside PyTorch**, not Transformers' separate Python fallback.
See the [pinned dispatcher](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cuda/Blas.cpp#L1785)
and [fallback](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/GroupedMMUtils.h#L115).

### Additional host overhead

A20s native CPU sample contains substantial regex/locale processing, alongside
CUDA launch work. The pinned cuBLAS handle implementation reparses
`CUBLAS_WORKSPACE_CONFIG` with a regular expression on workspace-size queries.
An isolated host-only query measured approximately0.96µs with the variable
unset and16.2–16.3µs with `:4096:8` (two alternating trials, median of five
1,000-query repetitions each). This supports an additional cost amplified by
the many small GEMMs; it does not establish what fraction of the whole
update's slowdown comes from that parser.

See the [pinned workspace implementation](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/cuda/CublasHandlePool.cpp).
The Python profiler's native-unwind attempt failed with `UNW_EBADREG` and is
not used as supporting evidence. Native CPU samples and Python stack samples
are separate measurements, not a complete CUDA timeline.

## Optimization priority

1. Validate an SM120-compatible fused grouped GEMM forward/backward for the
   existing expert LoRA shapes, including empty groups, routing and gradients.
2. Retain the fixed global batch and numerical objective. Measure the full
   update and independently verify finite losses, gradients, optimizer/RNG
   resume and checkpoint provenance before adopting a runtime change.
3. Reassess deterministic-runtime overhead, communication and optimizer costs
   after removing the per-expert launch loop. Do not infer an end-to-end speedup
   from the small dispatch probe.

The previous microbatch8 trial OOMed on its second update. Increasing gradient
accumulation changes the effective batch but does not increase simultaneous
work per forward. Neither more accumulation nor a higher power cap addresses
the confirmed fallback directly.

## Evidence and preservation

- `summary.json`: live telemetry aggregates and process I/O counters.
- `python-sampling-summary.json`: main-thread sample categories and counts.
- `grouped-dispatch.json`, `probe-grouped-dispatch.py`: operator-count probe.
- `runtime-source-evidence.json`: exact upstream revision, file SHA and lines.
- `snapshot.json`: timestamped production status and update timing.
- `preservation-audit.json`: frozen source verification and resource state.

All88 frozen numerical-source hashes remained unchanged. GPU1 mining remained
stopped. Other project containers retained their IDs, PIDs and start times.
Raw host profiler data remains private. No new automation, commit, push or
release was created. Production continued throughout; telemetry/profiling and
the tiny dispatch probe can introduce measurement overhead.
