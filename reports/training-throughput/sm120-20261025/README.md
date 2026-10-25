# SM120 backward-kernel migration

The original batch-64 run stopped cleanly after update **38**. Its four adapter
shards, index, config, training metadata, optimizer/all-rank RNG and SHA manifest
remain unchanged, as do its 88 frozen source files. The new run resumes that state;
it does not reset training or reuse disposable diagnostic weights.

## Adopted implementation

`triton_sm120_backward` keeps the original grouped-linear forward and uses Triton
for expert input/weight gradients. Expert boundaries stay on the GPU; each output
tile has one writer and no atomic reduction. This avoids the per-expert backward
GEMM fallback on the pinned SM120 runtime. See the
[implementation and launch instructions](../../../docs/sm120-training-kernel.md).

The architecture, rank32, batch64, AdamW/RNG state, data/objective and 100k LR
schedule are unchanged. This is **training throughput**, not inference acceleration
or evidence of higher LIBERO success. Forward remains on the original backend.

## Measured throughput

Two RTX PRO 6000 Blackwell 96GB GPUs, 600W caps, BF16 compute, FP32 LoRA/AdamW,
microbatch4 × accumulation8 ×2, strict deterministic settings. Both runs restore
the same update-38 state and use the same sample sequence. Five updates were run;
the first initialization update is excluded below.

| Backend | Median seconds/update | Steady range |
| --- | ---: | ---: |
| Original PyTorch grouped GEMM | 20.572 | 20.536–20.688 |
| Original forward + SM120 Triton backward | 12.261 | 12.228–12.467 |

The observed speedup is **1.678×**, a **40.4% reduction** in update time. Peak
allocated memory is 83.768GiB per rank. Timings include data preparation,
forward/backward and optimizer work; they exclude validation, checkpoint writing
and LIBERO evaluation. Four steady updates per backend are an initial comparison,
not a long-run benchmark. [Raw timing](timing.json).

## Numerical and restart checks

| Full-model comparison from the same parent and batch | Loss relative error | Raw-gradient relative L2 | Gradient cosine | Result |
| --- | ---: | ---: | ---: | --- |
| Original versus original control | 0 | 0 | 1 | Exact |
| Full Triton forward/backward | 0.2707% | 64.9904% | 0.795692 | Rejected |
| Triton only on LoRA projections | 0.1001% | 57.3848% | 0.832922 | Rejected |
| Original forward + Triton backward | 0 | 1.0901% | 0.999941 | Passed |

The predeclared screen is loss relative error <1%, raw-gradient L2 <5%, cosine
>0.999 and norm change <2%. The adopted norm change is 0.0745%. Initial adapters,
all input/corruption hashes and sample keys match. Thresholds were not relaxed.
The rejected trials passed isolated GEMM tests but failed the full-model screen;
their results remain visible in [full replacement](numerical-v1.json) and
[LoRA-only replacement](numerical-lora-v1.json). See the
[accepted comparison](numerical-backward-v1.json) and
[original-versus-original control](reference-control-v1.json).

Five continuous updates (38→43) and a fresh-process 2+3 split (38→40→43) have
**zero adapter and optimizer tensor differences**, exact all-rank RNG, and exact
sample/loss/gradient/LR sequences. This specific same-backend restart is bitwise
equal; old-versus-new gradients are not. No diagnostic weights are reused.
[Tensor/RNG audit](resume-gate.json) · [Sequence audit](resume-sequence.json).

The CPU suite passed 50 tests, with 11 GPU cases skipped. The GPU-enabled kernel
suite passed all 13 tests (11 GPU cases plus two CPU checks). Ruff passed.
The [controller rehearsal](control-rehearsal.json) checks all ten evaluation
boundaries and stops subsequent training on an injected evaluation failure.
The runtime adds a C compiler/libc headers for Triton JIT, without upgrading the
pinned Torch/Transformers/CUDA packages. [Runtime image provenance](runtime-image.json).

The [acceptance record](acceptance.json) verifies the unchanged parent state,
numerical source, 94 current source files and all ten prepared evaluation source
trees. Local and frozen runtime sources match the [source manifest](source-sha256.json).

## Continuation and limits

The continuation is `libero-spatial-r32-b64-100k-sm120-v1`, from original update38
to total100k, with 500 LIBERO-Spatial episodes at every10k. GPU1 mining stays
stopped; GPU2/3 workloads remain outside this project. Startup and subsequent
throughput are recorded separately from the disposable comparison.

At 2026-10-25 01:12 KST, production completed update45. The first five updates
exactly match the accepted disposable sample/loss/gradient/LR sequence, and the
initial adapter SHA matches the preserved parent. Six steady production updates
have a median of12.310s. Both power caps are600W; protected other-job container
identities/PIDs and the94 source hashes are unchanged. [Startup snapshot](startup.json).

A45-sample production window measured mean GPU utilization54.4% on both devices
and mean power306/310W. This is the NVIDIA activity counter, not Tensor Core FLOP
utilization or proof of full hardware saturation. The original forward fallback
remains. [Telemetry summary](production-telemetry-summary.json).

The official initial states are reused development conditions. All ten evaluations,
probability/action/SHA audits and documentation/review remain required before a
single final publication and selected-adapter release. No RTX5090 measurement,
trained NVFP4, calibrated task-success confidence or causal explanation is claimed.
