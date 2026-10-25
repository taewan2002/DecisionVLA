# Fresh batch-64 training: execution evidence

The owner stopped the small-batch continuation at update 2,534 on 2026-10-25.
Checkpoint 2,500 is its latest complete recovery point; the original 2k checkpoint,
logs and frozen source remain preserved. The replacement run starts fresh LoRA from
the pinned base, rather than carrying over pilot parameters or optimizer state.

The fixed [training contract](../../docs/training-b64-100k.md) is rank 32,
two RTX PRO 6000 Blackwell 96GB GPUs, global batch 64, 100,000 updates, and
500 LIBERO-Spatial episodes at every 10k checkpoint. The corrected resampling-only
objective applies from update 1. Policy performance from this run is pending.

[The production startup snapshot](startup.json) confirms the first completed update
at 2026-10-25 00:34 KST, with global batch 64, finite loss/gradients, both devices
capped at 600 W, GPU1 mining stopped, and the 88-file source manifest unchanged.
The first update took 36.39 seconds including warmup. This snapshot records launch,
not completion of the 100k budget or any closed-loop evaluation.

## Capacity checks

| Per-device microbatch | Accumulation | Global batch | Disposable result |
| ---: | ---: | ---: | --- |
| 4 | 8 | 64 | Three updates completed; post-warmup updates took 10.29 and 9.71 seconds |
| 8 | 4 | 64 | One update completed; update 2 ran out of memory on both devices |
| 4, deterministic algorithms | 8 | 64 | Three updates completed; post-warmup updates took 21.19 and 20.45 seconds |

The adopted configuration is microbatch 4. The failed larger-microbatch trial used
93.29/93.43 GiB of allocated PyTorch memory at failure. These measurements include
data preparation, forward/backward and optimizer work, but exclude periodic
validation, checkpoint writes and LIBERO evaluation. Only two post-warmup timings
are available at this stage; they are an initial runtime estimate, not a stable
throughput benchmark or an inference measurement. See [capacity-trials.json](capacity-trials.json).

The first disposable launcher attempt failed before any update because a
network-isolated container could not resolve its rendezvous hostname. Explicit
loopback rendezvous resolved that setup issue. All failed attempts remain separate
from production, and no disposable weights are reused.

The initial microbatch-4 resume comparison failed its unchanged 1e-5 absolute
parameter tolerance: the first failing tensor differed by up to 2.79e-5. Two fresh
executions had identical initial adapter hashes, sample keys and first forward loss,
but different first backward norms, before any checkpoint reload. This identifies
training nondeterminism as a problem; the exact offending kernel has not been
isolated. The new execution enables strict deterministic algorithms and a fixed
cuBLAS workspace. It passed the original gate without relaxing its tolerance.
The deterministic configuration peaked at 83.77 GiB of allocated PyTorch memory
per rank. Its two post-warmup timings are in
[deterministic-preflight-timing.json](deterministic-preflight-timing.json). At that
initial speed, training alone is roughly 24 days; ten evaluations add time.
The prior 10-second/update estimate does not describe the adopted configuration.

## Required checks

The CPU suite passed 46 tests, including a real two-process gradient-averaging
comparison against the global equal-observation objective. Ruff passed. The
[mocked controller rehearsal](control-rehearsal.json) checks all ten ordered
training/evaluation boundaries and an injected 60k evaluation failure that prevents
70k training. It is not evidence that those training segments or rollouts occurred.

The GPU gate compares continuous three-update training with two updates plus one
fresh-process resumed update. Adapter and optimizer tensors must meet absolute
tolerance 1e-5; every rank's RNG state must match exactly.
[The completed gate](resume-gate.json) measured zero adapter and optimizer
differences and exact RNG states. [An independent sequence check](resume-sequence-audit.json)
also verified sample keys, losses, gradient norms and LR at all three updates.
This specific check was bitwise equal; it does not promise equality across different
hardware, software or configurations. Fresh production training was launched after
the gate passed on 2026-10-25, with no disposable weights reused.

[The source audit](source-audit.json) verifies 88 frozen files, all ten prepared
evaluation source trees, preserved older source snapshots and an identical local
source tree. [The source manifest](source-sha256.json) supports reproducing the
numerical implementation. Private host operations remain outside the repository.

## Interpretation and release

This changes initialization, batch, objective history and training budget together.
It is not an isolated batch-size or objective ablation. Batch 64 × 100k gives
6.4 million observation presentations; that arithmetic does not establish equal
compute or expected performance to OpenVLA.

The same 500 official initial states are reused at ten checkpoints. All results and
audits are required before selection by task-macro success, with earliest-step ties.
These are development results, not independent tests. Raw confidence remains an
uncalibrated diagnostic. Final publication is held until the full comparison,
audits and review are complete.
