# Effective-batch-preserving throughput probe

Historical measurement from2026-10-24. On2026-10-25 the owner stopped the small-batch
continuation and authorized a [fresh two-GPU batch64/100k experiment](../../../docs/training-b64-100k.md).
That decision does not change the measurements or failed numerical screen below.

The owner requested a larger batch and a possible two-GPU configuration. This
isolated benchmark used one RTX PRO 6000 Blackwell, physical GPU1, with a temporary
600 W cap. The existing GPU0 training and queued 50k continuation were preserved.
GPU2 was occupied by a separate training/validation workload; no two-GPU training
or collective-communication benchmark was run.

## Measured throughput

Each condition restores the same step-2000 rank-32 adapter, AdamW state, sampler
and noise RNG. It performs eight **disposable** updates with the corrected
resampled-only objective and effective batch eight. The first two updates are
excluded from timing; medians below use six updates. The frozen BF16 base,
FP32 LoRA master parameters and optimizer, 280 visual tokens, 256-position canvas,
eight-action horizon, and gradient checkpointing are unchanged.

| Microbatch | Accumulation | Seconds/update, median | Relative speed | Peak allocated GiB |
|---:|---:|---:|---:|---:|
| 1 | 8 | 6.833 | 1.00x | 71.119 |
| 2 | 4 | 3.843 | 1.78x | 74.091 |
| 4 | 2 | 2.257 | 3.03x | 80.036 |

These are short training measurements while other jobs used the same host. They
include observation preprocessing, forward/backward and optimizer work, but not
model loading, checkpoint writing, periodic validation or LIBERO rollouts.
They are not standalone inference timings or RTX 5090 measurements. No OOM or
nonfinite values occurred. The benchmark baseline calls the existing update
implementation; experimental batching is not enabled in the production trainer.

## Numerical validation and limits

All 64 sample keys, per-update supervision counts, noise RNG states and sampler
states match across conditions. Padding is masked in encoder and decoder
attention; decoder rotary positions use each unpadded prefix length. Each
observation retains equal loss weight even when supervised token counts differ.
Forty CPU tests and Ruff pass.

Nevertheless, the predeclared gradient screen **failed** for both larger batches:

| Microbatch | First loss relative difference | Clipped gradient relative L2 difference |
|---:|---:|---:|
| 2 | 0.2785% | 16.8314% |
| 4 | 0.2985% | 15.8106% |

The limits were 1% for loss and 5% for gradient. These were an implementation
comparison screen, not a validated criterion for training quality. A failed screen
does not demonstrate worse policy performance; it does preclude claiming the same
numerical training trajectory. The thresholds were not relaxed after seeing the
results. No candidate was adopted and no benchmark weights were exported.

A separate zero-update probe used the same eight observations (prefix lengths
294–298). Padding single observations changed 2.46% of action-token argmaxes;
batching by two/four changed 3.13%/3.35%. Identical-length duplicated observations
also changed logits and routed expert sets, eliminating padding as the sole
explanation. For that one duplication control, approximately 26% of token-layer
expert sets changed; this count includes the native decoder's unused canvas
query positions, not just supervised actions. These results are consistent with
batch-shape-dependent low-precision arithmetic affecting discrete MoE routing;
they do not establish that every discrepancy comes from that mechanism.

PyTorch documents that batched and sliced operations need not be bitwise equal:
[Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
Changing precision or routing to remove this difference would itself require a
separate recorded configuration and verification. A longer learning comparison
and checkpoint/resume validation are needed before adopting the candidate.

## Artifacts and resource restoration

- `batch-1.json`, `batch-2.json`, `batch-4.json`: timings, memory, losses, sample keys,
  RNG hashes, and the failed numerical screens.
- `numerics.json`: padding and same-length duplication controls, zero updates.
- `telemetry.csv`: GPU1 memory, power, utilization and temperature during the sweep.
- `completion-audit.json`: independent summary arithmetic, file hashes and checks.

Both finite controllers completed and restored the same GPU1 miner container and
its original 500 W cap. GPU0 DecisionVLA, GPU2 training/validation and GPU3 serving
container IDs/start times remained unchanged. Preserved 10k source38, queued50k
source58 and both diagnostic source manifests passed SHA checks. The existing
10k process and the 50k single-GPU schedule remain in force. No recurring automation,
commit, push or release was created.
