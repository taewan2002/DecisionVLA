# SM120 forward investigation and validated migration

The original full-Triton forward changed intermediate BF16 arithmetic. The pinned
cuBLAS fallback permits reduced-precision reductions, while the Triton kernel uses
FP32 accumulators. Small projection differences change top-8 MoE choices and then
amplify through the model. This does not establish that the Triton arithmetic is
less mathematically accurate or that either policy has higher success.

Evidence:

- `micro-v1.json`: six synthetic projection shapes; K tiles 16/32/64/128,
  reduced-precision control, FP64 subset and direct cuBLAS comparison.
- `trace-v1.json`, `trace-v2.json`: the same first rank-0 microbatch from preserved
  update38; all input/corruption bytes match the original failed comparison.
  Triton changes 21,512/66,120 token-layer expert sets (32.53%). Keeping reference
  IDs reduces final-logit relative L2 from10.05% to2.20%; original weights are
  recomputed on those IDs, not copied. `triton_fixed_routes` records proposed
  router changes before forcing reference IDs. Cached cuBLAS is exact throughout.
- `forward-bench-v1.json`: six actual captured projections, 12 interleaved samples
  per method, with training/diagnostic contention. Individual GEMM speedups
  1.67–4.70× are not end-to-end training/inference results.
- `global-forward-v1.json`: both ranks' complete batch64 replay, 16 microbatches;
  router/expert tensors and logits match exactly. Its `mean_loss` is a **masked
  token-average diagnostic**, not the equal-observation DDP training objective.
  This forward-only test performs no optimizer updates and is not a restart gate.
- `independent-audit-before-migration.json`: raw-logit metrics recomputed, repeated
  diagnostic outputs exact, original94 source SHA and parent38 payloads unchanged.
- `full-numerical-v1.json`: after the owner's subsequent instruction to apply a
  validated fix, training stopped gracefully at109. From that complete checkpoint,
  actual two-GPU/global64 training loss and raw gradients of all families are
  **exactly equal numerically** between the accepted backward-only backend and cached
  cuBLAS forward. This is a real gradient check using the original DDP objective.
- `control-rehearsal.json`: CPU mocks check ten training/evaluation boundaries and
  failure cleanup. These are not real rollouts or training measurements.

The candidate keeps the same cuBLAS GEMM dimensions and arithmetic, calls handle
setup once per group, and reuses the accepted Triton backward. It does not fuse all
forward experts into a new GPU kernel. See
[implementation and interpretation](../../../docs/sm120-forward-numerics.md).

Native host code, model/data/config revisions and execution evidence remain
separate from private captured operands. The operands include base-model weights;
those tensors, optimizer states, raw checkpoints and private operations are not
public release assets. No RTX5090, NVFP4 or policy-success claim follows here.

## Accepted training migration

The full GPU0/1 gate uses the actual update109 checkpoint. No loss, raw-gradient,
sample, corruption, LR or five-update trajectory difference was measured. The
restart gate checks2,130 raw tensor byte arrays, plus all-rank RNG and metadata.
`gates/timing.json` records11.8788→7.4821s/update (1.5876×) over four warmed steps
per backend; peakallocated83.768GiB/GPU. Checkpoint, validation and evaluation
costs are excluded. This timing run has exclusive GPU0/1 access; the earlier
forward probes have explicit contention labels.

`gates/acceptance.json`, `gates/resume-gate.json`, `gates/resume-sequence.json` and
`source-sha256.json` seal the accepted implementation. CPU51 tests and Ruff pass;
15 GPU kernel cases pass (the kernel suite also contains one CPU config check).
The separate diagnostic's initial library-path errors and the CPU test staging
failure from omitted config files are preserved privately; neither changed
production weights. The repeated complete CPU suite passed after staging configs.

The former controller's AssertionError follows the deliberate checkpointed stop
before10k, as designed by its previous boundary assertion. It is preserved as raw
history. The new controller uses the saved update109 state and the same100k plan;
no diagnostic weights, new initialization or new recurring schedule is used.

Startup at01:36KST confirmed production117/100k; the first five updates match the
accepted disposable sequence, and seven steady updates have median7.4796s. Thirty
telemetry samples per GPU show83.8%/90.3% NVIDIA activity and431W/445W; these are not
Tensor Core utilization measurements. `startup.json` and `resource-audit.json`
record this snapshot; read live status for later progress.
