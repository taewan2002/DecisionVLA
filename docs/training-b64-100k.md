# Fresh rank-32 training with global batch 64

> **Historical procedure — superseded on2026-10-25.** Training was checkpoint-stopped; the forty-task run ended at46 and its automatic final evaluation did not start. These commands describe the earlier action-only implementation. See the [current DecisionVLA design](architecture.md) and [development plan](development.md); do not restart old controllers.

**Historical Spatial-only run, stopped at6,681 on2026-10-25.**
The replacement is the [fresh forty-task experiment](training-libero40.md).
The settings and execution descriptions below belong to the preserved Spatial run.

This run replaces the interrupted small-batch continuation. It is a new experiment,
initialized from the pinned DiffusionGemma base with fresh LoRA parameters and a new
optimizer. Pilot checkpoints, source snapshots and logs remain separate. Results are
pending; a larger batch is not a demonstrated success-rate or robustness improvement.

| Setting | Fixed value |
| --- | --- |
| Trainable parameters | Shared rank32 attention, dense MLP, per-expert LoRA, action embedding deltas |
| GPUs | Two RTX PRO 6000 Blackwell 96GB devices |
| Per-device microbatch | 4 |
| Gradient accumulation | 8 |
| Global batch | 4 × 8 × 2 = 64 observations per optimizer update |
| Training budget | 100,000 optimizer updates / 6.4 million observation presentations |
| Objective | Mean of each observation's cross-entropy over valid randomly resampled positions |
| Optimizer | AdamW, FP32 trainable parameters/state, weight decay0, clipping1 |
| Learning rate | 1,000-update warmup from5e-6 to5e-5, cosine to5e-6 at100k |
| Save / demonstration validation | Every500 updates / the same100 held-out observations |
| Closed-loop development evaluation | 500 Spatial episodes only at the fixed100k checkpoint |

Random resampling positions remain supervised even when the sampled noise happens
to equal the correct token. Unchanged positions do not receive an identity-copy
reward. Each observation has equal objective weight even when its number of
supervised tokens differs.

All ranks generate the same deterministic global index sequence and take disjoint
positions in it. Sampling remains with replacement, so naturally duplicated
observations can occur. Each rank has a separately seeded corruption generator.
DDP averages gradients across ranks; losses are divided by eight accumulation
passes. Synchronization occurs on the last pass before clipping and updating.
Model parameters are replicated; GPU memory is not combined into a single192GB pool.

The current implementation uses right-padded prefixes with explicit padding masks
and decoder positions derived from each unpadded prefix length. The numerical path
is not claimed to be identical to the earlier singleton-BF16 path. MoE routing can
change with batch-shaped floating-point arithmetic.

The original two-GPU run is preserved through update38; the backward-only
migration continued through update109. The current runtime uses
`configs/libero_spatial_r32_b64_100k_cublas.json` and resumes the complete update109
state with `cublas_sm120_forward`. It retains cuBLAS forward arithmetic, reduces
repeated host setup, and keeps the accepted Triton expert gradients. Full-model
loss and raw gradients match exactly. Continuous5 versus2+3 restart matches all
adapter/optimizer tensor bytes and all-rank RNG. A short matched measurement gives
11.88→7.48s/update (1.59×); the earlier20.57→12.26s result belongs to the first,
separate migration comparison. The original100k LR remains fixed. The latest owner instruction removes intermediate rollouts: only the final100k checkpoint is evaluated. This control change preserves the current production state; it does not reset training.
[Forward cause and migration](sm120-forward-numerics.md) ·
[Measured evidence](../reports/training-throughput/sm120-forward-20261025/).

## Reproducible launch

Use the pinned dependencies in `pyproject.toml`, an authorized local copy of the base
model and the original LIBERO HDF5 files. Model and dataset revisions, split, action
alignment and preprocessing are fixed in the configuration.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 \
  --max-restarts=0 -m decisionvla.train_distributed \
  --model /path/to/base --data /path/to/libero \
  --config configs/libero_spatial_r32_b64_100k_cublas.json \
  --output runs/b64-100k --stop-at-step 10000
```

At each process boundary, immediately resume the same output/config using
`--resume runs/b64-100k/checkpoint-010000 --stop-at-step 20000` (and subsequent
boundaries). These segments retain the original72-hour per-process limit; they do not wait for rollout evaluation. Process reload adds a short overhead. The configured end step always remains100k, so segmentation does not restart the LR schedule. Resume requires the same world size and batch configuration.
All per-rank sampler/noise/Python/NumPy/Torch/CUDA RNG states and the optimizer are
saved atomically. Full trainable-weight hashes must agree across replicas at every
checkpoint. A file named `STOP` in the output directory requests a checkpointed stop
after the current update; remove it only when intentionally resuming.

The trainer enables strict deterministic algorithms and cuDNN deterministic mode,
disables cuDNN benchmarking, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization. Unsupported nondeterministic operations raise an error. The
[PyTorch documentation](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html)
describes these controls; the real resume gate remains necessary because these
settings alone do not establish end-to-end reproducibility.

The private operator controller runs the ten training segments consecutively, then the final denoising probe,500-episode rollout and probability/action trace audits. The current contract is `configs/training_schedule_b64_100k_finalonly.json`. It stops on errors and restores temporary power changes.
The owner's explicit mining stop persists through evaluations and completion.
Disposable continuous3 versus segmented2+1 training must pass parameter/optimizer
absolute tolerance1e-5 and exact RNG checks before production. No smoke weights are
reused. Numerical tolerance is not a claim of bitwise-identical floating-point resume.

A larger per-device microbatch8 (four accumulation passes, still global64) completed
one disposable update but ran out of memory on update2 on the96GB devices. Its
failure evidence is preserved; no weights from the capacity trial are reused.
The adopted microbatch4 retains eight accumulation passes. Initial timing and
resume evidence are tracked in the [execution report](../reports/training-b64-100k/).

## Evaluation and publication

The fixed500 official initial states have already been used for development. The final checkpoint is fixed at100k before observing its results; there is no performance-based checkpoint search. One500-episode evaluation is planned, not5,000 episodes or an unseen test. Intermediate10k checkpoints remain available, but additional rollout evaluation requires a new decision.

The same100 held-out demonstration observations are checked every500 updates. These open-loop losses and action errors can reveal numerical or imitation problems, but do not establish closed-loop success or rule out a poor policy. Final failures and undefined single-class AUROCs are reported.
Raw confidence is not a calibrated success probability or causal explanation.

Publish code, configuration, figures and audited reports in one final commit/push.
The fixed100k experimental adapter release contains only the four safetensors shards,
index, adapter config, model card and SHA manifest. Do not publish base weights,
datasets, optimizer/RNG files, private operations or credentials.

The owner's deployment candidate is expert W4A16 with fused kernels and CUDA
Graph. After the fixed final BF16 evaluation, validate an isolated export on the
same500 initial states and measure the RTX5090 deployment path. The existing
training controller is unchanged; the [final deployment contract](final-deployment.md)
records this additional validation and publication work separately.

## Why batch64?

This is a declared experimental choice, not a batch-size ablation winner.
[OpenVLA's LIBERO model](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
uses rank32, global batch128 and50k updates. The
[OpenVLA-OFT LIBERO reproduction command](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)
uses rank32, eight GPUs with batch8 each (global64), and150k updates for Spatial.
[OpenPI's LIBERO LoRA examples](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
inherit a global batch32 default. These are different models and objectives.
Our6.4M observation presentations match the arithmetic128×50k, but do not establish
equal compute, supervised action count or expected performance.
