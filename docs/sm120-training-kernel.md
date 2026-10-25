# SM120 training kernels

Current evaluation plan (2026-10-25): continue the same100k training state and evaluate only the final100k checkpoint. The migration evidence below is preserved; its original ten-evaluation plan is superseded by the [final-only contract](training-b64-100k.md).

The current backend is `cublas_sm120_forward`: cached cuBLAS forward dispatch plus
the accepted Triton backward. Its update109 migration passed exact full-model
loss/gradient and tensor-byte/RNG restart checks, with11.88→7.48s/update in a short
matched comparison. See the [forward investigation](sm120-forward-numerics.md).
The following records the first, update38 backward-only migration.

The batch-64 run preserves its complete update-38 checkpoint and changes only the
expert GEMM backend. Rank, global batch, optimizer state, all-rank RNG, resampling
objective, data order and the original 100k learning-rate schedule remain fixed.
No disposable validation weights enter production.

The first selected backend was `triton_sm120_backward`. It keeps the original PyTorch
grouped-linear **forward** and replaces its input/weight-gradient GEMMs with
SM120-compatible Triton kernels. On the pinned PyTorch 2.9.1 build, the native
grouped GEMM takes a per-expert matrix-multiply fallback on SM120. The new backward
keeps expert offsets on the GPU and computes groups in one launch per gradient
operation. Each output tile has one writer; there are no floating-point atomic
reductions or runtime autotuning. Inputs/output gradients use BF16 with FP32
accumulation; LoRA master parameters and AdamW state remain FP32.

This is a training optimization. Forward and inference still use the original
implementation; these measurements establish no inference speedup. This does not
change DecisionVLA's architecture, denoising procedure or confidence definition.

## Why preserve the forward implementation?

Replacing all expert projections passed isolated GEMM tests but failed the
predeclared full-model gradient screen. Restricting the replacement to LoRA
projections also failed. Both attempts and their source snapshots are preserved.
The subsequent [forward investigation](sm120-forward-numerics.md) isolated two
parts of this divergence: cuBLAS reduced-precision intermediate accumulation in
long inner products, and amplification through changes in top-8 expert selection.
The later cuBLAS dispatch optimization passed its own gates and was adopted
from the preserved update109 state.

With the original forward retained, the same global batch from the same parent
state produces identical loss. Raw-gradient relative L2 error is 1.0901%, cosine
similarity is 0.9999409 and norm change is 0.0745%. All input/corruption hashes and
initial adapter hashes match. An original-versus-original control is exact.
This passes the unchanged 5% L2 / 0.999 cosine / 2% norm / 1% loss screen; it is
not bitwise equivalence between old and new gradients, nor evidence of improved
policy quality.

The independent restart gate compares five continuous updates against two updates
plus three updates in a fresh process. It checks every adapter/optimizer tensor,
all-rank RNG and the sample/loss/gradient/LR sequence. See the
[execution evidence](../reports/training-throughput/sm120-20261025/).

## First migration procedure (update38)

Use the pinned CUDA 13 runtime on an SM120 device, with Triton supplied by the
pinned Torch installation. The Dockerfile includes a C compiler and libc headers
for Triton's runtime JIT. No Torch, Transformers or CUDA dependency was upgraded
for this migration.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 \
  --max-restarts=0 -m decisionvla.train_distributed \
  --model /path/to/base --data /path/to/libero \
  --config configs/libero_spatial_r32_b64_100k_sm120.json \
  --resume /path/to/preserved/checkpoint-000038 --allow-kernel-migration \
  --output runs/b64-100k-sm120 --stop-at-step 10000
```

Migration requires a complete local checkpoint including its verified optimizer
and RNG state. `--allow-kernel-migration` permits only the explicit backend change;
it does not allow a simultaneous batch, learning-rate, dataset or budget change.
Later segments resume their own checkpoint with the identical configuration and
without that flag. The ten 500-episode development evaluations and final release
conditions remain in the [training contract](training-b64-100k.md).

`--diagnostic` marks short disposable checkpoints and limits a process to at most
five updates. Production rejects these checkpoints. `triton_sm120` and
`triton_sm120_lora` are retained for reproducing the rejected numerical trials;
they are not the selected training configuration.

`cublas_sm120_forward` is the subsequent validated implementation with the accepted
Triton backward. It retains the original per-expert cuBLAS arithmetic but acquires the
handle once per grouped call. It passed the full DDP gradient, restart and end-to-end timing gates before
production resumed from update109. The old source and checkpoints are preserved.
