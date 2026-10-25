# Forward arithmetic and MoE routing

Current evaluation plan (2026-10-25): continue the same100k training state and evaluate only the final100k checkpoint. The migration evidence below is preserved; its original ten-evaluation plan is superseded by the [final-only contract](training-b64-100k.md).

Initial diagnosis ran beside production without changing it. After the owner
authorized applying a validated fix, production saved its full state at update109
for exclusive GPU0/1 validation. The initial investigation used a separate
process, read-only update-38 checkpoint and
CPU masters with at most a 6 GiB CUDA allocator budget. Other GPU jobs, training
configuration, power settings and the optimizer are not modified. Concurrent
diagnostic intervals must not be treated as uncontended throughput measurements.

## Cause of the discrepancy

The pinned Torch 2.9.1 cuBLAS fallback permits BF16 reduced-precision intermediate
reductions. Our Triton GEMM accumulates its BF16 products in FP32. These paths need
not round identically. On the actual first microbatch, the first expert gate/up
projection differs by 0.1707% relative L2, and its rank-32 input projection by
0.1886%. Disabling reduced-precision reduction only in a separate comparison
process reduces the Triton/cuBLAS discrepancy to 0.0105% and 0.0119%, respectively.
Changing Triton K tiles from 16 to 128 does not fix the synthetic long-product
discrepancy. The synthetic FP64 check is closer to Triton, but selected actual
submatrices also contain cases where the original cuBLAS result is closer.
Agreement with a legacy implementation and mathematical accuracy are different
criteria; neither isolated result establishes policy quality.

The difference then reaches a discontinuous choice: the model selects 8 of 128
experts. The router computes its projection in BF16 before applying FP32 softmax.
In this microbatch, 6,629/66,120 token-layer positions have a tie at the reference
8th/9th boundary. The first route change appears in encoder layer 1 (zero-based),
after the first layer's expert arithmetic has changed.

| Same checkpoint/input | Changed expert sets / token-layers | Logit relative L2 | Action-bin argmax agreement |
|---|---:|---:|---:|
| Original repeated | 0 / 66,120 | 0 | 100% |
| All Triton forward | 21,512 / 66,120 (32.53%) | 10.05% | 94.64% |
| Triton, original expert IDs forced | IDs forced to reference | 2.20% | 100% |
| LoRA-only Triton forward | 20,391 / 66,120 (30.84%) | 15.39% | 98.66% |
| Cached cuBLAS dispatch candidate | 0 / 66,120 | 0 | 100% |

The forced-ID control recomputes weights from current router probabilities on the
original IDs; it does not copy the original router weights. Its raw trace records
the router's proposed choices **before** overriding them. The 10.05% → 2.20%
reduction supports routing amplification in this particular input. It is not a
universal decomposition of the earlier 64.99% gradient error. The same input and
corruption SHA matches the first rank-0 microbatch of that full-model comparison.
The original loss changes little despite large derivative changes because
different experts can produce similar aggregate loss with different gradients.

## Adopted optimization

`cublas_sm120_forward` retains the original expert shapes, order, BF16 storage,
FP32 compute, reduction setting, deterministic workspace and cuBLAS algorithm.
It acquires the Torch handle once and launches the expert GEMMs in a small C host
loop. This removes repeated framework/handle setup without padding the expert
matrices or changing their arithmetic. The existing Triton backward is reused.
This is host dispatch optimization, not a newly fused forward GPU GEMM.

On the pinned Linux CUDA13 runtime, the native host shim is compiled once from its content hash with the runtime C
compiler. It uses the existing CUDA 13 cuBLAS library and current Torch stream;
it creates no persistent CUDA handle. It validates boundaries and restores the
handle's math mode. The narrow implementation accepts contiguous BF16 operands
on SM120 only. Original and Triton implementations remain available for comparison.

Six captured first-layer projections are bitwise identical to the original.
Their contended microbenchmarks show 1.67–4.70× lower individual dispatch+GEMM
latency (12 interleaved repetitions). These are **not** full training or inference
speedups. An initial full encoder/decoder pass also has exact router/expert
outputs, action logits, probabilities and loss. Sixteen targeted checks cover
both BF16 reduction settings, first derivatives, empty experts, invalid boundaries
and explicit migration configuration checks. The later two-GPU gradient gate
from update109 measures zero loss and raw-gradient difference in every family.

The owner authorized an exclusive GPU0/1 validation window after diagnosis.
From the complete update109 state, full DDP/global64 loss and raw gradients have
zero measured difference in every family. Five continuous updates versus two plus
three in a fresh process have zero adapter/optimizer error, exact all-rank RNG,
and2,130 matching raw tensor byte arrays. Sample/loss/gradient/LR sequences also
match the original backend across all five updates.

The uncontended matched training measurement uses four warmed updates per backend:
median11.8788→7.4821seconds (1.5876×;37.0% shorter), peakallocated83.768GiB per GPU.
It includes input preparation, forward/backward, communication and optimizer;
checkpoint/validation/rollout overhead is excluded. It is a short measurement,
not a long-run estimate or an inference benchmark. Production resumes the actual
update109 checkpoint; disposable update114 weights are never reused. Rank32,
batch64, optimizer/RNG, the100k LR schedule and ten500-episode evaluations stay fixed.

The follow-up denoising probe also constructs CPU masters with the reference
backend, then selects the checkpoint's CUDA backend after placement. This avoids
rejecting CUDA-kernel metadata during CPU-only construction; execution arithmetic
remains on the authorized GPU.

To explicitly migrate a preserved backward-only checkpoint, use the new config
with `--resume /path/to/checkpoint-000109 --allow-kernel-migration`. The migration
allowlist permits only this backend change; changing the LR, batch or budget at
the same time is rejected. All later segments resume with the same new config.

Evidence and scripts: [report](../reports/training-throughput/sm120-forward-20261025/),
`scripts/diagnose_forward_numerics.py`, `scripts/verify_forward_batch.py`, and
`scripts/benchmark_forward_dispatch.py`. Private operands contain base-model
weights and are deliberately excluded from publication.

The upstream behavior is documented in [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)
and implemented in the [pinned cuBLAS source](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/cuda/CUDABlas.cpp).
