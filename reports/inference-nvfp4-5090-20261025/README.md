# RTX 5090 NVFP4 deployment experiment — 2026-10-25

Status: the 4,500-step deployment pilot is completed and audited. Training continues unchanged on its original two GPUs.
This experiment uses a frozen **4,500-step** adapter, not the eventual 100k release model.
No intermediate commit, push, or model release is authorized by this experiment.

## Completed comparison

Original BF16 scored **12/20**. The same initial states produced **11/20** with
W4A16 (two recovered, three lost) and **9/20** with W4A4 (three recovered, six lost).
The paired success-rate differences and descriptive hierarchical 95% intervals
are −5 points [−35, +25] and −15 points [−50, +20], respectively. This small
development pilot does not establish quality equivalence or superiority.

On an otherwise idle RTX5090, W4A16 + fused + prefix/decoder Graph measured
**188.43 / 190.35 ms p50/p95**, with 18.815 GiB peak allocated memory. W4A4 + Graph
measured **212.12 / 214.03 ms**, with 20.527 GiB peak allocated. The measurements
include four denoising passes, eight actions and confidence conversion. GPU0
mining was restored; other host jobs retained their identities.

[Final reports, contracts and hashes](final/) include paired rollout audits and
an independent check of 120 saved predictions / 6,720 distributions and timing
aggregates. The local collector initially failed on an SSH connection; recovery
collected already-completed remote results without new policy calls or rollouts.
Both the collection failure and original staging were preserved privately.

## Fixed comparison

- Same ten previously recorded held-out RGB observations and seeds for action differences.
- LIBERO-Spatial tasks 0–9, initial states 0–1: **20 episodes per viable arm**.
- Original 220-control horizon, 10 settling steps, four denoising passes, eight actions per query, 128×128 front RGB, and full confidence diagnostics.
- Original BF16 execution uses Torch 2.9.1 with CPU streaming because the full BF16 model does not fit this GPU. Candidate experts are fully resident. Other jobs and the training runtime are unchanged.
- Expert-W4A16: NVFP4 E2M1 weights, E4M3 block16 scales, per-expert FP32 global scale; BF16 activations. This path uses fused BF16 computation with packed FP4 weight decoding, **not FP4 activation Tensor Cores**.
- Expert-W4A4: native SM120 FP4 Tensor Core fused MoE, calibrated on 32 training observations. Zero-padding 704→768 intermediate channels is applied to an inference copy only. Closed-loop evaluation runs only after numerical and whole-policy repeat checks pass.
- Attention, dense MLPs, vision and router remain BF16; nonexpert LoRA remains in its original execution path. Expert LoRA is merged in an isolated copy before quantization.

## Current verified evidence

The stable W4A16 path reproduces full actions and probabilities exactly across repeated fixed-input calls and across two fresh processes. Independent route-packing checks and a real-expert 30-repeat check pass. All 150 expert calls in three full-policy repeats agree exactly.

The independent fixed-input audit is in `fixed-input-audit-w4a16.json`: normalized continuous-action MAE **0.0405719**, maximum difference **0.784314**, gripper agreement **80/80**, and mean probability total variation **0.251278**. These differences include runtime, LoRA merge, quantization and kernel arithmetic. They do not isolate quantization error or establish policy equivalence.

The W4A4 audit reports continuous-action MAE **0.0430556**, maximum difference **0.737255**, gripper agreement **80/80**, and mean probability total variation **0.277995** against the same BF16 reference. The maximum differences and distribution shifts are material; average error alone does not establish preserved behavior.

| Path | Contended p50 / p95 | Peak allocated | Completed eager LIBERO pilot |
|---|---:|---:|---:|
| NVFP4 expert W4A16, prefix/decoder graphs | 201.56 / 206.02 ms | 18.815 GiB | 11/20 |
| Native NVFP4 expert W4A4, prefix/decoder graphs | 224.88 / 228.03 ms | 20.527 GiB | 9/20 |

These earlier **contended exploratory timings** shared GPU0 with the original evaluation. The final standalone measurements above supersede them for deployment latency. Each query returns eight actions after four denoising passes; latency is not one independent action latency.

Both graph paths reproduce actions and full probabilities exactly after export/reload on ten fixed observations and two measured repeats each. The independent audit checks 3,360 distributions. The 20-episode success counts above come from eager fused execution. Every recorded query was subsequently reconstructed with exact simulator states and image hashes: W4A16 409 queries, W4A4 454 queries. Graph actions, full probabilities and all confidence metrics matched the eager traces exactly on all 863 queries. Independent trace/hash, probability-support/normalization and action-decode audits passed for 48,328 distributions. This validates graph parity on the scored trajectories; replay is not an additional success cohort.

The complete W4A4 versus W4A16 comparison has four recovered and six lost episodes: −10 percentage points, paired hierarchical 95% interval [−45, +30] points. All initial states, images, checkpoints, runtimes and 120 artifact SHA checks agree. The BF16 comparator and both candidate-versus-BF16 audits also completed; their final paired counts and intervals are reported above.

## Correctness findings

The initial fused W4A16 implementation was nondeterministic: atomic route packing and atomic top-k accumulation changed rounding order. Tiny first-layer differences propagated through routing and discrete denoising. That implementation is rejected. The accepted route packer fixes ordering by expert and original routed-pair index and uses a fixed-order output reduction.

FlashInfer 0.7.2.post1's W4A16 implementation required a local GeGLU-tanh patch. It is confined to the inference image. The production training kernel, optimizer, RNG and checkpoint are unchanged. CUDA Graph captures the prefix language model and decoder, with sampling RNG outside the graph. Vision preprocessing stays eager, and prefix KV tensors are copied once per observation. A GPU cache scalar must be initialized before capture; the earlier CPU-to-GPU scalar copy attempt failed and was preserved.

For W4A4, an incorrect reciprocal activation-scale convention produced zero outputs in an early microtest and was rejected. The corrected actual-scale convention passes the declared microtest tolerance. Its dynamic kernel failed repeatability; only the static backend proceeds to whole-policy validation. All failed attempts remain in private execution evidence.

The W4A4 export preserves the original noncontiguous scale-factor strides on reload; treating these tensors as ordinary contiguous matrices is incorrect. Removing the masked padding queries (canvas 256→56 or 64) failed exact action/probability parity and was rejected. The accepted implementations retain the original 256-token canvas.

## Interpretation and final-model work

The complete paired success comparison, artifact/probability/action audit, exclusive-GPU timings and restoration checks passed their execution/audit checks. A small reused development cohort cannot establish equivalence or superiority. Raw confidence is conditional token concentration, not task success probability or causal explanation. This is post-training inference quantization, not NVFP4-trained weights. The final100k model still requires its own [deployment and publication sequence](../../docs/final-deployment.md).

Base weights, converted weights, calibration images, raw traces/videos and private machine operations are excluded from Git. Public reports will contain measurements, protocol, source/asset hashes and audit summaries. The final 100k/500-episode release gates remain unchanged.
