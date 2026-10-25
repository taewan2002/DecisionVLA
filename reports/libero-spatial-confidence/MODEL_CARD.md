# DecisionVLA LIBERO-Spatial rank-32 pilot

Experimental adapter for `google/diffusiongemma-26B-A4B-it`, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`. Use DecisionVLA's shared-adapter loader; this is not a generic PEFT checkpoint.

## Training

- Run: `libero-spatial-r32-all-v2`, checkpoint update 2,000, seed 42.
- Official LIBERO-Spatial demonstrations: 450 training / 50 validation episodes, split by demo number. Known-task imitation learning, not pretrained robot adaptation from OpenVLA.
- One 128×128 RGB agentview, rotated 180°, plus language instruction; no proprioception.
- Post-action alignment: observation t predicts actions t+1 through t+8, with padding masked from the objective.
- Effective batch 8, microbatch 1 / accumulation 8; learning rate 1e-4; rank 32 / alpha 64.
- Shared attention Q/K/V/O, dense MLP and per-expert fused gate/up + down LoRA; tied action-row deltas. 989,474,816 trainable parameters; original backbone, vision and routers frozen.
- Final training CE 1.5288857. This is not a robot success metric.

## Inference and confidence

The BF16 reference requires roughly 53 GiB of device memory with this implementation and does not fit one 32GB RTX 5090. It uses four denoising steps and produces 8×7 normalized LIBERO controls. The adapter is stored as FP32 LoRA/action deltas in four safetensors shards, with an index and configuration. Base weights and optimizer state are not distributed.

`predict_with_confidence()` returns the same actions, plus final-step conditional action probabilities and raw distance-aware/categorical confidence. No extra trained confidence module is included. The probabilities are conditional on one sampled denoising path; confidence is not calibrated task-success probability or a causal explanation. No confidence-based execution intervention is enabled.

## Evaluation — preserved 2k pilot

The 2k pilot completed 500 LIBERO-Spatial episodes with 14 successes (2.8%). Primary within-task failure AUROC was 0.4311, with 95% bootstrap interval [0.2872, 0.5897]. The primary score did not establish useful failure discrimination. Probability/action recomputation, 1,500 artifact SHA checks and 33 frozen-source SHA checks passed. See `report.json`, `episodes.csv` and `completion-audit.json` in this directory.

This card describes the preserved 2k pilot. The final experimental release is deferred until the authorized 50k continuation and all five 500-episode evaluations are complete and reviewed; its selected-step card will be written separately.

The real-checkpoint preflight verified exact action parity between the original and instrumented APIs. It is excluded from scored evaluation. No Object/Goal/Long-horizon results, trained NVFP4 conversion, actual RTX 5090 latency, or physical-robot evaluation are available.

## Scope and attribution

This is a personal research artifact for simulation experiments. It is not a validated physical-robot controller. Code is MIT; the base model and LIBERO assets retain their upstream terms, documented in `docs/third-party.md`. Credit the original DiffusionGemma and LIBERO authors. The release contains only adapter deltas, metadata and evaluation documentation; users obtain base weights separately.
