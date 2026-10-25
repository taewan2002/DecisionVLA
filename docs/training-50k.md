# The 50,000-update development run

> **Historical procedure — superseded on2026-10-25.** Training was checkpoint-stopped; the forty-task run ended at46 and its automatic final evaluation did not start. These commands describe the earlier action-only implementation. See the [current DecisionVLA design](architecture.md) and [development plan](development.md); do not restart old controllers.

Historical plan, superseded on2026-10-25 before the50k continuation started. The
small-batch parent was stopped at update2,534; checkpoint2,500 and its raw history
remain preserved. Use the [fresh batch64/100k contract](training-b64-100k.md) for
the current execution. The text below records the earlier approved plan.

This is a continuation of one rank-32 policy, not five independent training runs. The running 10k training process, its source snapshot and its already-connected 500-episode evaluation remain unchanged. A separate controller waits for that complete evaluation and its audits before acquiring the same single GPU.

## Fixed contract

| Item | Contract |
|---|---|
| Total optimizer updates | 50,000, including the original 2,000 |
| Objective history | 0–2k: all valid positions; 2k–50k: resampled valid positions, including chance matches |
| Rank / alpha / effective batch | 32 / 64 / 8, unchanged |
| Split / observation / action convention | Existing pinned 450/50 demonstration split, agentview + language, action offset +1 |
| Checkpoint / validation | Every 500 updates; 100 fixed held-out observations |
| Rollout checkpoints | 10k, 20k, 30k, 40k, 50k |
| Episodes at each checkpoint | Spatial 10 tasks × 50 initial states = 500 |
| Total rollout count | 2,500, including the existing 10k evaluation; repeated conditions, not 2,500 independent initial states |
| Selection | Highest task-macro success rate; earliest checkpoint on ties |
| Execution | One GPU; training and rollout alternate |
| Failure handling | Preserve evidence and stop the chain; never advance past a failed/incomplete audit |
| Performance stopping | No automatic performance-based early stopping |

The machine-readable contract is [training_schedule_50k.json](../configs/training_schedule_50k.json). The five success rates will be reported, including unsuccessful checkpoints. Confidence metrics do not choose the released checkpoint.

All 500 official initial states have already been used in the 2k pilot. This is a **development checkpoint comparison**, not an independent test or an unbiased selected-model estimate. One training seed cannot establish robustness across training seeds. The 2k comparison also changes the objective and training budget together.

## Learning rate and process boundaries

The existing 2k→10k LR schedule is preserved. At 10k, adapter weights, AdamW state, sampler RNG, noise RNG and PyTorch CPU/CUDA RNG move into a new run with [libero_spatial_r32_corr_50k.json](../configs/libero_spatial_r32_corr_50k.json).

The new schedule warms from 5e-6 to 5e-5 over 500 updates, then follows one cosine decay to 5e-6 at 50k. It does **not** restart at 20k, 30k or 40k.

| Update | Learning rate |
|---:|---:|
| 10,001 | 5.09e-6 |
| 10,500 | 5.00e-5 |
| 20,000 | 4.38773e-5 |
| 30,000 | 2.79473e-5 |
| 40,000 | 1.17491e-5 |
| 50,000 | 5.00e-6 |

`--stop-at-step` is a process boundary, not a change to the training config. The process saves a full checkpoint and reports `awaiting_evaluation`. `--resume` requires the identical config and verifies the checkpoint's hashes before loading optimizer/RNG state. The final boundary reports `completed`.

```bash
# Only after the preserved 10k evaluation and audits finish:
CUDA_VISIBLE_DEVICES=0 python -m decisionvla.train \
  --model models/base --data data --output runs/libero-spatial-r32-corr-50k-v1 \
  --config configs/libero_spatial_r32_corr_50k.json \
  --extend-from runs/libero-spatial-r32-corr-10k-v1/checkpoint-010000 \
  --stop-at-step 20000

# Evaluate/audit checkpoint-020000, then continue the SAME schedule:
CUDA_VISIBLE_DEVICES=0 python -m decisionvla.train \
  --model models/base --data data --output runs/libero-spatial-r32-corr-50k-v1 \
  --config configs/libero_spatial_r32_corr_50k.json \
  --resume runs/libero-spatial-r32-corr-50k-v1/checkpoint-020000 \
  --stop-at-step 30000
```

Repeat the resume operation from 30k→40k and 40k→50k, with a complete evaluation between segments. The wall-clock training limit is 30 hours **per process segment**, excluding evaluation. This does not change the optimizer-update budget.

Before the new main run, the controller uses separate disposable runs to compare three continuous updates against two updates plus a fresh-process resume for the third. It verifies adapter and optimizer tensors within absolute tolerance 1e-5 and RNG state exactly. These checks use the declared 40k LR schedule, not a compressed three-update schedule. The GPU check is queued; its pass is required before main continuation. CPU serialization/optimizer/RNG tests already pass. GPU bitwise identity is not promised.

## Evaluation and diagnostics

At each new checkpoint, run the same 20-observation × three-seed denoising probe, including deliberately wrong-canvas controls, and independently recompute its 280 saved distributions. Then run the [LIBERO evaluator](../evaluation/README.md) with the corresponding `eval_libero_spatial_confidence_20k.json` (or 30k/40k/50k) and `--expected-step` for that checkpoint's policy server.

Each evaluation has its own output directory, checkpoint identity, frozen source manifest and preflight. The fixed horizon is 220 controls, four denoising passes and eight executed controls per query. Probability/action API parity is checked before scored episodes. Two CPU helpers may assist whole tasks 8 and 9 through the existing policy server, only after that checkpoint's two-episode trace-parity gate. Initial states always run in order within a task. Do not restart a partly completed task by skipping previous initial states: environment reset history affected the observed simulation state.

Run `decisionvla.analyze_confidence` using the matching `confidence_analysis_20k.json` (or 30k/40k/50k). It verifies raw probabilities, executed actions, metrics and artifact hashes before calculating first-query within-task failure AUROC and its 95% interval. Undefined AUROC remains null. No threshold or calibration is fitted, and confidence does not change robot execution.

The final selection requires all five completed reports and valid per-task counts. `decisionvla.schedule.write_selection` produces the complete learning curve and selected step. Evaluation outcomes remain development measurements. Timings recorded during CPU assistance or diagnostics are marked as contended, not standalone latency.

## Public artifacts

Keep source/config revisions, dataset/base revisions, objective and LR changes, all five reports/CSV files, contracts, artifact hashes, denoising audits and a compact learning curve. Preserve full checkpoints locally; do not delete earlier evidence to present only a favorable result. Approximately one terabyte of additional local storage is budgeted for the 80 new 500-step checkpoints; storage is checked before each segment.

After all evaluations and review, publish one combined code/documentation/Figure update and the selected experimental adapter. The release contains exactly four adapter shards, their index, adapter config, finalized model card and SHA manifest. The packaging tool excludes training metadata, optimizer/RNG state, base weights, datasets and private host operations. If 10k wins, its adapter is selected from the preserved parent run. The model card must identify the selected step and the full training/selection history.

The current figure depicts implemented inference and the raw-confidence branch. It does not imply confidence-based control, a trained NVFP4 export, or measured RTX 5090 deployment.
