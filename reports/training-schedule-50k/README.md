# Authorized 50k continuation

Prepared on 2026-10-24. The preserved 10k run is training; the 50k controller is waiting for the existing 10k training, denoising probe and 500-episode audited evaluation. No 10k→50k updates or new scored rollouts have occurred yet.

The [public execution contract](../../docs/training-50k.md) and [fixed plan](../../configs/training_schedule_50k.json) define 20k/30k/40k/50k boundaries and 500 episodes per checkpoint. Checkpoint selection is task-macro success, earliest step on ties, after all five reports finish. All initial states are reused development conditions.

Preparation evidence:

- [Source hashes](source-sha256.json): 58 public files in the frozen follow-up source.
- [Deployment audit](deployment-audit.json): 38 original 10k source files unchanged and 13 policy/data/evaluation core files identical in the new snapshot.
- [Controller rehearsal](control-rehearsal.json): CPU-only mocked success/failure paths; not a GPU or rollout result.
- [Learning rate](learning-rate.json): numerical values on both sides of evaluation boundaries.
- All 37 CPU tests passed in the pinned training image, including actual checkpoint/optimizer/RNG serialization and the release allowlist. Ruff passed on changed public code. The real GPU segment/resume gate remains queued and is mandatory before main continuation.

The real run will add 40k updates to the preserved 10k policy, using resampled-only loss and one uninterrupted cosine schedule. Training pauses only at the declared boundaries for evaluation. It does not automatically stop based on intermediate success rates. Errors preserve their evidence and block subsequent training.

Publication remains on hold until the full 50k chain, evaluation audits and final documentation/review finish. This directory currently contains preparation evidence, not 50k performance claims.
