# Fresh forty-task training: startup evidence

At2026-10-25 15:53KST, `libero40-r32-b64-100k-v1` had completed its first
three production updates on GPU0/1. This is a startup snapshot, not a current
progress counter or a measured robot-success result.

- Previous Spatial run checkpoint-stopped at6,681. Its eight payload files
  passed SHA verification; adapter, optimizer/RNG, logs and frozen source remain.
- Fresh rank32 LoRA/action embeddings and optimizer/RNG from the pinned base;
  no Spatial or disposable smoke weights reused. Global batch64,100k updates.
- Forty original tasks,2,000 demonstrations:1,800 train /200 validation.
  All336,575 observation-to-next-state alignments passed with zero joint error.
- Uniform task then uniform frame sampling. The first192 sampled observations
  included Spatial43, Object53, Goal51 and Long45. Every task is present in the
  dataset manifest; finite samples are not expected to have identical counts.
- Initial validation covered200 fixed observations,50 per suite. Denoising
  CE6.25069 and action MAE0.53993 are open-loop initialization measurements.
- First three training losses:6.32096,6.21405,6.18451. All four trainable families
  had finite, nonzero gradients. Updates2/3 took7.61/7.77seconds; this short
  startup measurement is not a sustained throughput estimate. Peak allocated
  memory across ranks was83.768GiB.
- Disposable continuous3 versus2+1 restart: adapter/optimizer differences0,
  all-rank RNG exact. Fresh production has the same initial adapter hash.
- Real evaluation preflight:4 original-suite cases plus28 Plus cases covering
  seven perturbation categories per suite. All32 cases completed eight controls;
  action API / confidence API / repeated prediction parity passed. Independent
  audit verified1,792 action distributions, derived metrics and action traces.
  These short preflight episodes are excluded from quality scores.
- Catalog audit covers all40 original tasks and10,030 pinned Plus variants.
  Training source133 files and evaluator source36 files passed frozen SHA checks.
  Protected training/serving identities were unchanged; GPU1 mining stays off.

At startup, the planned automatic chain was training through100k, then BF16 original LIBERO2,000
and Plus10,030 episodes, followed by raw artifact audits, report.json and
episodes.csv. Plus contributes no training examples. Final W4A16/RTX5090
deployment validation remains a separate publication gate. No Git commit,
push, Release or recurring automation was created by this transition.

Files: `startup.json`, `data-audit.json`, `preserved-spatial-checkpoint.json`,
`resume-gate.json`, `evaluation-preflight-audit.json`, `catalog-summary.json`,
`provenance.json`. `sha256.json` checks the collected files. The source snapshots,
old full checkpoint, raw simulator traces and machine controllers remain private.
See [the current training/evaluation contract](../../docs/training-libero40.md).

**Later owner-requested stop:** this run saved checkpoint46 and exited before any final evaluation. See the [stop audit](../openjev-redesign-20261025/stop-audit.json). The startup records and their hashes remain unchanged.
