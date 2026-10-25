# Final checkpoint deployment and publication

> **Historical procedure — superseded on2026-10-25.** Training was checkpoint-stopped; the forty-task run ended at46 and its automatic final evaluation did not start. These commands describe the earlier action-only implementation. See the [current DecisionVLA design](architecture.md) and [development plan](development.md); do not restart old controllers.

**Current scope, 2026-10-25:** fresh training covers all forty original
Spatial/Object/Goal/Long tasks. The previous Spatial run stopped at6,681 and is
preserved. The [current controller](training-libero40.md) connects100k training
to2,000 original LIBERO episodes and10,030 LIBERO Plus episodes with BF16
reference policies. Plus is evaluation-only. Final optimized-model checks below
remain publication gates; they have not been run for these new weights.

The deployment candidate is **NVFP4 expert W4A16 + fused kernels + prefix/decoder
CUDA Graph**. The implementation has been tested on the frozen 4,500-step pilot.
Its 188.43 ms p50 RTX5090 result and 11/20 successes do not validate the eventual
100k checkpoint. The paired BF16 pilot scored 12/20; the difference is −5
percentage points with descriptive paired 95% interval [−35, +25].
[Completed pilot evidence](../reports/inference-nvfp4-5090-20261025/final/).

The older machine-readable [Spatial deployment contract](../configs/deployment_b64_100k_w4a16.json)
is superseded for training/evaluation scope. The current training and BF16
evaluation contract is `configs/training_schedule_libero40_b64_100k.json`.
The deployment sequence for the eventual final checkpoint is:

1. Finish the fresh forty-task100,000 updates with its fixed LR and global
   batch64, preserving optimizer/RNG across segments. Keep the fixed final
   checkpoint and original hashes. Complete the connected LIBERO/Plus audits.
2. Use an isolated inference copy. Merge expert LoRA once and export W4A16;
   retain the validated attention/dense LoRA and action-embedding paths. This
   weight-only method computes scales from weights and needs no activation
   calibration dataset. Merging additional modules would require separate
   numerical verification and is not silently included in this recipe.
3. Repeat export/reload, deterministic prediction, graph parity and action/
   probability/confidence checks on the final weights. Compare against their
   own BF16 reference, not the earlier pilot's predictions or success rate.
4. Reuse the controller's final BF16 **2,000-episode** original LIBERO result
   after its audits. Evaluate W4A16 on the **same2,000 initial states**, preserving
   task/init ordering, suite-specific220/280/300/520-control horizons, four
   denoising passes and eight-action execution.
   Report recovered/lost episodes, task scores, paired intervals and confidence/
   failure association. These are4,000 trajectories over2,000 paired development
   conditions, not an unseen test or a new checkpoint-selection sweep.
   Label the10,030-episode Plus result as BF16; claiming optimized W4A16 Plus
   performance requires a corresponding evaluation of that exported model.
5. Measure the final resident policy on an otherwise idle RTX5090: whole-query
   p50/p95 and memory, with compilation/capture separated. Speed is per eight
   actions plus confidence conversion; it excludes robot execution/network time.
6. Update both READMEs, the model card, figure/caption and audited reports with
   the final measurements. Review the code and publication file list, then make
   the owner's authorized single combined commit/push. Publish the adapter's
   four shards, index, config, model card and SHA manifest as an experimental
   GitHub Release. The base, merged/quantized base copies, dataset, optimizer/RNG
   and private operations are excluded. Provide the conversion recipe as code.

Other robot embodiments are deferred. This release sequence covers LIBERO and
LIBERO Plus only.

Quality regressions or inconclusive comparisons remain visible. A fast runtime
or an interval containing zero is not evidence of equivalent policy quality.
No final success threshold or beneficial effect is invented after seeing results.
Raw confidence remains conditional token concentration rather than calibrated
task-success probability or causal explanation.

The existing controller already connects training, the final BF16 rollout and
its audits. The remaining deployment steps are recorded here as follow-up work;
this change creates no recurring automation or unattended GitHub publishing job.
Benchmark helpers that assume the 4,500-step/20-episode pilot must be generalized
and checked before the final100k/four-suite run; the pilot is not a completed
end-to-end release validation.

Historical metadata clarification: the preserved Spatial confidence protocol
contains an old sentence about ten candidate checkpoints. Its later `finalonly`
schedule fixed100k alone. The replacement forty-task plan independently fixes
100k before results. Preserve earlier protocol files and hashes as history.
