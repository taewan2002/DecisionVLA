# Configuration catalog

**No configuration currently launches the proposed native DecisionVLA path.**
The native architecture is specified in [architecture](../docs/architecture.md);
training is intentionally stopped. Existing JSON files preserve reproducible
historical contracts and must not be relabeled as native-sampler experiments.

| Group | Files / purpose |
|---|---|
| Initial pilots | `libero_spatial.json`, `libero_spatial_r32_all.json` |
| Spatial continuations | `libero_spatial_r32_corr_*`, `training_schedule_50k.json` |
| Batch64 / kernel experiments | `libero_spatial_r32_b64_*`, `training_schedule_b64_*` |
| Stopped forty-task run | `libero_40_r32_b64_100k.json`, `training_schedule_libero40_b64_100k.json` |
| Original / Plus catalog | `eval_libero40_plus_100k.json`; evaluator infrastructure, not a completed result |
| Rollout / confidence | `eval_libero_spatial*`, `confidence_analysis*`; read each exact checkpoint/protocol |
| Deployment | `deployment_b64_100k_w4a16.json`; historical final-model validation plan |

The [experiment ledger](../docs/experiments.md) identifies completed measurements,
stopped runs and unexecuted plans. Configuration presence is not evidence of execution.
