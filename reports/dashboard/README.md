# Inference observatory implementation check

The local dashboard exposes existing model outputs: camera image, instruction,
8 × 7 actions, component confidence, all 256-bin distributions (binary gripper),
normalized entropy, policy latency and query history. It does not control the robot.

The preview uses two actual 4500-step RTX5090 W4A16 eager episodes: the first
episode and first failed episode in task/init order. Their 37 observations contain
2,072 component distributions. This explicit UI example is not a representative
performance sample or an additional success cohort. Graph/eager output parity was
already established in the [inference experiment](../inference-nvfp4-5090-20261025/final/).

Checks completed:

- Every imported image, action, probability and recorded policy latency matches
  its original trace exactly. All confidence metrics agree within 2e-6.
- Twelve CPU tests passed across the dashboard and existing confidence tests,
  including unchanged prediction return/RNG, one policy call per observation,
  RPC timing preservation, rejected malformed distributions, bounded queue drops,
  empty/live HTTP updates, read-only routes and session overwrite protection.
- Ruff, JavaScript syntax and whitespace checks passed. The wheel contains all
  three HTML/CSS/JS assets.
- Browser inspection verified observation/action alignment, continuous and binary
  distributions, episode switching, next-query navigation and playback through
  all 28 queries of the failed episode. The preview remains explicitly labeled REPLAY.

No new GPU inference, training change, recurring automation, commit or push was
performed. Live recording was validated with CPU fixtures; a new live GPU session
and its telemetry overhead have not been measured. The preview's saved eager
latencies must not be relabeled as the separate 188ms standalone CUDA Graph result.

Raw sessions, source images/traces and the preview screenshot remain in ignored
`private/`. See [dashboard setup](../../docs/dashboard.md) for live integration,
replay import, timing boundaries and confidence interpretation.

## Evaluation view and continuous replay update

The evaluation view now includes all 56 distributions, per-worker episode totals,
actual outcomes and a first-query-confidence/outcome comparison. Optional rollout
and original/Plus benchmark observers preserve the model calls, RNG and actions.
Simulator errors remain separate from task failures. The preview still uses only
the same two selected episodes, not a representative benchmark sample.

The original videos are attached with their audited SHA values: 70 and 221 frames
at 20fps, including the initial frame. All 37 query identities were checked against
the original trace, and existing query JSON is unchanged. Video time drives query
selection; the exact policy-input image remains available separately. The video
can be replayed with recorded call waits or in simulation-only mode. The default
includes each measured query wait and nominal 20Hz controls: the success example
lasts 6.438379s (2.988379s of waits + 3.45s of actions). Per-control simulator
processing time is unavailable, so this is not an end-to-end speed measurement.
Live telemetry still displays query observations; continuous video is a replay feature.

The update's validation is recorded in
[evaluation-view-20261025.json](evaluation-view-20261025.json).
