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
