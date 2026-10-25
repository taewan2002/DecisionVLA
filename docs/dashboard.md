# LIBERO evaluation observatory

A local, read-only dashboard for existing DecisionVLA predictions. It displays the
actual camera input, 8 × 7 action chunk, per-component confidence, the full allowed-bin
distribution, normalized entropy, policy latency and query history. Selecting a cell
changes the inspected timestep/control. The gripper shows its two allowed values.

The interface is packaged with Python; no Node build, CDN or cloud service is required.
It binds to `127.0.0.1` and has no robot-control or model-invocation endpoint. For a remote
machine use SSH port forwarding, for example `ssh -L 8765:127.0.0.1:8765 your-host`.

## Follow LIBERO evaluation

The evaluation board shows completed episodes, actual simulator successes/failures,
errors separately, per-suite counts and policy-call p50/p95 over the available
query history. Counts are for **one worker**, not all shards combined. A saved
selection of episodes is explicitly labeled **SELECTED REPLAY**, with no benchmark
success-rate claim.

- **56 distributions** switches the action matrix to all eight timesteps × seven
  controls. Each miniature uses every allowed bin and scales to its own peak;
  select a cell for absolute probabilities. Color continues to encode confidence.
- **Inspect least certain** selects the minimum per-component confidence in the
  current chunk; it is an inspection aid, not a stop/replan threshold.
- **Confidence meets reality** plots one first-query score per completed episode
  against its actual success/failure label. It does not use a later query if the
  first score is missing. This is descriptive; it is not a calibration or AUROC
  analysis, and it does not establish failure prediction.
- The episode list and plots show the latest 200 episodes (20 buttons), while
  evaluation totals use all episodes. Query navigation keeps the latest 600
  recorded queries. Full frames and episode records remain on disk; an episode
  without query images in this window is not clickable.

Both `decisionvla.rollout` and the original/Plus `decisionvla.benchmark` worker
accept `--dashboard-session /absolute/path/to/new-session`. Add the flag to an
already approved evaluation command, keeping its checkpoint and protocol intact.
Use a **different session directory for every worker/attempt**, outside the
evaluation artifact directory. The observer requires `record_confidence` in the
existing protocol; it never enables it silently.

```bash
# Append to an existing evaluation command:
# --dashboard-session /exchange/telemetry/worker-0

# Once the worker creates that session, run the viewer on the same host:
python -m decisionvla.dashboard \
  --session /exchange/telemetry/worker-0 --mode live --port 8765
```

For Docker, mount the telemetry directory into both runtimes. Serve the viewer on
the host or use a separate loopback-bound viewer; the evaluator's network isolation
does not need to change. Additional workers use different sessions and viewer ports.

The observer receives the existing RGB input, action chunk, final probabilities,
query seed and simulator control step. It issues **no model calls or random draws**.
Optional telemetry adds CPU snapshot/serialization work, and episode-boundary
status writes add disk overhead. These costs are outside the displayed policy
latency. Telemetry errors are logged and counted without changing policy execution;
simulator/evaluation errors remain errors, not failed task labels. Completed reused
episodes contribute verified outcomes/first scores without fabricated images.

Live evaluation summaries are observations, not replacements for the final trace
hash/probability/action audit. The native denoising architecture remains untrained;
the bundled local preview uses the preserved 4,500-step action-only pilot records.

## Observe live inference

Wrap the already loaded policy, including a resident W4A16 + fused + CUDA Graph policy:

```python
from decisionvla.dashboard.telemetry import ObservedPolicy, Recorder

# policy is your existing loaded model; no extra model is loaded by the dashboard.
with Recorder("results/live-session", {
    "source": "live_inference",
    "device": "RTX 5090",
    "precision": "NVFP4 W4A16",
    "checkpoint_step": 100000,  # Set to the actual loaded checkpoint.
    "execution": "Expert W4A16 · fused · prefix + decoder CUDA Graph",
}) as recorder:
    observed = ObservedPolicy(policy, recorder)
    # Use this call inside your existing observation loop.
    result = observed.predict_with_confidence(image, instruction, steps=4, seed=42)
    actions = result["actions"]
```

The metadata above is an example, not evidence that a 100k model has been validated.
Use a new directory for every run. The wrapper also supports `PolicyClient`: its
`(actions, policy_seconds, diagnostics)` tuple and server timing are preserved.
For direct policies, timing covers the existing `predict_with_confidence()` call up
to its CPU result, excluding telemetry submission/writes. Logging has CPU overhead;
benchmark the deployment with telemetry separately before making timing claims.

While the process records predictions, start the viewer on the same machine:

```bash
python -m decisionvla.dashboard --session results/live-session --mode live
```

Open `http://127.0.0.1:8765`. The viewer polls every 250ms. Live mode follows new
queries automatically; selecting an earlier query freezes the view until **Follow
latest** is pressed. Waiting, disconnected and recorder-closed states are explicit.
No additional forward pass, sampling draw, threshold decision or action modification
is performed. CPU serialization runs on a bounded queue: slow logging drops telemetry
rather than blocking the policy on disk I/O. Drop/error counts remain visible. Closing
the recorder flushes its queue. It is intended for one serial prediction producer.

## Replay actual recorded inference

```bash
python -m decisionvla.dashboard --session results/recorded-session --mode replay
```

Replay is always labeled **REPLAY**. With an attached audited rollout, the default
**Recorded inference waits** mode pauses the original video at each query boundary
for that query's saved `query_seconds` (including the RPC wait). It then executes
the chunk at nominal 20Hz control speed. The first output remains hidden until
its recorded wait ends; later waits explicitly retain the previous output. A
separate playback timeline includes both waits and actions. Pause, resume, restart
and seek use the same timing plan. No probability interpolation is performed.

This is **recorded call waits + nominal control duration**, not a live GPU run or
an exact reconstruction of simulator wall time. Per-control simulator processing
timestamps were not recorded. The success example has 2.988379s of measured query
waits and 69/20 = 3.45s of controls, giving 6.438379s of modeled playback. The existing
188ms standalone Graph measurement is not substituted into these eager traces.

**Simulation only** plays the original 20fps video with waits omitted. Frame zero
is the initial post-settle observation; frame N follows control N. In that mode,
panels follow the latest query at or before the current step (0.4s intervals for
eight-step chunks). Neither video fps nor the timed replay is a new throughput
measurement. The policy latency tile always uses the original `policy_seconds`.

Query navigation seeks the video, and **Inspect policy input** switches to the
exact RGB query input (the encoded video itself is lossy). Sessions without an
attached video retain explicit 0.75s/query snapshot playback. Live mode currently
follows query images, not a continuous per-control video stream.

Recorded episode success is a historical label, never an estimate derived from
confidence. Images and all 56 distributions remain linked to the same query.
Query JSON can be downloaded.

Existing LIBERO traces can be imported with the matching, SHA-verified query images
from `benchmarks/replay_policy_observations.py`:

```bash
python -m decisionvla.dashboard.import_trace \
  --observations results/query-images --results results/rollout \
  --metadata results/session-metadata.json --output results/recorded-session \
  --episodes task-00-init-00 task-03-init-01
```

The metadata JSON supplies device, precision, checkpoint step and the actual execution
path. Import checks contract/trace/image hashes and recomputes confidence from saved
probabilities. It runs neither a policy nor a simulator. A missing or inconsistent
artifact stops the import; an incomplete destination is retained for diagnosis.

To attach videos after importing a complete episode, use the matching rollout
audit (`artifacts_sha256`) and install `ffprobe` from FFmpeg on the import host:

```bash
python -m decisionvla.dashboard.video \
  --session results/recorded-session --results results/rollout \
  --audit results/rollout-audit.json --episodes task-00-init-00 task-03-init-01
```

This verifies the session contract, audited MP4/result/trace hashes, each query's
seed/image/control-step identity, finite call/model timings, and decoded video frame count and rate. It copies
the MP4 bytes without re-encoding and leaves the query records unchanged. The
read-only server supports byte-range seeking and serves only manifest-listed files
inside the session directory. Playback requires neither FFmpeg nor a GPU.

## Interpretation and data

Confidence summarizes concentration around the selected action. Continuous controls
use ordered confidence; the binary gripper uses choice confidence. The color scale
has no action/stop threshold. High confidence is **not** high task-success probability,
and the viewer does not provide a causal explanation. The [confidence contract](confidence.md)
defines the metrics and their limits.

Only post-denoising outputs are displayed. Per-pass traces, live VRAM monitoring,
confidence-based replanning and calibrated failure prediction are not implemented here.
The completed 5090 standalone graph benchmark remains separate from the viewer's
recorded per-query timings. Our initial preview uses two selected 4500-step W4A16
eager episodes (one success, one failure, 37 queries), whose graph outputs were
previously verified equal. This is an interface example, not a new success-rate study.

Keep sessions under ignored `results/` or `reports/**/private/`. They contain images,
instructions and model outputs; neither raw datasets nor weights are bundled with
the public UI. The UI and integration code can be released with the final project.
