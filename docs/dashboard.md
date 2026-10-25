# Inference observatory

A local, read-only dashboard for existing DecisionVLA predictions. It displays the
actual camera input, 8 × 7 action chunk, per-component confidence, the full allowed-bin
distribution, normalized entropy, policy latency and query history. Selecting a cell
changes the inspected timestep/control. The gripper shows its two allowed values.

The interface is packaged with Python; no Node build, CDN or cloud service is required.
It binds to `127.0.0.1` and has no robot-control or model-invocation endpoint. For a remote
machine use SSH port forwarding, for example `ssh -L 8765:127.0.0.1:8765 your-host`.

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

Replay is always labeled **REPLAY**. Its 0.75s/query playback speed is independent of
the saved inference latency; it is not a live GPU benchmark. Recorded episode success
is a historical label, never an estimate derived from confidence. Images and all
56 distributions remain linked to the same query. Query JSON can be downloaded.

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
