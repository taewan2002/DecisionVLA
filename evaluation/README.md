# LIBERO evaluation

> Existing policies and results use the historical action-only sampler. Native DecisionVLA evaluation is planned but has not started. See [development](../docs/development.md) for current scope.

The current forty-task model uses `configs/eval_libero40_plus_100k.json` and
`python -m decisionvla.benchmark`: **2,000 original LIBERO episodes plus10,030
LIBERO Plus episodes**, automatically after100k training. The pinned Plus runtime
uses `Dockerfile.plus`; external assets are downloaded separately and verified.
See the [current protocol and evidence contract](../docs/training-libero40.md).

The sections below document the preserved Spatial-only pilot evaluator.

The completed rank-32 run used effective batch 8 (microbatch 1, accumulation 8). Increasing accumulation would increase work per optimizer update; it is not a throughput optimization. Larger true microbatches require a separate memory/throughput check and do not change this in-flight training run.

## Fixed protocol

[`configs/eval_libero_spatial.json`](../configs/eval_libero_spatial.json) specifies the evaluation before observing rollout results:

- Final rank-32 checkpoint at update 2,000, without checkpoint selection on these rollouts.
- LIBERO-Spatial task order 0, all ten tasks, official initial states 0–49: **500 episodes**.
- Environment seed 0, ten settling controls `[0, 0, 0, 0, 0, 0, -1]`, then up to 220 policy controls at the simulator's 20 Hz control frequency.
- Raw `agentview` at 128 × 128, matching the training HDF5 resolution. Rotate 180 degrees exactly once. No center crop or extra JPEG conversion.
- Four denoising passes, predict eight controls, execute all eight sequentially and replan from the next observation. The final chunk is truncated to respect the 220-control horizon.
- Native LIBERO gripper sign is preserved. This model did not use OpenVLA's gripper sign normalization, so there is no additional sign inversion.
- Policy query seed = `42 + task_id * 100000 + initial_state_id * 1000 + query_index`.
- Success comes from LIBERO's explicit task predicate, not a generic `done` flag. Errors abort the run and are reported separately; they are not silently removed from the denominator.

The settling, default initial states and Spatial horizon follow the [OpenVLA evaluation implementation](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py). Camera processing and action chunking follow this model's own training contract. This is not an exact OpenVLA protocol reproduction. The official initial-state assets are benchmark evaluation inputs; they are **not claimed to be novel or disjoint from all training demonstrations**.

## Separate runtimes

The policy runs in the existing Python 3.12 / Torch 2.9.1 training image on GPU0. LIBERO uses a separate Python 3.10 / Torch CPU 2.5.1 / NumPy 1.26.4 simulator image with OSMesa software rendering. A Unix-domain socket transfers lossless PNG observations and action arrays. Neither service opens a network port, and the simulator needs no GPU.

```bash
docker build -t decisionvla-policy .
docker build -f evaluation/Dockerfile.simulator -t decisionvla-simulator .
mkdir -p evaluation-output

# Terminal 1: use an absolute existing checkpoint path for CHECKPOINT.
docker run --rm --gpus device=0 --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/models/base:/model:ro" -v "$CHECKPOINT:/adapter:ro" \
  -v "$PWD/evaluation-output:/exchange" \
  decisionvla-policy python -m decisionvla.serve \
  --model /model --adapter /adapter --socket /exchange/policy.sock --expected-step 2000

# Terminal 2: after policy_ready appears, verify the real policy first.
docker run --rm --network none --cpus 4 --memory 8g --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/evaluation-output:/exchange" decisionvla-simulator \
  -m decisionvla.rollout --socket /exchange/policy.sock \
  --config /workspace/configs/eval_libero_spatial_confidence.json \
  --output /exchange/preflight --preflight

docker run --rm --network none --cpus 4 --memory 8g --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/evaluation-output:/exchange" decisionvla-simulator \
  -m decisionvla.rollout --socket /exchange/policy.sock \
  --config /workspace/configs/eval_libero_spatial_confidence.json --output /exchange/results
```

The preflight uses one initial state, checks repeated prediction for the same observation/seed, and executes eight controls. It is excluded from the 500-episode score and never updates weights. CPU transport/simulator smoke tests use `--stub` on the server and `--smoke` on the evaluator; their output is explicitly marked as a smoke test and cannot be used as policy performance.

Each completed episode is atomically stored with actions, simulator states, generated chunks, query seeds, input-image hashes, timings, a replay video and artifact checksums. Re-running the exact evaluation contract reuses verified completed episodes. A partial episode directory is retained on errors for diagnosis. The final summary contains per-task and overall success rates and BF16 reference inference timings, not NVFP4 or RTX 5090 timings.

The original2k pilot handoff is preserved as completed history. The current batch64 run performs no intermediate rollout: after normal100k completion, it verifies the fixed final adapter, runs the denoising audit and real-policy preflight, then evaluates500 episodes on GPU0. The same final protocol and analysis configuration are retained. Training failure or timeout does not trigger evaluation of a partial checkpoint. The main rollout has a13-hour limit; unrelated GPU workloads and serving remain unchanged.

## Confidence and failure association

The policy socket is private (`0600`). Normally run both containers and the controller with the same host UID/GID. If an existing checkpoint is root-readable only, keep its permissions unchanged, run the policy container as root and pass `--socket-owner "$(id -u):$(id -g)"` to `decisionvla.serve`; continue running the simulator as the host UID. Only the newly bound socket changes owner. An earlier root-owned socket prevented host readiness checking before any real rollout; that failed attempt is preserved separately.

The confidence configuration retains all action generation and execution settings. It records the final conditional distribution `[queries, 8, 7, 256]` plus metrics in each trace. The real-checkpoint preflight compares the instrumented output with the legacy action-only API using identical input and seed. The final readout consumes no extra random numbers or forward passes.

The predeclared [analysis protocol](../configs/confidence_analysis.json) uses the **first query only** for each episode. The primary endpoint is within-task pair-weighted AUROC for failure, using `1 - chunk_confidence`. This avoids treating successive queries as independent observations and avoids outcome-dependent episode length in the predictor. Fixed-task bootstrap intervals and within-task label permutations accompany pooled and per-task secondary metrics. All-success/all-failure cases remain undefined; no fictitious AUROC is assigned. This is a development diagnostic on known tasks, one checkpoint and one training seed.

After all 500 episodes complete:

```bash
docker run --rm --network none --cpus 4 --memory 8g --user "$(id -u):$(id -g)" \
  -v "$PWD/evaluation-output:/exchange" decisionvla-simulator \
  -m decisionvla.analyze_confidence --results /exchange/results \
  --protocol /workspace/configs/confidence_analysis.json --output /exchange/analysis
```

The analyzer verifies artifact hashes, probability normalization, gripper constraints, metric recomputation, action argmax/decoding and executed chunk alignment before reporting association. Raw confidence is not a success probability; success-probability ECE/Brier and adaptive control are outside this evaluation. The full per-episode audit remains on the experiment host; the report and episode-level table can be published without distributing the base model or dataset.
