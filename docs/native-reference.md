# Native DecisionVLA reference

Implemented on 2026-10-25 in a separate runtime. CPU tests execute a real tiny
DiffusionGemma Transformer, native `generate()`, entropy-bound sampler and soft
self-conditioning. The 26B model, GPU memory/latency, real-data learning, DDP
resume and LIBERO outcomes are **not yet validated**. No new main training or
rollout was launched. Historical checkpoints and their source remain unchanged.

## Isolated runtime

Do not install these requirements over the pilot's Torch2.9.1/Transformers5.20.2
environment. The native code checks Torch2.14.0, Transformers5.11.0 and the hashes
of the native generation/modeling files against the inspected upstream revision.

```bash
uv venv --python 3.12 .venv-native
uv pip install --python .venv-native/bin/python -r requirements-native.txt pytest==9.0.2
PYTHONPATH=src .venv-native/bin/python -m pytest \
  tests/test_native.py tests/test_native_decisions.py tests/test_native_compare.py \
  tests/test_typed_decisions.py tests/test_decision_service.py -q
```

Use `PYTHONPATH=src` for this runtime. `pip install -e .` resolves the historical
pilot dependencies instead. All GPU commands below are explicit commands for a
future approved run; no launcher, scheduler or automatic training is installed.

## Native mechanism and adapter

`decision_canvas.py` compiles typed questions into a 256-token answer canvas by token ID.
The fixed robot schema has 56 slots in a flat `answers` array; the earlier nested
`actions` layout is retained only as a low-level reference fixture.
Its56 variable slots start with uniform full-vocabulary noise. Structure and EOS
remain fixed. The processor preserves the entire vocabulary at variable slots;
it reads the256 action labels (two endpoints for the gripper) only at the end.
`allowed_mass` and `allowed_log_mass` retain the unrestricted normalizer.
Probabilities come from native softcapped logits before sampler temperature.

`native_sampling.py` calls unmodified native `model.generate()`. Scoped overrides
only initialize the fixed template and disable adaptive stopping for exactly1/2/4
passes. The native loop owns categorical proposals, entropy acceptance, re-noising,
temperature indexing and processed-logit self-conditioning. Each observation gets
a new prefix cache. Calls are serialized; the selected device's RNG is seeded and
restored without touching another device's RNG. Optional tracing is separate from
performance measurement and can be large because it retains full-vocabulary tensors.

`native_adapters.py` installs fresh shared attention/MLP/expert rank32 LoRA. Frozen
vision and router weights remain unchanged. A single parametrized action-row delta
is used for input lookup, the full output projection and the native soft-embedding
expectation. The reference materializes one effective embedding matrix inside a
cached context: this prioritizes correctness and **is not a5090 memory optimization**.

The reference explicitly uses eager experts. Initial testing accidentally compared
the default grouped expert implementation to eager LoRA arithmetic and found small
floating-point differences. Comparing the same eager operations restored exact
zero-adapter equality; tolerances were not widened. Fused/grouped deployment needs
its own numerical and outcome comparison later.

`native_training.py` implements full-vocabulary corruption/CE, replaced-valid-slot
supervision including accidental target matches, per-observation normalization and
50% stop-gradient self-conditioning. Missing future positions get random tokens,
never copied target padding. Empty replacement draws contribute zero loss. A
zero conditioning tensor is not substituted for absent conditioning. The training
mixture is our80% uniform-rate /20% fully noisy adaptation, not an upstream recipe.
`configs/native_libero40_r32.json` preserves the40-task split and proposes B64/100k;
the full trainer, real-data gradient/capacity and all-rank resume gates remain pending.
The CPU resume check covers a tiny model on one CPU only.

## One checkpoint, three pass budgets

Only native checkpoints with matching architecture/config/payload hashes and a
positive trained step are accepted. Legacy46/2k/4.5k/6,681-step adapters are rejected.
The comparison cannot currently run on a trained native policy because none exists.

For latency, supply a JSON list of saved observations. Each has `image` (relative
to the manifest), `instruction`, and optional `target_actions` shaped8×7. Images
must be128×128 and already rotated into the audited policy convention.

```bash
PYTHONPATH=src .venv-native/bin/python -m decisionvla.native_compare \
  --adapter /absolute/path/to/native-checkpoint \
  --observations /absolute/path/to/observations.json \
  --output results/native-pass-latency
```

All budgets use the same checkpoint, input and seed. Each has warmups; measured
order rotates to reduce systematic order effects. Artifacts include input/source/
checkpoint hashes, generation configs, raw timing rows, all action probabilities,
allowed mass, differences versus4-pass and demonstration MAE when targets exist.
The report separates preprocessing, prefix, denoising+terminal readout and CPU return,
with p50/p95 and query Hz. Stage timings use CUDA events on GPU. Cold loading,
network, telemetry and simulator execution are excluded. Demonstration MAE is not
robot success; repeated timing cases are not independent quality observations.

Serve a native checkpoint in its runtime, then evaluate in the existing isolated
LIBERO simulator runtime:

```bash
PYTHONPATH=src .venv-native/bin/python -m decisionvla.native_serve \
  --adapter /absolute/path/to/native-checkpoint --socket /exchange/native.sock

# In the simulator runtime with the pinned original LIBERO checkout:
PYTHONPATH=src python -m decisionvla.native_evaluate \
  --socket /exchange/native.sock --output evaluation-output/native-pass-comparison
```

The frozen development protocol is40tasks×5initial states×3budgets=600episodes,
200 per budget, paired by initial state and seed. Initial states0–4 are reused
development cases, not untouched test data. Plus is excluded from selection.
Errors remain errors; missing or unmatched pairs abort aggregation. The report
contains task-stratified paired bootstrap95% intervals versus4-pass and a−5pp
screen. Degenerate bootstrap intervals cannot establish equivalence and leave the
screen unresolved. The runner never automatically changes the deployment policy.
Outputs must be new directories; failed attempts are preserved, not overwritten.

## Typed decisions

See [typed decisions](typed-decisions.md). The default policy conditions on the
fixed robot questions and uses the same schema for future training preparation.
`decide()` returns ordered typed answers; `predict_with_confidence(typed=True)`
also returns a `decisions` envelope from the same model call. The local socket
supports `op: decide` and a `DecisionClient.create()` wrapper. Custom questions
require explicit experimental opt-in and do not produce robot controls.
This adds CPU formatting/serialization cost; inspect `timing` for its scope.

`native_training.prepare_robot_example()` shares the exact inference prefix and
converts gripper endpoint bins into the predicate's true/false option order.
Do not use the old pilot's observation preparer for a native decision run.
The new architecture tag is `decisionvla.native-decisions.v1`; schema and canvas
hashes in the checkpoint config prevent accidental use of older layouts.

Reference denoising passes and independent noisy reads are different experiments.
The currently implemented comparison is1/2/4 sequential native passes. It does
not reproduce razorback16/openjev's adaptive independent-read averaging or its
vLLM backend. Neither typed output nor native sampling proves calibration or
real-time robotic inference. The old188ms RTX5090 result remains pilot-only.
