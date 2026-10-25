# DecisionVLA development plan

**2026-10-25: typed decision API and native reference. Publication before training.**
The owner requested native denoising/sampling, then a public repository refresh
before another training launch, then authorized the separate native implementation
and1/2/4-pass comparison tools. The previous forty-task run stopped at46 with
adapter, optimizer and all-rank RNG preserved; the earlier Spatial run remains
at6,681. No new rollout or GPU optimization run was started for this redesign.
The old controller's incomplete10k-boundary assertion is expected and must not
trigger recovery. [Verified stop](../reports/openjev-redesign-20261025/stop-audit.json).

## Implementation sequence

| Stage | Work | Required evidence before advancing |
|---|---|---|
| 1. Reference | Isolated pinned native runtime and typed canvas compiler | Same tokenizer/template; upstream text-case and robot-canvas traces |
| 2. Policy | Full-vocabulary denoising, native sampler/self-conditioning, terminal readout | Same seed/RNG, logits, proposals, entropy masks, re-noising, conditioning and actions against the native oracle |
| 3. Training | Full-vocabulary CE and corruption; tied action rows in all paths | Finite gradients, loss-mask audit, small real-data learning check and exact all-rank resume |
| 4. Capacity | Two96GB GPUs; choose microbatch and accumulation | Measured peak memory and steady update time at effective batch64 |
| 5. Main run | Fresh rank32, original40 tasks, planned100k updates | Sealed source/config/data and recorded training/validation curves |
| 6. Evaluation | Original LIBERO, then evaluation-only Plus | Complete action/probability traces, expected episode keys, independent aggregates and hashes |
| 7. Deployment | Expert W4A16, fused kernels, then CUDA Graph | Native BF16 reference comparisons, paired robot outcomes and actual5090 measurements |

Implemented reference modules: `native_policy.py` wraps native generation;
`typed_decisions.py` declares question semantics and typed readouts;
`decision_canvas.py` binds those questions to fixed syntax and variable slots;
`native_training.py` implements the explicit VLA objective;
`native_sampling.py` records optional sampler and confidence trajectories.
`native_compare.py` and `native_evaluate.py` compare1/2/4 passes; commands and the
remaining GPU/real-data gates are in [native reference](native-reference.md).
CPU checks cover tiny native models, not the26B GPU model. Preserve the old code
as the reproducible pilot instead of changing its meaning in place.

The reference path must pass before porting faster execution. For identical
runtime/operations, expect exact stochastic traces with identical RNG. For changed
precision/kernels, predeclare numerical tolerances, report every changed action
and compare paired outcomes; do not loosen tolerances after seeing a failure.
An optimization that fails is excluded while the valid reference remains available.

## Learning checks and budget

Keep the existing audited40-task data protocol. No Plus observation or outcome
is used to train or select a model. Use a tiny training subset only to establish
that full-noise targets can be learned; report it as an overfit/debug check.
Use fixed held-out demonstration observations for denoising diagnosis, not as
closed-loop success measurements. Save per-suite validation and raw uncertainty
alongside CE; compare pass1→2→4 action error, confidence, allowed mass and copying.

The proposed main budget is **rank32 / effective batch64 /100k updates**. Start
with microbatch1 and accumulation32 per rank on two GPUs, then measure whether
microbatch2 or4 fits. Preserve the batch64 denominator across ranks and variable
valid-slot counts. The previous microbatch4 capacity is not established for the
larger full-vocabulary/self-conditioning training graph.

Proposed LR:1,000-update warmup from5e-6 to5e-5, cosine to5e-6 at100k, no boundary
reset. Checkpoint and validate every500; retain10k boundaries and the latest two
recovery checkpoints. Verify continuous3 updates versus2+1 from a complete
checkpoint, including optimizer, every-rank RNG, sample IDs and LR. The fresh
main adapter must not inherit disposable smoke updates.

100k is a budget, not a convergence result. No reliable completion estimate is
available before the new graph's update time is measured. Report training time,
validation/checkpoint overhead and rollout time separately; do not reuse the old
sampler's throughput estimate.

## Quality–latency comparison

Hold the model, observations, initial states, image resolution, action horizon,
execution cadence and base revision fixed. First compare native1/2/4-pass policies
on the same development cases. Four native passes are the reference, not four
passes of the old action-only model. A comparison against an old trained model
changes both training and inference; it cannot isolate the sampler's causal effect.

A first bounded pilot can use40 tasks ×5 declared development initial states
(200 episodes per candidate), paired by initial state and policy seed. Report
paired success differences and95% intervals; a proposed deployment margin is
−5 percentage points versus the four-pass native reference. Choose the fastest
candidate only if the lower confidence bound exceeds that margin; otherwise keep
four passes or report inconclusive evidence. This is a development screen, not
an assertion that200 episodes ensure enough precision. Report all candidates.
Freeze the choice before final original/Plus evaluation; no repeated tuning on
Plus. Existing reused initial states are not an untouched test set.

The planned final coverage remains40×50=2,000 original episodes and10,030 official
Plus variants, with no Plus fine-tuning. Existing evaluator infrastructure and
catalog audits can be reused; policy/config hashes and denoising protocol must
be regenerated for the new method. Historical preflights do not validate it.

Measure5090 batch1 p50/p95 including image preprocessing, prefix, denoising,
readout and CPU return; separate cold compile, telemetry, network and simulation.
Report query Hz separately from control frequency. Full-vocabulary projection
and soft-embedding self-conditioning cost more per pass than the old256-row
head. A256×262,144 FP32 logits tensor alone is256MiB; several such tensors and
activation buffers may be live. **Neither188ms nor18.815GiB is a forecast for the
new method.** One/two-pass quality retention is the speed hypothesis to test.

Only after native parity consider full-vocabulary fused reductions, stable expert
packing and static-shape Graph capture. Final-step label-only projection is an
optional optimization, not a drop-in intermediate projection: intermediate
entropy and self-conditioning require full-vocabulary information, and reporting
allowed mass still requires its normalizer. Preserve RNG and generation semantics.

## Uncertainty checks and public delivery

Compare within-task failure AUROC with95% intervals, using a fixed first-query
score and declared grouping. Keep all-success/all-failure groups undefined.
Separate concentration, allowed mass, across-seed disagreement and actual failure
labels. Any fitted calibration model needs a separate development split and
calibration evaluation; no confidence threshold will silently change execution.

The repository can publish the existing implementation, curated evidence and this
proposal now. A later experimental model release includes adapter shards, index,
config, model card and hashes only, after evaluation/deployment gates. No base
weights, datasets, optimizer/RNG or private operations. Publication of this plan
does not start training, register recurring schedules, or release a model.
