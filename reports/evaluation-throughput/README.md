# Evaluation throughput investigation

This is a runtime investigation of the BF16 RTX PRO 6000 reference evaluation. It is not an RTX 5090 measurement, a quantized deployment result, or a change to the policy.

## Serial measurement

The last 50 completed episodes ending at task 4, initial state 32 took 1,564.924 seconds. Saved policy timers account for 704.810 seconds; end-to-end policy requests account for 723.187 seconds. The remaining 841.736 seconds include simulator resets, physics, rendering and artifact handling.

- Median episode completion interval: 31.676 seconds.
- Median total policy time per episode: 14.329 seconds.
- Mean policy time per query: 511.473 ms (8 controls, 4 denoising passes).
- The serial execution alternates CPU simulation and GPU inference.

`timing-before.json` records the input artifact hashes and individual timings. These are consecutive original completion timestamps, not timestamps from downloaded copies.

## Rejected shortcut and reproducibility check

Standalone CPU replay of task 4, initial state 30 did not reproduce the original settled object state, even with the original rendering thread setting. Setting `LP_NUM_THREADS=1` produced exactly the same replay states and image hashes as the default replay, but was slower in this small comparison (24.023 versus 19.094 seconds). Video writing and trace compression together took less than 0.11 seconds. These incomplete-history replays are diagnostics, not valid replacement rollouts or a controlled rendering benchmark.

Replaying initial state 0 first reproduced that first episode exactly, but then jumping to initial state 30 still did not reproduce the original later episode. The precise internal reset-history mechanism has not been isolated. Consequently, the optimization does **not** shard arbitrary initial states.

The live policy preflight instead replayed task 4 initial states 0 and 1 in their original order, with a fresh environment and the same policy server. Both complete episodes matched **all recorded observations, simulator states, actions, probabilities and confidence metrics exactly**; only latency fields were excluded. See `shared-policy-parity.json`. This validates that tested task prefix; it is not an exhaustive comparison of every task.

## Adopted execution change

Two additional CPU workers process task 8 and task 9. Each starts a fresh environment and evaluates all 50 initial states in their original order. They share the existing single GPU0 policy server; no additional GPU model or GPU allocation is created. Seeds, camera, initial states, 220-control horizon, 8-control chunks, 4 denoising passes, checkpoint and original source/analysis protocol remain unchanged.

Each worker stages a whole task privately. All 50 episodes must pass artifact SHA, probability/metric and executed-action checks before publication. The serial evaluator must still be at least five earlier episodes away, and no destination for that task may exist. Otherwise the staged task is left unpublished and the original evaluator finishes normally. The original evaluator loads complete published tasks and performs the original 500-episode aggregation and confidence audit. Partial tasks are never used to resume a new environment midway through its history.

The 10k evaluation has the same assistance mechanism queued. It must first pass a new complete two-episode parity check against its own checkpoint and task-0 prefix. A failed assistance check leaves the serial evaluation responsible for completion.

The main evaluation's 33 frozen source files were verified unchanged before launch. The assistance wrapper has a separate recorded SHA. The optimization changes scheduling, so concurrent request latency must not be presented as standalone inference latency. The first 10 completion events span 139.756 seconds between the first and last event, or 15.528 seconds per aggregate completion interval (approximately 2.04× the earlier serial throughput). This is a short, unpaired observation across different tasks, not a final speedup estimate. Only 100 episodes are assisted, so the whole-run speedup will be smaller. The model's per-query compute has not been optimized by this change. See `timing-early.json`.

No intermediate commit, push, release, new training run or other GPU change was made by this investigation.

## Completion

Both assistance workers completed and published all 50 episodes, each in about 2,268 seconds. The original 500-episode evaluator completed at 23:02 KST and its audit verified all 1,500 artifact files. No duplicate task/initial pairs were present. See `task-08-completion.json`, `task-09-completion.json`, and the [full evaluation audit](../libero-spatial-confidence/completion-audit.json).
