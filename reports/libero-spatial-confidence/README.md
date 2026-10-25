# Action-confidence / failure association

Status: all 500 episodes and the frozen probability/action audit completed on 2026-10-24 at 23:02 KST. Success: **14/500 (2.8%)**. Primary within-task failure AUROC: **0.4311**, conditional 95% bootstrap interval **[0.2872, 0.5897]**. The interval includes 0.5; this checkpoint does not establish useful failure discrimination from the primary raw-confidence score. Only 14 successes occurred across five tasks; single-outcome task AUROCs remain undefined.

- Checkpoint: `libero-spatial-r32-all-v2/checkpoint-002000`, unchanged rank-32 adapter.
- Evaluation run: `libero-spatial-r32-confidence-v3`, started 2026-10-24 19:17:59 KST.
- One RTX PRO 6000 Blackwell, GPU0, restored 500W limit; BF16 reference policy. These are not RTX 5090 or NVFP4 measurements.
- Policy: one 128×128 agentview image + instruction, four denoising steps, eight executed controls per query, no extra policy forward for diagnostics.
- Suite: LIBERO-Spatial, ten tasks × official initial states 0–49, up to 220 policy controls after ten settling controls.
- Protocol: [`confidence_analysis.json`](../../configs/confidence_analysis.json), SHA256 `c64c01539e1f578be8e48285d9c92708addc10261e161060ee9e91663a2e21d2`, fixed before real-policy rollout labels.
- Primary: first-query uncertainty versus final failure, within-task pair-weighted AUROC. Secondary pooled/per-task scores, 2,000 fixed-task bootstrap samples and within-task label permutations. No calibration or control intervention.

## Verification before rollout

23 CPU tests passed in the policy image, including sampler output/intermediate canvas/RNG parity, confidence endpoints, nearby versus distant ambiguity, invalid distributions, transport bounds, rollout chunk recording, AUC ties/undefined cases, adapters and packaging. Ruff passed. A CPU stub exercised all ten tasks and 140 total settling/policy controls, recording full confidence traces. This smoke test is not a learned-policy result.

A non-root simulator exposed Git's ownership check for the pinned root-installed LIBERO source; a command-local `safe.directory` exception fixes the revision read without changing global Git settings. The first evaluation attempt failed at host socket readiness; the next failed reading root-only checkpoint shards. Neither produced a policy rollout. The final setup retains the checkpoint permissions, runs the policy as root and transfers ownership only of its newly bound `0600` socket to the host client. Host readiness for that ownership arrangement passed.

The real-checkpoint preflight passed: action-only and instrumented predictions were exactly equal for the same input/seed, the repeat prediction matched, and eight controls executed successfully through the simulator. It is excluded from the score. The server worker runs the final audit automatically after all 500 episodes and preserves partial/failed attempts.

## Completed outputs

The final `report.json`, episode-level `episodes.csv`, `contract.json`, `summary.json`, `verified-artifacts.json` and `completion-audit.json` are collected here. All 1,500 episode artifact hashes and 33 original source hashes were rechecked; the primary AUROC was independently recomputed from the CSV over 624 eligible within-task failure/success pairs. Downloaded report/CSV/contract/artifact-index hashes match the remote audit. The primary score and 2k checkpoint were fixed before this evaluation; later training changes were informed by this development diagnostic. All-failure or all-success task AUROCs remain undefined. The data are a development diagnostic for known tasks, one checkpoint and one training seed; they do not establish causal explanation, calibrated success probability, or safety benefits.

Publication remains held until the corrected 10k continuation and its subsequent evaluation are verified. The continuation started at 23:03 KST after real-gradient and numerical resume checks. It extends the original 2k checkpoint with 8k resampled-only updates; it is not a fresh 10k run. No intermediate commit or push is part of this run.

Evidence: [preflight summary](preflight-summary.json), [parity result](preflight-result.json), [checkpoint metadata](policy-metadata.json), [frozen evaluation source](source-sha256.json), and [CPU stub summary](cpu-smoke-summary.json). The first preflight timing includes cold-start setup and parity verification; it is not a steady-state inference benchmark.

An owner-requested [post-hoc training and task-0 diagnosis](diagnostic-task0/README.md) is available separately: 1/50 successes on the first task, limited training exposure, mostly absent sustained target contact, and high raw confidence on failures. It does not replace or alter the fixed full-run analysis. All findings remain scoped to this early cohort.

Initial runtime audit: the first **3 episodes / 84 policy queries** passed artifact hashes, distribution/metric recomputation, and action consistency checks. Warm reference policy p50/p95 was **0.510 / 0.531 seconds per chunk** in this small early sample. It includes observation processing, model inference and diagnostic extraction, excludes simulator/transport, and is not a final benchmark. The first three episodes took 102.46 seconds including simulation; early full-run extrapolation is approximately 4–5 hours, with task-dependent variation. [Initial audit](initial-trace-audit.json).

## Runtime scheduling

Two CPU workers completed and audited tasks 8 and 9 (100 episodes), then published whole tasks into the original serial evaluator. The final result contains 500 unique task/initial-state pairs and no duplicate episodes. See [throughput evidence](../evaluation-throughput/README.md). Aggregate runtime includes concurrent denoising diagnostics and shared-server assistance, so its policy/query latency percentiles are not standalone inference benchmarks.
