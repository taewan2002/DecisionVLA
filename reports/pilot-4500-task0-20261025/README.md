# One-off4500-step task0 pilot

The owner requested a small interim rollout after the final-only100k plan. This is one exploratory pilot, not a return to automatic10k checkpoint selection. The fixed100k final model and final500-episode evaluation remain unchanged.

The latest complete checkpoint at inspection was4500. Spatial task0 and initial-state order0–9 were fixed before rollout. A timing-only preflight rule determines the sample count, at most10: `min(10, floor(1200 / (28 * second_warmed_preflight_p95_policy_seconds *1.2 +5)))`. Abort if this permits no episode. The scored episode count is fixed before any scored rollout outcome is observed. Preflight is excluded.

Training on GPUs0/1 and serving on GPU3 remain running. A separate process uses GPU0 with a6GiB allocator limit; about49.29GiB of immutable parameters remain on CPU and whole modules are copied to CUDA for arithmetic. It uses the preserved production source and original prediction API. Current-checkpoint full-resident versus streamed parity is not newly measured; this exploratory placement limitation is recorded in the runtime contract. The real action/confidence/repeat API preflight is required before scored rollout. Latency under streaming and concurrent training is not a standalone inference benchmark.

The worker stops only its own containers if GPU0 free memory falls below2GiB, production training stops, or sampled recent training updates show sustained slowdown above30%. The original600W training power limits, optimizer/RNG, LR and checkpoint state are not changed. The outer pilot deadline is one hour.

Every episode retains its video, full probability trace, action chunk and simulator-state trace. The normal CPU analyzer verifies artifact hashes, probabilities, confidence metrics and action/execution alignment. AUROC remains undefined when there is only one outcome class; a tiny single-task sample cannot estimate suite-wide performance or calibrated reliability.

The first attempt stopped before scored rollout because its cold44.14-second first query exceeded the time budget. That evidence is preserved. The second attempt adds a warmed preflight, preserves the same memory/slowdown/time limits, and measured15.659 seconds per chunk. The timing-only rule fixed two scored episodes (initial states0 and1) before results. Both API preflights passed; the first cold preflight is excluded along with the warm preflight.

## Completed result

Both scored initial states succeeded: init0 in102 controls (13 policy queries) and init1 in72 controls (9 queries), with a220-control maximum. This is two examples of one known task, not an estimate of suite-wide success. The recorded videos are kept local-only under `private/`.

First-query raw confidence was0.900891 and0.946629; trajectory means were0.921451 and0.934163. Both outcomes are successes, so failure-discrimination AUROC and its interval are undefined, not perfect. These scores are not calibrated success probabilities.

The standard artifact/probability/metric/action audit passed for all six episode files. Independent saved-state restoration reproduced both success labels and confirmed sustained bilateral contact and >3cm lifting in each case; peak object height gains were9.63cm and10.46cm. State-restoration error was at most2.22e-16. The post-hoc diagnostic now emits null failure-only means on all-success inputs, avoiding invalid NaN JSON; geometry arithmetic is unchanged.

The first-cold and warmed preflight actions, complete probabilities, confidence and saved simulator states were bitwise equal; both were excluded from scores. New full-resident versus streamed parity for4500 is still not measured.

Main training remained the same process and advanced from4651 to4712 during v2 and its audit. Sampled median update time increased from7.573s baseline to8.810s during the pilot (about16.3%); GPU0 had at least4780MiB free. The pilot containers were removed after completion. Main training, serving and other monitored GPU jobs retained their IDs/PIDs/start times. The original100k-only final500 evaluation remains connected; no periodic evaluation or new automation was registered.

`rollout-config.json`, `analysis-protocol.json`, `timing-gate.json` and `plan.json` record the fixed scope. `report.json`, `episodes.csv`, `summary.json`, `geometry.json`, `verified-artifacts.json` and `completion-audit.json` record the results and checks. Local SHA and CSV recomputation passed. No commit, push or release has been made.
