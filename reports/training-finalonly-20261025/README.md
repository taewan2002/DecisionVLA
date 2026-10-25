# Fixed final100k evaluation

The owner chose to keep GPU3 serving available and remove intermediate closed-loop evaluations. This supersedes the previous ten-checkpoint selection plan; the final model is fixed at100k before observing its rollout results.

- Train on GPU0/1 with the unchanged rank32/global64 objective, kernel, optimizer and original100k LR schedule.
- Keep checkpoint saving and the same100-observation open-loop validation every500 updates, including all10k checkpoints.
- Run the ten process segments consecutively without intermediate denoising probes or LIBERO rollouts. Segmentation retains the original72-hour process budget; it causes only process reload overhead.
- After normal100k completion, run the denoising probe/audit, then one500-episode LIBERO-Spatial evaluation and confidence/action/probability/SHA audits. Preserve the original final protocol and development-state caveats.
- No performance-based checkpoint search, new recurring automation or intermediate publication. Release the fixed100k adapter only after the final evaluation, audits, documentation and review.

## State preservation and verification

The previous production run stopped after update219 and saved all eight checkpoint payload files: four adapter shards, index, config, training metadata and optimizer/all-rank RNG state. The manifest SHA is `32957e228756caab46b2d4c317d6ac865983fccae62d580ff107f23467442f3a`. The old run/source/logs remain preserved; its controller's early-boundary assertion is intentional stop history.

All numerical training source and configuration match the accepted runtime. The only changed existing source file is the CPU-only final-report selection helper; it enforces a fixed final model and writes `final-evaluation.csv` instead of a one-point learning curve. The sealed runtime has108 files. The previously accepted five-update continuous versus2+3 exact parameter/optimizer/RNG gate applies to the unchanged training path; no new numerical-equivalence experiment is claimed.

The staged runtime passed57 CPU tests, with11 CUDA-only tests skipped, and Ruff. Mocked orchestration passed all ten training segments followed by exactly one final probe/evaluation, plus independent training-failure and evaluation-failure cleanup paths. Checkpoint root-only tensor permissions were preserved; a prelaunch host read was denied, then integrity checking was performed in a CPU-only container with a read-only checkpoint mount. No training update failed in that check.

`transition-verification.json` records checkpoint integrity and unchanged protected container identities. `control-rehearsal.json` records mocked control paths, not real GPU training or rollout. `source-comparison.json` and `source-sha256.json` record source provenance. `startup.json` confirms the real resumed execution reached222/100k. Its initial adapter hash matches parent219, all observed LRs match the original schedule, and the first update220 has finite loss3.738667/gradient norm5.199653. The next two updates have median7.480 seconds; this is a short startup check, not a new sustained throughput benchmark. The serving and RetoVLA container IDs, PIDs and start times are unchanged.

## Interpretation and time estimate

Removing nine intermediate500-episode rollouts avoids about33.3 hours if the previous3.705-hour evaluation timing transfers. It does not change the6.4 million observation presentations or establish a better policy. Closed-loop quality will remain unknown until100k; demonstration validation is not a substitute for rollout success.

At about7.5 seconds/update, training plus periodic checkpoint/validation and one final evaluation is estimated at roughly nine days from2026-10-25. This is an estimate, not a measured end-to-end completion time. The reused500 official initial states are development conditions, not an independent test. Raw confidence is not calibrated task success probability or causal explanation.
