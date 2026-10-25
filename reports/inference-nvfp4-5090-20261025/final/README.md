# RTX5090 NVFP4 development comparison

Frozen4500-step model,20 paired LIBERO-Spatial initial states per arm. Training was unchanged.

Original BF16: **12/20** successes.

| Candidate | Successes | Recovered / lost vs original | Continuous-action MAE |
|---|---:|---:|---:|
| w4a16 | 11/20 | 2 / 3 | 0.040572 |
| w4a4 | 9/20 | 3 / 6 | 0.043056 |

Exclusive GPU0 timings include image/text preprocessing, four denoising passes, eight actions and confidence conversion. Each input was warmed; compilation/capture times are recorded separately.

| Path | p50 / p95 (ms) | Peak allocated (GiB) |
|---|---:|---:|
| w4a4-graph | 212.12 / 214.03 | 20.527 |
| w4a4-eager | 246.66 / 248.07 | 20.467 |
| w4a16-eager | 315.21 / 316.12 | 18.095 |
| w4a16-graph | 188.43 / 190.35 | 18.815 |

The JSON report contains paired95% intervals and complete artifact hashes. This20-episode development comparison does not establish equivalence or superiority. Weight-only W4A16 and native activation-quantized W4A4 are reported separately. GPU0 mining was restored; other inference-host jobs retained their identities. No weights or data are included in this report.

The W4A16−BF16 paired success difference is −5 percentage points (95% interval [−35,+25]); W4A4−BF16 is −15 points ([−50,+20]). Graph execution reproduced all 863 candidate trajectory observations exactly, including confidence, with independent verification of 48,328 distributions. Independent standalone-array checks cover 120 predictions / 6,720 distributions and recompute reported latency aggregates. These repeated predictions are not extra success trials. The original local collector encountered an SSH error; recovery collected the completed remote evidence without rerunning inference.
