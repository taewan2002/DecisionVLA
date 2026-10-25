# Action probabilities and confidence

`DecisionVLA.predict_with_confidence()` observes the same final denoising readout as `predict()`. It returns the same action chunk for the same observation, instruction, steps and seed, plus an uncalibrated diagnostic distribution. It adds a final softmax and CPU statistics, with no additional model forward or random draw. It does not alter action execution, add a confidence head, or update the checkpoint.

```python
result = policy.predict_with_confidence(image, instruction, steps=4, seed=42)
actions = result["actions"]                     # NumPy [8, 7]
diagnostics = result["diagnostics"]
p = diagnostics["probabilities"]               # JSON-compatible [8, 7, 256]
score = diagnostics["metrics"]["ordered_confidence"]  # [8, 7]
```

The axes are timestep, control dimension, and bin. Continuous controls use `linspace(-1, 1, 256)`. The seventh control is the binary gripper: only bins 0 and 255 are allowed; other probabilities are exactly zero. `action_bins` is the per-position argmax. A bin probability is probability mass over a discrete allowed option, not a density at an exact physical value.

## Distribution and metrics

The schema is `decisionvla.action-confidence.v1`. `calibrated` is always `false`. The conditioning is `final_denoising_step_given_sampled_canvas`: it describes the final readout given the intermediate stochastic denoising path. It is not the marginal over all noise seeds, epistemic uncertainty, a joint action-trajectory distribution, or an estimated task-success probability.

Let `p` be the distribution at one position, `m = argmax p`, and `n` the number of allowed options (256 or 2).

| Field | Definition |
|---|---|
| `selected_probability` | `p[m]` |
| `choice_confidence` | `(p[m] - 1/n) / (1 - 1/n)`, clipped to [0, 1] |
| `normalized_entropy` | `-sum(p log p) / log(n)`; larger means more uncertain |
| `top2_margin` | Largest minus second-largest allowed probability |
| `mean`, `std` | Moments in normalized control units |
| `expected_absolute_deviation` | `sum(p[i] * abs(x[i] - x[m]))` |
| `ordered_confidence` | Continuous: `max(0, 1 - expected_absolute_deviation / mean(abs(x)))`; gripper: `choice_confidence` |
| `chunk_confidence` | Arithmetic mean of `ordered_confidence` over 8 × 7 positions |

The categorical normalization and ordered distance-from-mode construction follow the public [JEV confidence definitions](https://docs.typesafe.ai/confidence). The mixed continuous/gripper chunk average is this project's fixed diagnostic aggregation, not a JEV success estimator. Equal normalized-control weighting is a simple baseline, not physical importance weighting; a certain component can conceal an uncertain critical component. Full distributions and per-component metrics remain available for later diagnostics.

Adjacent modes get a higher ordered confidence than equally probable modes far apart. A high value therefore does not necessarily mean high exact-bin accuracy. Choice confidence is only a monotonic transformation of peak probability, so they are not independent evidence. No product across 56 correlated positions is used as chunk success probability. Entropy and margin are secondary diagnostics, not independently calibrated predictors.

## Failure association

The [frozen protocol](../configs/confidence_analysis.json) was recorded before the first real-policy rollout outcome. Each of 500 LIBERO-Spatial episodes contributes one first-query score and one final success/failure label. The primary predictor is `1 - chunk_confidence`; higher means more uncertain. The primary statistic is failure-positive within-task pair-weighted AUROC. A pair is eligible only when its two episodes are from the same task and have opposite outcomes; tied scores receive half credit. Tasks with no outcome variation contribute no pairs, and their counts are reported explicitly.

Secondary results include pooled AUROC, per-task AUROC, and pooled entropy / low-peak-probability AUROC. The report includes 2,000 within-task episode bootstrap resamples and 2,000 within-task label permutations with a fixed seed. Intervals are conditional on these ten tasks and one training seed, not uncertainty across new tasks or retraining. All-failure/all-success data cannot identify failure discrimination. If there are no eligible pairs, AUROC and its interval remain `null`.

Fixed confidence bins show empirical failure fractions without fitting a threshold. These are descriptive associations, not calibration of task-success probability. No success-probability ECE/Brier, operating threshold, or confidence-based intervention is fitted on these episodes. We use the first query to prevent episode length and later observations from leaking the final outcome into the predictor. A weak first-query association does not rule out useful phase-dependent signals; such analysis would be a separate, explicitly exploratory study.

Full probabilities are stored in compressed episode traces. The analyzer verifies their checksums and recomputes every metric, checks allowed support and normalization, matches argmax to decoded actions, and checks executed actions against the queued chunks before aggregation. See [evaluation instructions](../evaluation/README.md).

## What this establishes

This implementation makes policy uncertainty observable and auditable. Whether it predicts failure is an empirical question. It does not establish a causal explanation for an action, calibrated reliability under distribution shift, or a benefit from stopping/replanning. Those claims require additional labels, separate calibration/evaluation data, or controlled intervention comparisons.
