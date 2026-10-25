"""Uncalibrated diagnostics of the final conditional action-token distribution.

These are not task success probabilities or a joint distribution over trajectories.
Continuous controls use ordered bins; the gripper has two categorical endpoints.
"""
import numpy as np

SCHEMA = "decisionvla.action-confidence.v1"
METRICS = ("selected_probability", "choice_confidence", "normalized_entropy", "top2_margin",
           "ordered_confidence", "mean", "std", "expected_absolute_deviation")


def summarize(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (8, 7, 256) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Expected finite nonnegative [8, 7, 256] probabilities")
    if not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0) or np.any(p[:, 6, 1:-1] != 0):
        raise ValueError("Invalid normalization or forbidden gripper probability")
    ids = p.argmax(-1)
    selected = np.take_along_axis(p, ids[..., None], -1)[..., 0]
    n = np.full((8, 7), 256)
    n[:, 6] = 2
    entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=-1) / np.log(n)
    top = np.partition(p, -2, axis=-1)[..., -2:]
    values = np.linspace(-1, 1, 256)
    mean = np.sum(p * values, axis=-1)
    variance = np.sum(p * (values - mean[..., None]) ** 2, axis=-1)
    deviation = np.sum(p * np.abs(values - values[ids][..., None]), axis=-1)
    choice = np.clip((selected - 1 / n) / (1 - 1 / n), 0, 1)
    # JevScore-style distance from the mode, normalized by uniform MAD about
    # the support midpoint. For the binary gripper use JevChoice instead.
    ordered = np.clip(1 - deviation / np.mean(np.abs(values)), 0, 1)
    ordered[:, 6] = choice[:, 6]
    metrics = dict(zip(METRICS, [selected, choice, np.clip(entropy, 0, 1), top[..., 1] - top[..., 0],
                                ordered, mean, np.sqrt(variance), deviation]))
    return {"schema": SCHEMA, "calibrated": False,
            "conditioning": "final_denoising_step_given_sampled_canvas",
            "action_bins": ids.tolist(), "probabilities": p.astype(np.float32).tolist(),
            "metrics": {key: value.tolist() for key, value in metrics.items()},
            "chunk_confidence": float(ordered.mean())}


def validate_diagnostics(value, actions):
    from .codec import ActionCodec

    if value.get("schema") != SCHEMA or value.get("calibrated") is not False:
        raise ValueError("Unsupported confidence schema")
    computed = summarize(value["probabilities"])
    if value.get("conditioning") != computed["conditioning"]:
        raise ValueError("Unsupported distribution conditioning")
    if not np.array_equal(value["action_bins"], computed["action_bins"]):
        raise ValueError("Action bins are not distribution argmax")
    if not np.allclose(ActionCodec(256).decode(np.asarray(computed["action_bins"])), actions,
                       atol=1e-6, rtol=0):
        raise ValueError("Reported actions disagree with their probabilities")
    for key in METRICS:
        actual = np.asarray(value["metrics"][key])
        if actual.shape != (8, 7) or not np.allclose(actual, computed["metrics"][key], atol=2e-6, rtol=0):
            raise ValueError(f"Invalid confidence metric: {key}")
    if not np.isclose(value["chunk_confidence"], computed["chunk_confidence"], atol=2e-6, rtol=0):
        raise ValueError("Invalid chunk confidence")
    return value
