import pytest

from decisionvla.native_compare import paired_success, percentiles


def paired_rows():
    return [{"passes": p, "suite": "libero_spatial", "task_id": t, "initial_state_id": i,
             "initial_state_sha256": f"{t}-{i}", "policy_seed": 42,
             "checkpoint_sha256": {"adapter": "same"}, "success": i > 1}
            for p in (1, 2, 4) for t in (0, 1) for i in range(5)]


def test_paired_success_and_identical_outcomes():
    report = paired_success(paired_rows(), bootstrap=100)
    assert report["episodes_per_pass"] == 10
    assert report["conditions"]["1"]["paired_ci95"] == [0, 0]
    assert report["conditions"]["4"]["success_rate"] == .6


@pytest.mark.parametrize("damage", ["missing", "duplicate", "checkpoint", "initial", "error"])
def test_pairing_fail_closed(damage):
    rows = paired_rows()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "checkpoint":
        rows[0]["checkpoint_sha256"] = {"adapter": "different"}
    elif damage == "initial":
        rows[0]["initial_state_sha256"] = "different"
    else:
        rows[0]["error"] = "simulator crash"
    with pytest.raises(ValueError):
        paired_success(rows, bootstrap=100)


def test_latency_quantiles():
    report = percentiles([.1, .2, .3])
    assert report["p50_seconds"] == .2 and report["query_hz_at_p50"] == 5
    with pytest.raises(ValueError):
        percentiles([float("nan")])
