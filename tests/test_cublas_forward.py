"""Exact output and derivative gates against the accepted backward-only backend."""
import os

import pytest
import torch

from decisionvla.grouped_gemm import grouped_linear

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="requires SM120")


@pytest.mark.parametrize("k,n", [(2816, 1408), (2816, 32), (32, 1408),
                                (704, 2816), (704, 32), (32, 2816), (35, 19)])
@pytest.mark.parametrize("reduce", [True, False])
@gpu_only
def test_exact_forward_backward(k, n, reduce):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduce
    try:
        torch.manual_seed(k+n)
        counts = [0, 1, 7, 23, 64, 127, 0]
        offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        x = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(len(counts), n, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(sum(counts), n, device="cuda", dtype=torch.bfloat16)
        if k == 35:
            # Torch grouped_mm rejects unaligned dimensions; independent mm accepts them.
            start, pieces = 0, []
            for e, count in enumerate(counts):
                pieces.append(torch.nn.functional.linear(x[start:start+count], w[e]))
                start += count
            expected = torch.cat(pieces)
        else:
            expected = grouped_linear(x, w, offsets, "triton_sm120_backward")
        actual = grouped_linear(x, w, offsets, "cublas_sm120_forward")
        assert torch.equal(expected, actual)
        if k != 35:
            for a, b in zip(torch.autograd.grad(expected, (x, w), dy),
                            torch.autograd.grad(actual, (x, w), dy), strict=True):
                assert torch.equal(a, b)
        assert torch.equal(actual, grouped_linear(x, w, offsets, "cublas_sm120_forward"))
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is reduce
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old


@gpu_only
def test_empty_groups_and_invalid_offsets():
    from decisionvla._cublas_forward import linear_forward

    w = torch.zeros(3, 32, 32, device="cuda", dtype=torch.bfloat16)
    offsets = torch.zeros(3, device="cuda", dtype=torch.int32)
    x = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
    assert linear_forward(x, w, offsets).shape == (0, 32)
    x = torch.zeros(4, 32, device="cuda", dtype=torch.bfloat16)
    for values in ([1, 0, 4], [1, 4, 8], [0, 1, 3]):
        offsets = torch.tensor(values, device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="dispatch failed"):
            linear_forward(x, w, offsets)


def test_forward_migration_requires_explicit_flag_and_unchanged_training_config():
    from decisionvla.continuation import validate_resume_config

    old = {"steps": 100000, "expert_kernel": "triton_sm120_backward"}
    with pytest.raises(ValueError, match="identical config"):
        validate_resume_config(old, {**old, "expert_kernel": "cublas_sm120_forward"})
    new = {**old, "expert_kernel": "cublas_sm120_forward"}
    assert validate_resume_config(old, new, allow_kernel_migration=True) == {
        "expert_kernel": {"before": "triton_sm120_backward", "after": "cublas_sm120_forward"}}
    with pytest.raises(ValueError, match="identical config"):
        validate_resume_config(old, {**new, "steps": 200000}, allow_kernel_migration=True)
