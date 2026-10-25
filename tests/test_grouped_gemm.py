"""Numerical coverage for ragged expert GEMMs and their first derivatives."""
import pytest
import torch

from decisionvla.grouped_gemm import check_expert_kernel, grouped_linear


def test_kernel_selection_rejects_unknown_or_unsupported_device():
    check_expert_kernel("torch_grouped_mm", "cpu")
    with pytest.raises(ValueError, match="Unknown expert"):
        check_expert_kernel("misspelled", "cpu")
    with pytest.raises(ValueError, match="SM120"):
        check_expert_kernel("triton_sm120", "cpu")


def _sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _sm120(), reason="requires SM120")
@pytest.mark.parametrize("counts,k,n,strided", [
    ([0, 1, 0, 33, 7, 0], 35, 19, False),
    ([1, 31, 32, 33, 65], 64, 128, True),
    ([0, 0, 0], 32, 32, False),
    ([0, 127, 0, 0], 32, 128, False),
    ([64] * 128, 2816, 32, False),
    ([64] * 128, 32, 1408, False),
    ([64] * 128, 704, 32, False),
    ([64] * 128, 32, 2816, False),
])
def test_ragged_forward_backward_and_determinism(counts, k, n, strided):
    torch.manual_seed(123)
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    rows, experts = sum(counts), len(counts)
    x = torch.randn((k, rows) if strided else (rows, k), device="cuda", dtype=torch.bfloat16)
    if strided:
        x = x.T
    x.requires_grad_()
    weight = (torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5).requires_grad_()
    dy = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for backend in ("independent_reference", "triton_sm120", "triton_sm120"):
        if backend == "independent_reference":
            start, pieces = 0, []
            for expert, count in enumerate(counts):
                pieces.append(torch.nn.functional.linear(x[start:start + count], weight[expert]))
                start += count
            y = torch.cat(pieces)
        else:
            y = grouped_linear(x, weight, offsets, backend)
        dx, dw = torch.autograd.grad(y, (x, weight), dy)
        outputs.append((y.detach(), dx, dw))
    for expected, actual in zip(outputs[0], outputs[1], strict=True):
        # BF16 reductions can differ in rounding; normalize by the reference RMS
        # rather than permitting unbounded relative errors around zero.
        if expected.numel():
            rms = expected.float().square().mean().sqrt().clamp_min(1e-10)
            error = (actual.float() - expected.float()) / rms
            assert error.square().mean().sqrt() < 0.006
            assert error.abs().max() < 0.04
    for a, b in zip(outputs[1], outputs[2], strict=True):
        assert torch.equal(a, b)
    for expert, count in enumerate(counts):
        if count == 0:
            assert torch.count_nonzero(outputs[1][2][expert]) == 0


@pytest.mark.skipif(not _sm120(), reason="requires SM120")
@pytest.mark.parametrize("k,n", [(2816, 1408), (704, 2816)])
def test_actual_frozen_expert_shapes(k, n):
    torch.manual_seed(12)
    x = torch.randn(8192, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    offsets = torch.arange(64, 8193, 64, device="cuda", dtype=torch.int32)
    dy = torch.randn(8192, n, device="cuda", dtype=torch.bfloat16)
    expected = grouped_linear(x, weight, offsets)
    actual = grouped_linear(x, weight, offsets, "triton_sm120")
    for left, right in [(expected, actual),
                        (torch.autograd.grad(expected, x, dy)[0], torch.autograd.grad(actual, x, dy)[0])]:
        error = (left.float() - right.float()) / left.float().square().mean().sqrt()
        assert error.square().mean().sqrt() < 0.006
        assert error.abs().max() < 0.04


def test_hybrid_keeps_frozen_base_backend(monkeypatch):
    from types import SimpleNamespace
    from decisionvla.adapters import LoRAExperts
    import decisionvla.grouped_gemm as gemm

    selected = []
    monkeypatch.setattr(gemm, "grouped_linear", lambda x, w, o, kernel: selected.append(kernel))
    layer = SimpleNamespace(kernel="triton_sm120_lora")
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    w = torch.zeros(1, 16, 16)
    offsets = torch.tensor([1], dtype=torch.int32)
    LoRAExperts._linear(layer, x, w, offsets, frozen_base=True)
    LoRAExperts._linear(layer, x, w, offsets)
    assert selected == ["torch_grouped_mm", "triton_sm120"]


@pytest.mark.skipif(not _sm120(), reason="requires SM120")
def test_backward_only_preserves_forward_exactly():
    torch.manual_seed(8)
    counts = [0, 17, 32, 79, 0, 3]
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = torch.randn(sum(counts), 704, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(len(counts), 32, 704, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(sum(counts), 32, device="cuda", dtype=torch.bfloat16)
    reference = grouped_linear(x, w, offsets)
    actual = grouped_linear(x, w, offsets, "triton_sm120_backward")
    assert torch.equal(reference, actual)
    for a, b in zip(torch.autograd.grad(reference, (x, w), dy),
                    torch.autograd.grad(actual, (x, w), dy), strict=True):
        error = (a.float() - b.float()) / a.float().square().mean().sqrt()
        assert error.square().mean().sqrt() < 0.006
        assert error.abs().max() < 0.04
