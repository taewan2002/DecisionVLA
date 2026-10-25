"""Deterministic BF16 grouped GEMMs using SM120-compatible Tensor Core tiles.

Every output tile has one writer. Expert boundaries stay on the device, and
weight gradients reduce their entire expert segment without atomic additions.
"""
import triton
import triton.language as tl


@triton.jit
def ragged_linear(X, W, OFF, Y, K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
                  XS0: tl.constexpr, XS1: tl.constexpr,
                  WS0: tl.constexpr, WS1: tl.constexpr, WS2: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GE: tl.constexpr):
    tile = tl.program_id(0)
    groups = tl.arange(0, GE)
    ends = tl.load(OFF + groups, groups < G, other=0)
    starts = tl.load(OFF + groups - 1, (groups > 0) & (groups < G), other=0)
    tiles = tl.cdiv(ends - starts, BM)
    boundaries = tl.cumsum(tiles)
    group = tl.sum(((tile >= boundaries) & (groups < G)).to(tl.int32))
    if group < G:
        first_tile = tl.sum(tl.where(groups < group, tiles, 0))
        start = tl.sum(tl.where(groups == group, starts, 0))
        end = tl.sum(tl.where(groups == group, ends, 0))
        rows = start + (tile - first_tile) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + ks
            a = tl.load(X + rows[:, None] * XS0 + k[None, :] * XS1,
                        (rows[:, None] < end) & (k[None, :] < K), other=0)
            b = tl.load(W + group * WS0 + cols[None, :] * WS1 + k[:, None] * WS2,
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
            acc = tl.dot(a, b, acc)
        tl.store(Y + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < end) & (cols[None, :] < N))


@triton.jit
def weight_gradient(X, DY, OFF, DW, K: tl.constexpr, N: tl.constexpr,
                    XS0: tl.constexpr, XS1: tl.constexpr,
                    DS0: tl.constexpr, DS1: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    ns = (tile // tl.cdiv(K, BK)) * BN + tl.arange(0, BN)
    ks = (tile % tl.cdiv(K, BK)) * BK + tl.arange(0, BK)
    start = tl.load(OFF + group - 1, group > 0, other=0)
    end = tl.load(OFF + group)
    rr = tl.arange(0, BR)
    acc = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(end - start, BR)):
        rows = start + block * BR + rr
        left = tl.load(DY + rows[None, :] * DS0 + ns[:, None] * DS1,
                       (rows[None, :] < end) & (ns[:, None] < N), other=0)
        right = tl.load(X + rows[:, None] * XS0 + ks[None, :] * XS1,
                        (rows[:, None] < end) & (ks[None, :] < K), other=0)
        acc = tl.dot(left, right, acc)
    tl.store(DW + group * N * K + ns[:, None] * K + ks[None, :], acc,
             (ns[:, None] < N) & (ks[None, :] < K))


def linear_forward(x, weight, offsets):
    import torch

    groups, out_features, in_features = weight.shape
    output = torch.empty((x.shape[0], out_features), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        bm, bn, bk = 32, 64, min(64, triton.next_power_of_2(in_features))
        bk = max(16, bk)
        # sum(ceil(group_rows/BM)) <= ceil(total_rows/BM) + groups - 1.
        grid = (triton.cdiv(x.shape[0], bm) + groups - 1, triton.cdiv(out_features, bn))
        ragged_linear[grid](x, weight, offsets, output, in_features, out_features, groups,
                            *x.stride(), *weight.stride(), bm, bn, bk,
                            triton.next_power_of_2(groups), num_warps=4, num_stages=2)
    return output


def linear_weight_gradient(x, dy, offsets, weight_shape):
    import torch

    groups, out_features, in_features = weight_shape
    dw = torch.empty(weight_shape, device=x.device, dtype=x.dtype)
    bn, bk, br = 32, 64, 32
    grid = (groups, triton.cdiv(out_features, bn) * triton.cdiv(in_features, bk))
    weight_gradient[grid](x, dy, offsets, dw, in_features, out_features,
                         *x.stride(), *dy.stride(), bn, bk, br, num_warps=4, num_stages=2)
    return dw
