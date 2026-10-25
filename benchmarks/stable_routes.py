"""Stable expert route packing for reproducible fused MoE inference on SM120."""

import torch
import triton
import triton.language as tl


@triton.jit
def _count(ids, counts, R: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    selected = tl.load(ids + row, row < R, other=-1) == expert
    tl.store(counts + expert, tl.sum(selected.to(tl.int32)))


@triton.jit
def _prefix(counts, offsets, total, E: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    n = tl.load(counts + e, e < E, other=0)
    padded = ((n + B - 1) // B) * B
    end = tl.cumsum(padded)
    tl.store(offsets + e, end - padded, e < E)
    amount = tl.sum(padded)
    tl.store(offsets + E, amount)
    tl.store(total, amount)


@triton.jit
def _scatter(ids, offsets, routes, experts, R: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    selected = tl.load(ids + row, row < R, other=-1) == e
    rank = tl.cumsum(selected.to(tl.int32)) - 1
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    tl.store(routes + start + rank, row, selected)
    block = start // B + row
    tl.store(experts + block, e, block < end // B)


def stable_pack(
    topk_ids, block_size, num_experts, *, expert_map=None,
    packed_route_indices=None, block_expert_ids=None, packed_route_count=None,
    expert_offsets=None, stream=None,
):
    """Pack in (expert ID, original routed-pair index) order, without atomics."""
    if expert_map is not None:
        raise ValueError("DecisionVLA's stable route packer does not support expert parallelism")
    if topk_ids.device.type != "cuda" or topk_ids.dtype != torch.int32:
        raise ValueError("Expected CUDA int32 expert IDs")
    if not topk_ids.is_contiguous() or topk_ids.ndim != 2:
        raise ValueError("Expected contiguous [tokens, top_k] IDs")
    if block_size < 1 or num_experts < 1:
        raise ValueError("Invalid geometry")
    if stream is not None and int(stream) != torch.cuda.current_stream().cuda_stream:
        raise ValueError("Stable route packing requires the current CUDA stream")
    r = topk_ids.numel()
    if not 0 < r <= 16384:
        raise ValueError("This bounded inference packer supports at most 16384 routed pairs")
    slots = r + num_experts * (block_size - 1)
    slots = triton.cdiv(slots, block_size) * block_size
    device = topk_ids.device
    if packed_route_indices is None:
        packed_route_indices = torch.empty(slots, device=device, dtype=torch.int32)
    if block_expert_ids is None:
        block_expert_ids = torch.empty(triton.cdiv(packed_route_indices.numel(), block_size), device=device, dtype=torch.int32)
    if packed_route_count is None:
        packed_route_count = torch.empty(1, device=device, dtype=torch.int32)
    if expert_offsets is None:
        expert_offsets = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    # A padded total is a multiple of B, bounded by both R*B and
    # R + E*(B-1). Reject undersized caller-owned workspaces before launching.
    required = min(r * block_size, r + num_experts * (block_size - 1))
    required = required // block_size * block_size
    for name, value, minimum in (
        ("packed_route_indices", packed_route_indices, required),
        ("block_expert_ids", block_expert_ids, required // block_size),
        ("packed_route_count", packed_route_count, 1),
        ("expert_offsets", expert_offsets, num_experts + 1),
    ):
        if (value.device != device or value.dtype != torch.int32
                or not value.is_contiguous() or value.numel() < minimum):
            raise ValueError(f"Invalid {name} workspace")
    counts = torch.empty(num_experts, device=device, dtype=torch.int32)
    packed_route_indices.fill_(r)
    block_expert_ids.fill_(-1)
    block = triton.next_power_of_2(r)
    _count[(num_experts,)](topk_ids, counts, r, block)
    _prefix[(1,)](counts, expert_offsets, packed_route_count, num_experts, block_size, triton.next_power_of_2(num_experts))
    _scatter[(num_experts,)](topk_ids, expert_offsets, packed_route_indices, block_expert_ids, r, block_size, block)
    return packed_route_indices, block_expert_ids, packed_route_count


@torch.no_grad()
def check_stable_pack():
    """Independent CPU ordering/padding and repeated GPU output checks."""
    cases = []
    g = torch.Generator().manual_seed(19)
    for rows, topk, e, b in [(1, 2, 4, 16), (37, 8, 128, 16), (256, 8, 128, 32), (345, 8, 128, 64)]:
        ids = torch.randint(e, (rows, topk), generator=g, dtype=torch.int32)
        ids[0, 0] = -1
        expected, experts = [], []
        for expert in range(e):
            selected = (ids.flatten() == expert).nonzero().flatten().tolist()
            n = triton.cdiv(len(selected), b) * b
            expected.extend(selected + [ids.numel()] * (n - len(selected)))
            experts.extend([expert] * (n // b))
        x = ids.cuda()
        a = stable_pack(x, b, e)
        c = stable_pack(x, b, e)
        count = int(a[2])
        assert count == len(expected)
        assert a[0][:count].tolist() == expected
        assert a[1][:count // b].tolist() == experts
        assert all(torch.equal(u, v) for u, v in zip(a, c, strict=True))
        cases.append({"rows": rows, "topk": topk, "experts": e, "block": b, "passed": True})
    return cases
