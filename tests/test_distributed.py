from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from decisionvla.batching import mean_sample_loss
from decisionvla.distributed import backward_microbatch, check_batch_config, sample_rank_indices

CONFIG = {"world_size": 2, "microbatch": 4, "gradient_accumulation": 8, "global_batch_size": 64}


def test_global_sample_positions_are_partitioned_and_resumeable():
    a, b, reference = [np.random.default_rng(42) for _ in range(3)]
    for _ in range(3):
        indices = reference.integers(100000, size=64)
        left = sample_rank_indices(a, 100000, CONFIG, 0)
        right = sample_rank_indices(b, 100000, CONFIG, 1)
        np.testing.assert_array_equal(np.concatenate([left.ravel(), right.ravel()]), indices)
        assert a.bit_generator.state == b.bit_generator.state == reference.bit_generator.state
    restored = np.random.default_rng()
    restored.bit_generator.state = a.bit_generator.state
    np.testing.assert_array_equal(sample_rank_indices(a, 100000, CONFIG, 0),
                                  sample_rank_indices(restored, 100000, CONFIG, 0))


@pytest.mark.parametrize("change", [{"world_size": 1}, {"microbatch": 0},
                                    {"global_batch_size": 32}, {"gradient_accumulation": 1.5}])
def test_invalid_global_batch_rejected(change):
    with pytest.raises(ValueError):
        check_batch_config({**CONFIG, **change}, 2)


class ToyObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    def forward(self, inputs, targets, masks):
        return mean_sample_loss(self.linear(inputs), targets, masks)


def _averaging_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(9)
        model = ToyObjective().double()
        reference = ToyObjective().double()
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(8, 7, 3, dtype=torch.float64)
        targets = torch.randint(4, (8, 7))
        masks = torch.arange(7)[None] <= torch.arange(8)[:, None]
        ddp = DistributedDataParallel(model, gradient_as_bucket_view=True)
        for micro in range(2):
            start = rank * 4 + micro * 2
            backward_microbatch(ddp, (inputs[start:start+2], targets[start:start+2], masks[start:start+2]),
                                micro, 2)
        reference(inputs, targets, masks).backward()
        torch.testing.assert_close(model.linear.weight.grad, reference.linear.weight.grad,
                                   rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_matches_global_equal_observation_gradient(tmp_path):
    mp.spawn(_averaging_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)
