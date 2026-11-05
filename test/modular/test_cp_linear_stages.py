"""``cp_linear`` keeps as many pipeline stages as the device's shared memory
holds: the stage count is a host-side choice, so this runs without a GPU."""
import pytest
import torch

from mstar.model.components import code_predictor_kernels as cpk

# shared memory per block, bytes
H200, MI325X = 233472, 65536


@pytest.mark.parametrize(
    "block_m,block_n,block_k,operands,stages,lds,want",
    [
        # the 64-row tile: 128-wide K steps, 5 stages as tuned on an H200
        (64, 16, 128, 1, 5, H200, 5),
        (64, 16, 128, 1, 5, MI325X, 4),
        (64, 32, 128, 2, 5, H200, 5),
        (64, 32, 128, 2, 5, MI325X, 3),
        # the 32-row tile: 256-wide K steps, 4 stages
        (32, 16, 256, 1, 4, H200, 4),
        (32, 16, 256, 1, 4, MI325X, 3),
        (32, 32, 256, 2, 4, H200, 4),
        (32, 32, 256, 2, 4, MI325X, 2),
    ],
)
def test_stages_fit_the_shared_memory(monkeypatch, block_m, block_n, block_k, operands, stages, lds, want):
    monkeypatch.setattr(cpk, "_shared_memory_per_block", lambda device: lds)
    got = cpk._stages_that_fit(block_m, block_n, block_k, operands, stages, torch.device("cpu"))
    assert got == want
    # the stages - 1 resident tiles fit, and one more stage would not
    per_stage = (block_m + operands * block_n) * block_k * 2
    assert (got - 1) * per_stage <= lds
    assert got == stages or got * per_stage > lds
