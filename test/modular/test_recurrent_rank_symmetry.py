"""Every TP rank's recurrent pool starts serving with the same free slots: each rank admits
from its own pool, so a divergence here is one in every later verdict. The all-gather is a stub.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool


class _Group:
    """A comm group whose all-gather returns what the other ranks hold."""

    def __init__(self, values: list[int] | None):
        self.values = values
        self.world_size = 1 if values is None else len(values)

    def all_gather(self, local: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0 and local.shape == (1,)
        values = self.values or [int(local.item())]
        return torch.tensor(values, dtype=local.dtype)


class _Joint:
    def __init__(self, tp: list[int] | None, sp: list[int] | None = None):
        self.tp_group, self.sp_group = _Group(tp), _Group(sp)

    @property
    def world_size(self):
        return self.tp_group.world_size * self.sp_group.world_size


def build_pool(comm_group) -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        "state", {"llm"},
        RecurrentStateConfig(
            num_layers=1,
            blocks={"state": RecurrentBlockConfig(shape=(4, 2), dtype=torch.float32, shard_dims=(0,))},
            max_slots=5,
        ),
    )
    info = EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=comm_group)
    return RecurrentStatePool.build(spec, info)


@pytest.mark.parametrize("group, diverged", [
    (None, False),
    (_Joint(tp=None), False),
    (_Joint(tp=[4, 4]), False),
    (_Joint(tp=[4, 3]), True),
    (_Joint(tp=[4, 4], sp=[4, 2]), True),
], ids=["no-group", "one-rank", "matching", "tp-diverged", "sp-diverged"])
def test_post_warmup_validate(group, diverged):
    pool = build_pool(group)
    if not diverged:
        pool.post_warmup_validate()
        return
    with pytest.raises(RuntimeError, match="asymmetric free slots"):
        pool.post_warmup_validate()
