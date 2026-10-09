"""Every TP rank's recurrent pool starts serving with the same free slots.

Each rank admits from its own pool, so a divergence here is a divergence in
every later verdict. The check runs once, after warmup, over both comm groups.

CPU-only: the all-gather is a stub.
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


def test_matching_ranks_pass():
    build_pool(_Joint(tp=[4, 4])).post_warmup_validate()


def test_a_rank_with_fewer_free_slots_fails_loudly():
    pool = build_pool(_Joint(tp=[4, 3]))
    with pytest.raises(RuntimeError, match=r"asymmetric free slots.*\[4, 3\]"):
        pool.post_warmup_validate()


def test_the_sp_group_is_checked_too():
    pool = build_pool(_Joint(tp=[4, 4], sp=[4, 2]))
    with pytest.raises(RuntimeError, match="asymmetric"):
        pool.post_warmup_validate()


def test_a_single_rank_has_nothing_to_compare():
    build_pool(None).post_warmup_validate()
    build_pool(_Joint(tp=None)).post_warmup_validate()
