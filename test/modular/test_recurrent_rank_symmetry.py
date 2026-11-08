"""Every rank of an instance starts serving with the same free slots in its recurrent pool:
each rank admits from its own pool, so a divergence here is one in every later verdict. The
vote is a stub.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool


class _Joint:
    """A joint group whose agree reduces what each rank of the instance holds."""

    def __init__(self, free: list[int]):
        self.free = free
        self.world_size = len(free)
        self.tp_group, self.sp_group = SimpleNamespace(world_size=len(free)), SimpleNamespace(world_size=1)

    def agree(self, values: list[int]) -> list[int]:
        assert len(values) == 2
        return [min(self.free), -max(self.free)]


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
    (_Joint([4]), False),
    (_Joint([4, 4, 4, 4]), False),
    (_Joint([4, 4, 4, 3]), True),
], ids=["no-group", "one-rank", "matching", "diverged"])
def test_post_warmup_validate(group, diverged):
    pool = build_pool(group)
    if not diverged:
        pool.post_warmup_validate()
        return
    with pytest.raises(RuntimeError, match="asymmetric free slots"):
        pool.post_warmup_validate()
