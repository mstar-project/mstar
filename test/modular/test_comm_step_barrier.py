"""MSTAR_TP_STEP_BARRIER gates the per-step engine barrier, nothing else.

The flag exists because dist.barrier() on the NCCL backend ends in a
current-stream synchronize: it blocks the host until the previous step drains,
which forbids the N+1-behind-N enqueue that TP-async scheduling needs. Default
must stay "1" (unchanged behaviour); "0" must skip ONLY ``step_barrier`` — the
plain ``barrier()`` used at capture time is untouched either way.
"""

from unittest.mock import patch

import pytest

from mstar.distributed import communication as comm


def _group(world_size: int) -> comm.CommGroup:
    return comm.CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(world_size)))


@pytest.mark.parametrize("value,expect_call", [(None, True), ("1", True), ("0", False), ("off", False)])
def test_step_barrier_honours_flag(monkeypatch, value, expect_call):
    if value is None:
        monkeypatch.delenv(comm.TP_STEP_BARRIER_ENV, raising=False)
    else:
        monkeypatch.setenv(comm.TP_STEP_BARRIER_ENV, value)
    g = _group(2)
    with patch.object(comm.dist, "barrier") as barrier:
        g.step_barrier()
    assert barrier.called is expect_call


def test_plain_barrier_ignores_flag(monkeypatch):
    monkeypatch.setenv(comm.TP_STEP_BARRIER_ENV, "0")
    g = _group(2)
    with patch.object(comm.dist, "barrier") as barrier:
        g.barrier()
    assert barrier.called


def test_trivial_group_never_barriers(monkeypatch):
    monkeypatch.delenv(comm.TP_STEP_BARRIER_ENV, raising=False)
    g = _group(1)
    with patch.object(comm.dist, "barrier") as barrier:
        g.step_barrier()
        g.barrier()
    assert not barrier.called
