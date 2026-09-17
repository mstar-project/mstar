"""``MSTAR_TP_ALLREDUCE``: which all-reduces go through symmetric memory."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, ".")

import mstar.distributed.communication as comm
from mstar.distributed.communication import (
    TP_ALLREDUCE_ENV,
    TP_SYMM_AR_MAX_KB_ENV,
    CommGroup,
    WorkerParallelGroups,
)


class _FakeSymm:
    """Stands in for ``_SymmAllReduce``: records what it was built with and
    which tensors were routed to it."""

    instances: list[_FakeSymm] = []

    def __init__(self, device_group, device, max_bytes, mode):
        self.device_group = device_group
        self.device = device
        self.max_bytes = max_bytes
        self.mode = mode
        self.calls: list = []
        _FakeSymm.instances.append(self)

    def all_reduce_(self, t):
        self.calls.append(t)
        return t


class _RaisingSymm(_FakeSymm):
    def __init__(self, *a, **k):
        raise RuntimeError("no multicast support")


@dataclass
class _CudaTensor:
    """Only the attributes ``CommGroup.all_reduce`` inspects before routing."""

    nbytes: int
    contiguous: bool = True
    is_cuda: bool = True

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(TP_ALLREDUCE_ENV, raising=False)
    monkeypatch.delenv(TP_SYMM_AR_MAX_KB_ENV, raising=False)
    monkeypatch.setattr(comm, "_SymmAllReduce", _FakeSymm)
    _FakeSymm.instances.clear()


@pytest.fixture
def nccl_calls(monkeypatch):
    calls: list = []
    monkeypatch.setattr(dist, "all_reduce", lambda t, group=None: calls.append((t, group)))
    return calls


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def _group(world_size: int = 2) -> CommGroup:
    g = CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(world_size)))
    g.device_group = object()
    g.initialized = True
    return g


# --- setup: when is the symm path built at all -----------------------------


def test_default_mode_never_builds_symm(fake_cuda):
    g = _group()
    g._maybe_init_symm_allreduce()
    assert g._symm_ar is None
    assert _FakeSymm.instances == []


@pytest.mark.parametrize("mode", ["symm_oneshot", "symm_multimem", " Symm_OneShot "])
def test_symm_modes_build_it_with_default_cutoff(monkeypatch, fake_cuda, mode):
    monkeypatch.setenv(TP_ALLREDUCE_ENV, mode)
    g = _group()
    g._maybe_init_symm_allreduce()
    assert g._symm_ar is _FakeSymm.instances[0]
    assert g._symm_ar.max_bytes == 512 * 1024
    assert g._symm_ar.mode == mode.strip().lower()
    assert g._symm_ar.device_group is g.device_group


def test_max_kb_env_sets_cutoff(monkeypatch, fake_cuda):
    monkeypatch.setenv(TP_ALLREDUCE_ENV, "symm_oneshot")
    monkeypatch.setenv(TP_SYMM_AR_MAX_KB_ENV, "64")
    g = _group()
    g._maybe_init_symm_allreduce()
    assert g._symm_ar.max_bytes == 64 * 1024


def test_unknown_mode_is_nccl(monkeypatch, fake_cuda):
    monkeypatch.setenv(TP_ALLREDUCE_ENV, "symm_twoshot")
    g = _group()
    g._maybe_init_symm_allreduce()
    assert g._symm_ar is None


def test_single_rank_and_no_cuda_skip(monkeypatch):
    monkeypatch.setenv(TP_ALLREDUCE_ENV, "symm_oneshot")
    g = _group(world_size=1)
    g._maybe_init_symm_allreduce()
    assert g._symm_ar is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    g = _group()
    g._maybe_init_symm_allreduce()
    assert g._symm_ar is None


def test_setup_failure_falls_back_to_nccl_with_warning(monkeypatch, fake_cuda, caplog):
    monkeypatch.setenv(TP_ALLREDUCE_ENV, "symm_multimem")
    monkeypatch.setattr(comm, "_SymmAllReduce", _RaisingSymm)
    g = _group()
    with caplog.at_level(logging.WARNING, logger=comm.__name__):
        g._maybe_init_symm_allreduce()
    assert g._symm_ar is None
    assert any("using NCCL" in r.getMessage() for r in caplog.records)


def test_init_dist_sets_up_every_comm_group(monkeypatch, fake_cuda):
    """The hook point: ``init_dist`` builds the symm path for each group it
    wires a process group into."""
    monkeypatch.setenv(TP_ALLREDUCE_ENV, "symm_oneshot")
    pgs: list = []
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: None)
    monkeypatch.setattr(dist, "new_group", lambda **kw: pgs.append(object()) or pgs[-1])
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda _t: "gloo")

    groups = WorkerParallelGroups(
        num_workers=4, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1), (0, 1, 2, 3)],
    )
    tp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    sp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1, 2, 3])
    groups.add("llm", tp)
    groups.add("llm_cfg", tp)  # same object twice: set up once
    groups.add_sp("llm", sp)
    groups.init_dist(device=torch.device("cpu"))

    assert len(_FakeSymm.instances) == 2
    assert tp._symm_ar.device_group is pgs[0]
    assert sp._symm_ar.device_group is pgs[1]


# --- routing: which tensors take the symm path ------------------------------


def _symm_group(max_bytes: int = 1024) -> tuple[CommGroup, _FakeSymm]:
    g = _group()
    g._symm_ar = _FakeSymm(g.device_group, torch.device("cpu"), max_bytes, "symm_oneshot")
    return g, g._symm_ar


def test_small_contiguous_cuda_goes_symm(nccl_calls):
    g, symm = _symm_group(max_bytes=1024)
    t = _CudaTensor(nbytes=1024)
    assert g.all_reduce(t) is t
    assert symm.calls == [t]
    assert nccl_calls == []


def test_over_cutoff_goes_nccl(nccl_calls):
    g, symm = _symm_group(max_bytes=1024)
    t = _CudaTensor(nbytes=1025)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


def test_non_contiguous_goes_nccl(nccl_calls):
    g, symm = _symm_group()
    t = _CudaTensor(nbytes=16, contiguous=False)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


def test_cpu_tensor_goes_nccl(nccl_calls):
    g, symm = _symm_group()
    t = torch.zeros(4)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


def test_mode_nccl_routes_everything_to_nccl(nccl_calls, fake_cuda):
    g = _group()
    g._maybe_init_symm_allreduce()  # default mode: no symm path
    t = _CudaTensor(nbytes=16)
    g.all_reduce(t)
    assert nccl_calls == [(t, g.device_group)]
    assert _FakeSymm.instances == []


def test_single_rank_is_a_no_op(nccl_calls):
    g = CommGroup.trivial()
    t = torch.zeros(4)
    assert g.all_reduce(t) is t
    assert nccl_calls == []
