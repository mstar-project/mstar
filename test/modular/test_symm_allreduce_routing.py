"""``tp_allreduce`` / ``MSTAR_TP_ALLREDUCE``: which all-reduces go through
symmetric memory."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, ".")

import mstar.distributed.communication as comm
from mstar.distributed.communication import (
    TP_ALLREDUCE_ENV,
    TP_SYMM_AR_MAX_KB_ENV,
    CommGroup,
    GlobalParallelConfig,
    WorkerParallelGroups,
)

_RealSymm = comm._SymmAllReduce


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
    esize: int = 1
    dtype: torch.dtype = torch.bfloat16

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return self.nbytes // self.esize

    def element_size(self):
        return self.esize


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

    def new_group(**kw):
        pg = object()
        if kw.get("backend") != "gloo":  # the gloo twins carry no all-reduce
            pgs.append(pg)
        return pg

    monkeypatch.setattr(dist, "init_process_group", lambda **kw: None)
    monkeypatch.setattr(dist, "new_group", new_group)
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


# --- the deployment config's tp_allreduce -----------------------------------


@pytest.mark.parametrize("config, env, want", [
    ("symm_multimem", None, "symm_multimem"),
    ("symm_oneshot", None, "symm_oneshot"),
    ("nccl", None, None),
    ("symm_multimem", "nccl", None),  # the env var overrides the config
    ("symm_multimem", "symm_oneshot", "symm_oneshot"),
    (None, "symm_multimem", "symm_multimem"),
])
def test_config_mode_and_env_override(monkeypatch, fake_cuda, config, env, want):
    if env is not None:
        monkeypatch.setenv(TP_ALLREDUCE_ENV, env)
    g = _group()
    g._maybe_init_symm_allreduce(config)
    assert getattr(g._symm_ar, "mode", None) == want


@pytest.mark.parametrize("config_kb, env, want_kb", [
    (None, None, 512), (32768, None, 32768), (32768, "64", 64), (None, "64", 64),
])
def test_config_max_kb_and_env_override(monkeypatch, fake_cuda, config_kb, env, want_kb):
    if env is not None:
        monkeypatch.setenv(comm.TP_SYMM_AR_MAX_KB_ENV, env)
    g = _group()
    g._maybe_init_symm_allreduce("symm_multimem", config_kb)
    assert g._symm_ar.max_bytes == want_kb * 1024


def test_unknown_config_mode_warns_and_is_nccl(fake_cuda, caplog):
    g = _group()
    with caplog.at_level(logging.WARNING, logger=comm.__name__):
        g._maybe_init_symm_allreduce("symm_twoshot")
    assert g._symm_ar is None
    assert any("unknown all-reduce mode" in r.getMessage() for r in caplog.records)


def test_init_dist_takes_the_config_mode(monkeypatch, fake_cuda):
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: None)
    monkeypatch.setattr(dist, "new_group", lambda **kw: object())
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda _t: "gloo")
    groups = WorkerParallelGroups(
        num_workers=2, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1)], tp_allreduce="symm_multimem",
    )
    tp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    groups.add("llm", tp)
    groups.init_dist(device=torch.device("cpu"))
    assert tp._symm_ar.mode == "symm_multimem"


@pytest.mark.parametrize("tp_allreduce", [None, "symm_multimem"])
def test_global_config_hands_it_to_every_worker(tp_allreduce):
    wg = SimpleNamespace(
        tp_size=2, sp_size=1, _tp_ranks=[[0, 1]], _sp_ranks=[[0], [1]],
        ranks=[0, 1], _tp_comm_size=2, _instance_ranks=[[0, 1]],
        section=SimpleNamespace(get_nodes=lambda: {"llm": None}),
    )
    cfg = GlobalParallelConfig(
        {"wg": wg}, ["worker_0", "worker_1"], tp_allreduce=tp_allreduce
    )
    assert [w.tp_allreduce for w in cfg.per_worker_config.values()] == [tp_allreduce] * 2


@pytest.fixture
def fake_symm_mem(monkeypatch):
    """The real ``_SymmAllReduce`` over a faked symm_mem module; the
    returned handle's ``multicast_ptr`` is 0 as on a box without NVLS."""
    symm = pytest.importorskip("torch.distributed._symmetric_memory")
    handle = SimpleNamespace(multicast_ptr=0)
    monkeypatch.setattr(symm, "enable_symm_mem_for_group", lambda name: None)
    monkeypatch.setattr(symm, "empty", lambda n, dtype, device: torch.empty(n, dtype=dtype))
    monkeypatch.setattr(symm, "rendezvous", lambda buf, name: handle)
    monkeypatch.setattr(comm, "_SymmAllReduce", _RealSymm)
    return handle


@pytest.mark.parametrize("mode, multicast_ptr, builds", [
    ("symm_multimem", 0, False),
    ("symm_multimem", 0x7F0000, True),
    ("symm_oneshot", 0, True),
])
def test_multimem_needs_multicast(
    monkeypatch, fake_cuda, fake_symm_mem, caplog, mode, multicast_ptr, builds
):
    fake_symm_mem.multicast_ptr = multicast_ptr
    monkeypatch.setenv(TP_ALLREDUCE_ENV, mode)
    g = _group()
    g.device_group = SimpleNamespace(group_name="tp")
    with caplog.at_level(logging.WARNING, logger=comm.__name__):
        g._maybe_init_symm_allreduce()
    assert (g._symm_ar is not None) == builds
    assert any("multicast" in r.getMessage() for r in caplog.records) == (not builds)


@pytest.mark.parametrize("extra_bytes, offset, kernel", [
    (0, 0, "multimem_one_shot_all_reduce_out"),
    (16, 0, "multimem_all_reduce_"),
    (0, 4, "multimem_all_reduce_"),  # starts 8 B into its storage
    (0, 8, "multimem_one_shot_all_reduce_out"),  # 16 B in: aligned
])
def test_multimem_kernel_by_size_and_alignment(
    monkeypatch, fake_symm_mem, extra_bytes, offset, kernel
):
    fake_symm_mem.multicast_ptr = 0x7F0000
    calls: list = []
    ops = SimpleNamespace(
        multimem_one_shot_all_reduce_out=lambda buf, op, group, out: calls.append(
            ("multimem_one_shot_all_reduce_out", out)),
        multimem_all_reduce_=lambda buf, op, group: calls.append(("multimem_all_reduce_", buf)),
    )
    monkeypatch.setattr(torch.ops, "symm_mem", ops, raising=False)
    symm = _RealSymm(SimpleNamespace(group_name="tp"), torch.device("cpu"), 512 * 1024,
                     "symm_multimem")
    n = (comm.MULTIMEM_ONESHOT_MAX_BYTES + extra_bytes) // 2
    x = torch.ones(offset + n, dtype=torch.bfloat16)[offset:]
    assert x.data_ptr() % 16 == 2 * offset % 16
    assert symm.all_reduce_(x) is x
    assert [name for name, _ in calls] == [kernel]
    if kernel == "multimem_one_shot_all_reduce_out":  # reduces straight into x
        assert calls[0][1].data_ptr() == x.data_ptr()


def _multimem_group(monkeypatch, fake_symm_mem, calls: list) -> CommGroup:
    """A group over the real ``_SymmAllReduce`` (CPU buffer, faked kernels)."""
    fake_symm_mem.multicast_ptr = 0x7F0000
    monkeypatch.setattr(torch.ops, "symm_mem", SimpleNamespace(
        multimem_one_shot_all_reduce_out=lambda buf, op, group, out: calls.append(("one_shot", out)),
        multimem_all_reduce_=lambda buf, op, group: calls.append(("split", None)),
    ), raising=False)
    g = _group()
    g._symm_ar = _RealSymm(SimpleNamespace(group_name="tp"), torch.device("cpu"), 512 * 1024,
                           "symm_multimem")
    return g


@pytest.mark.parametrize("shape, dtype, is_buffer", [
    ((8, 4096), torch.bfloat16, True),
    ((64, 4096), torch.bfloat16, True),  # 512 KiB, the max
    ((65, 4096), torch.bfloat16, False),  # NCCL-sized
    ((8, 4096), torch.float16, False),  # NCCL dtype
])
def test_all_reduce_buffer_is_the_symm_buffer_when_it_fits(
    monkeypatch, fake_symm_mem, shape, dtype, is_buffer
):
    g = _multimem_group(monkeypatch, fake_symm_mem, [])
    buf = g.all_reduce_buffer(shape, dtype, torch.device("cpu"))
    assert buf.shape == shape and buf.dtype == dtype
    assert (buf.data_ptr() == g._symm_ar._buf.data_ptr()) == is_buffer


def test_all_reduce_buffer_without_symm_is_a_new_tensor():
    g = _group()
    buf = g.all_reduce_buffer((8, 4096), torch.bfloat16, torch.device("cpu"))
    assert buf.shape == (8, 4096) and buf.dtype == torch.bfloat16


@pytest.mark.parametrize("extra_rows, kernel", [(0, "one_shot"), (1, "split")])
def test_reducing_the_buffer_returns_the_sum_in_a_new_tensor(
    monkeypatch, fake_symm_mem, extra_rows, kernel
):
    rows = comm.MULTIMEM_ONESHOT_MAX_BYTES // (4096 * 2) + extra_rows
    calls: list = []
    g = _multimem_group(monkeypatch, fake_symm_mem, calls)
    buf = g.all_reduce_buffer((rows, 4096), torch.bfloat16, torch.device("cpu"))
    buf.fill_(3)  # the producer's partial
    y = g._symm_ar.all_reduce_(buf)
    assert y.shape == buf.shape and y.data_ptr() != buf.data_ptr()
    assert [name for name, _ in calls] == [kernel]
    if kernel == "one_shot":
        assert calls[0][1].data_ptr() == y.data_ptr()
    else:  # copied out of the buffer the kernel reduced in place
        assert bool((y == 3).all())


def test_a_compiled_forward_traces_the_symm_path_in_one_graph(monkeypatch, fake_symm_mem):
    """GLM-5.2 captures a torch.compile'd forward: data_ptr() would break its
    graph at every all-reduce, so compiled code gets no buffer handout and
    takes the split kernel (fullgraph=True raises on any break)."""
    calls: list = []
    g = _multimem_group(monkeypatch, fake_symm_mem, calls)

    def forward(x):
        buf = g.all_reduce_buffer(x.shape, x.dtype, x.device)
        torch.add(x, 1, out=buf)
        return g._symm_ar.all_reduce_(buf) * 2

    # dynamo imports inductor, which the conftest's triton stub breaks; a real
    # triton stays, since dropping it but not its submodules breaks its re-import
    for name in ("triton", "triton.language"):
        if name in sys.modules and not hasattr(sys.modules[name], "__file__"):
            monkeypatch.delitem(sys.modules, name)
    torch._dynamo.reset()
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    out = torch.compile(forward, backend="eager", fullgraph=True)(x)
    assert [name for name, _ in calls] == ["split"]
    assert bool((out == 4).all())


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


@pytest.mark.parametrize("nbytes", [8186, 20, 1000, 4])
def test_misaligned_byte_size_goes_nccl(nccl_calls, nbytes):
    """multimem_all_reduce_ / one_shot_all_reduce reject a message whose
    byte size is not aligned (a (1, 4093) bf16 is 8186 bytes) and raise
    inside the op, past the setup-time fallback; route those to NCCL."""
    from mstar.distributed.communication import TP_SYMM_AR_ALIGN_BYTES

    assert nbytes % TP_SYMM_AR_ALIGN_BYTES != 0
    g, symm = _symm_group(max_bytes=16 * 1024)
    t = _CudaTensor(nbytes=nbytes, esize=2)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_dtypes_the_kernels_lack_go_nccl(nccl_calls, dtype):
    g, symm = _symm_group()
    t = _CudaTensor(nbytes=16, dtype=dtype)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


@pytest.mark.parametrize("nbytes", [16, 8192, 6144 * 2])
def test_aligned_byte_size_goes_symm(nccl_calls, nbytes):
    g, symm = _symm_group(max_bytes=16 * 1024)
    t = _CudaTensor(nbytes=nbytes, esize=2)
    g.all_reduce(t)
    assert symm.calls == [t]
    assert nccl_calls == []


def test_fp32_goes_symm(nccl_calls):
    g, symm = _symm_group()
    t = _CudaTensor(nbytes=16, dtype=torch.float32)
    g.all_reduce(t)
    assert symm.calls == [t]
    assert nccl_calls == []


def test_empty_message_goes_nccl(nccl_calls):
    g, symm = _symm_group()
    t = _CudaTensor(nbytes=0)
    g.all_reduce(t)
    assert symm.calls == []
    assert nccl_calls == [(t, g.device_group)]


def test_size_not_in_4_byte_units_goes_nccl(nccl_calls):
    g, symm = _symm_group()
    t = _CudaTensor(nbytes=6)  # e.g. 3 bf16 elements
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
