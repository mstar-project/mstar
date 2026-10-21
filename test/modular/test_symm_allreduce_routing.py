"""``tp_allreduce``: the backend ``AllReduce.new`` sets up, and which all-reduces
go through symmetric memory."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, ".")

import mstar.distributed.all_reduce as ar
import mstar.distributed.communication as comm_mod
from mstar.distributed.all_reduce import (
    AllReduce,
    MultimemAllReduce,
    OneShotAllReduce,
    SymmMemAllReduce,
)
from mstar.distributed.communication import (
    CommGroup,
    GlobalParallelConfig,
    WorkerParallelGroups,
)

_REAL_BACKENDS = dict(ar.SYMM_BACKENDS)


class _FakeSymm(AllReduce):
    """Stands in for a ``SymmMemAllReduce`` subclass: records what it was built with."""

    instances: list[_FakeSymm] = []

    def __init__(self, device_group, device, max_bytes):
        super().__init__(device_group)
        self.device = device
        self.max_bytes = max_bytes
        _FakeSymm.instances.append(self)


class _RaisingSymm(_FakeSymm):
    def __init__(self, *a, **k):
        raise RuntimeError("no multicast support")


def _fakes(base: type[AllReduce]) -> dict[str, type[AllReduce]]:
    """``SYMM_BACKENDS`` with each mode's backend a ``base`` of that mode."""
    return {mode: type(base.__name__, (base,), {"mode": mode}) for mode in _REAL_BACKENDS}


@dataclass
class _CudaTensor:
    """Only the attributes ``SymmMemAllReduce.all_reduce`` inspects before routing."""

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
def _one_process(monkeypatch):
    monkeypatch.setattr(ar, "SYMM_BACKENDS", _fakes(_FakeSymm))
    # one process: every rank is this one
    monkeypatch.setattr(ar, "_all_ranks_agree", lambda flag, cpu_group: flag)
    monkeypatch.setattr(comm_mod, "_all_ranks_agree", lambda flag, cpu_group: flag)
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
    g.device_group = SimpleNamespace(group_name="tp")
    g.initialized = True
    return g


# --- setup: which backend AllReduce.new builds -----------------------------


def test_default_mode_is_nccl(fake_cuda):
    assert type(AllReduce.new(_group())) is AllReduce
    assert _FakeSymm.instances == []


@pytest.mark.parametrize("mode", ["symm_oneshot", "symm_multimem", " Symm_OneShot "])
def test_symm_modes_build_it_with_default_cutoff(fake_cuda, mode):
    g = _group()
    backend = AllReduce.new(g, mode)
    assert backend is _FakeSymm.instances[0]
    assert backend.max_bytes == 512 * 1024
    assert backend.mode == mode.strip().lower()
    assert backend.device_group is g.device_group


def test_single_rank_and_no_cuda_skip(monkeypatch):
    assert type(AllReduce.new(_group(world_size=1), "symm_oneshot")) is AllReduce

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert type(AllReduce.new(_group(), "symm_oneshot")) is AllReduce
    assert _FakeSymm.instances == []


def test_setup_failure_falls_back_to_nccl_with_warning(monkeypatch, fake_cuda, caplog):
    monkeypatch.setattr(ar, "SYMM_BACKENDS", _fakes(_RaisingSymm))
    with caplog.at_level(logging.WARNING, logger=ar.__name__):
        backend = AllReduce.new(_group(), "symm_multimem")
    assert type(backend) is AllReduce
    assert any("using NCCL" in r.getMessage() for r in caplog.records)


def test_a_peer_that_failed_takes_every_rank_to_nccl(monkeypatch, fake_cuda, caplog):
    """One rank on NCCL while its peers wait in the symmetric kernel for its signal
    deadlocked the first all-reduce."""
    monkeypatch.setattr(ar, "_all_ranks_agree", lambda flag, cpu_group: False)
    with caplog.at_level(logging.WARNING, logger=ar.__name__):
        backend = AllReduce.new(_group(), "symm_multimem")
    assert type(backend) is AllReduce
    assert any("another rank" in r.getMessage() for r in caplog.records)


def test_init_dist_sets_up_every_comm_group(monkeypatch, fake_cuda):
    """The hook point: ``init_dist`` sets the backend up for each group it
    wires a process group into."""
    pgs: list = []

    def new_group(**kw):
        pg = object()
        if kw.get("backend") != "gloo":  # the gloo twins carry no all-reduce
            pgs.append(pg)
        return pg

    monkeypatch.setattr(dist, "init_process_group", lambda **kw: None)
    monkeypatch.setattr(comm_mod, "rendezvous_store", lambda *args: object())
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda _t: "gloo")

    groups = WorkerParallelGroups(
        num_workers=4, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1), (0, 1, 2, 3)], tp_allreduce="symm_oneshot",
    )
    tp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    sp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1, 2, 3])
    groups.add("llm", tp)
    groups.add("llm_cfg", tp)  # same object twice: set up once
    groups.add_sp("llm", sp)
    groups.init_dist(device=torch.device("cpu"))

    assert len(_FakeSymm.instances) == 2
    assert tp._all_reduce.device_group is pgs[0]
    assert sp._all_reduce.device_group is pgs[1]


# --- the deployment config's tp_allreduce -----------------------------------


@pytest.mark.parametrize("config, want", [
    ("symm_multimem", "symm_multimem"),
    ("symm_oneshot", "symm_oneshot"),
    ("nccl", "nccl"),
    (None, "nccl"),
])
def test_config_mode(fake_cuda, config, want):
    assert AllReduce.new(_group(), config).mode == want


@pytest.mark.parametrize("config_kb, want_kb", [(None, 512), (64, 64), (32768, 32768), (0, None)])
def test_config_max_kb(fake_cuda, config_kb, want_kb):
    backend = AllReduce.new(_group(), "symm_multimem", config_kb)
    # 0 turns it off (it used to read as unset: 512)
    assert getattr(backend, "max_bytes", None) == (want_kb and want_kb * 1024)


def test_an_unknown_config_mode_is_refused(fake_cuda):
    # a typo would otherwise serve on NCCL behind a warning
    with pytest.raises(ValueError, match="tp_allreduce='symm_twoshot'"):
        AllReduce.new(_group(), "symm_twoshot")
    assert _FakeSymm.instances == []


def test_init_dist_takes_the_config_mode(monkeypatch, fake_cuda):
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: None)
    monkeypatch.setattr(comm_mod, "rendezvous_store", lambda *args: object())
    monkeypatch.setattr(dist, "new_group", lambda **kw: object())
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda _t: "gloo")
    groups = WorkerParallelGroups(
        num_workers=2, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1)], tp_allreduce="symm_multimem",
    )
    tp = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    groups.add("llm", tp)
    groups.init_dist(device=torch.device("cpu"))
    assert tp._all_reduce.mode == "symm_multimem"


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
    """The real backends over a faked symm_mem module; the returned handle's
    ``multicast_ptr`` is 0 as on a box without NVLS."""
    symm = pytest.importorskip("torch.distributed._symmetric_memory")
    handle = SimpleNamespace(multicast_ptr=0)
    monkeypatch.setattr(symm, "enable_symm_mem_for_group", lambda name: None)
    monkeypatch.setattr(symm, "empty", lambda n, dtype, device: torch.empty(n, dtype=dtype))
    monkeypatch.setattr(symm, "rendezvous", lambda buf, name: handle)
    monkeypatch.setattr(ar, "SYMM_BACKENDS", _REAL_BACKENDS)
    return handle


@pytest.mark.parametrize("mode, multicast_ptr, builds", [
    ("symm_multimem", 0, False),
    ("symm_multimem", 0x7F0000, True),
    ("symm_oneshot", 0, True),
])
def test_multimem_needs_multicast(fake_cuda, fake_symm_mem, caplog, mode, multicast_ptr, builds):
    fake_symm_mem.multicast_ptr = multicast_ptr
    with caplog.at_level(logging.WARNING, logger=ar.__name__):
        backend = AllReduce.new(_group(), mode)
    assert isinstance(backend, SymmMemAllReduce) == builds
    assert any("multicast" in r.getMessage() for r in caplog.records) == (not builds)


def _multimem(monkeypatch, fake_symm_mem, calls: list) -> MultimemAllReduce:
    """A real ``MultimemAllReduce`` over kernels that record themselves; the
    one-shot one writes 7s, the sum it would have stored."""
    fake_symm_mem.multicast_ptr = 0x7F0000

    def one_shot(buf, op, group, out):
        calls.append(("multimem_one_shot_all_reduce_out", out))
        out.fill_(7)

    ops = SimpleNamespace(
        multimem_one_shot_all_reduce_out=one_shot,
        multimem_all_reduce_=lambda buf, op, group: calls.append(("multimem_all_reduce_", buf)),
    )
    monkeypatch.setattr(torch.ops, "symm_mem", ops, raising=False)
    return MultimemAllReduce(SimpleNamespace(group_name="tp"), torch.device("cpu"), 512 * 1024)


@pytest.mark.parametrize("extra_bytes, kernel", [
    (0, "multimem_one_shot_all_reduce_out"), (16, "multimem_all_reduce_"),
])
def test_multimem_kernel_by_size(monkeypatch, fake_symm_mem, extra_bytes, kernel):
    calls: list = []
    symm = _multimem(monkeypatch, fake_symm_mem, calls)
    x = torch.ones((ar.MULTIMEM_ONESHOT_MAX_BYTES + extra_bytes) // 2, dtype=torch.bfloat16)
    assert symm._reduce(x) is x
    assert [name for name, _ in calls] == [kernel]
    if kernel == "multimem_one_shot_all_reduce_out":  # reduces straight into x
        assert calls[0][1].data_ptr() == x.data_ptr()


@pytest.mark.parametrize("offset", [0, 4, 8])  # bf16 elements: 0, 8 and 16 B in
def test_the_multimem_kernel_does_not_follow_alignment(monkeypatch, fake_symm_mem, offset):
    """Alignment is one rank's fact, and two ranks on different kernels hang the
    collective. A tensor the one-shot kernel can't store to gets the sum in an
    aligned one, copied back."""
    calls: list = []
    symm = _multimem(monkeypatch, fake_symm_mem, calls)
    n = ar.MULTIMEM_ONESHOT_MAX_BYTES // 2
    x = torch.ones(offset + n, dtype=torch.bfloat16)[offset:]
    aligned = x.data_ptr() % 16 == 0
    assert aligned == (offset != 4)
    assert symm._reduce(x) is x
    assert [name for name, _ in calls] == ["multimem_one_shot_all_reduce_out"]
    out = calls[0][1]
    assert out.data_ptr() % 16 == 0
    assert (out.data_ptr() == x.data_ptr()) == aligned
    assert bool((x == 7).all()), "the sum lands in x either way"


def test_one_shot_writes_the_sum_into_the_input(monkeypatch, fake_symm_mem):
    monkeypatch.setattr(torch.ops, "symm_mem", SimpleNamespace(
        one_shot_all_reduce=lambda buf, op, group: torch.full_like(buf, 7),
    ), raising=False)
    symm = OneShotAllReduce(SimpleNamespace(group_name="tp"), torch.device("cpu"), 512 * 1024)
    x = torch.ones(8, dtype=torch.bfloat16)
    assert symm._reduce(x) is x
    assert bool((x == 7).all())


def _multimem_group(monkeypatch, fake_symm_mem, calls: list) -> CommGroup:
    """A group over a real ``MultimemAllReduce`` (CPU buffer, faked kernels)."""
    fake_symm_mem.multicast_ptr = 0x7F0000
    monkeypatch.setattr(torch.ops, "symm_mem", SimpleNamespace(
        multimem_one_shot_all_reduce_out=lambda buf, op, group, out: calls.append(("one_shot", out)),
        multimem_all_reduce_=lambda buf, op, group: calls.append(("split", None)),
    ), raising=False)
    g = _group()
    g._all_reduce = MultimemAllReduce(g.device_group, torch.device("cpu"), 512 * 1024)
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
    assert (buf.data_ptr() == g._all_reduce._buf.data_ptr()) == is_buffer


def test_all_reduce_buffer_without_symm_is_a_new_tensor():
    g = _group()
    g.init_all_reduce()
    buf = g.all_reduce_buffer((8, 4096), torch.bfloat16, torch.device("cpu"))
    assert buf.shape == (8, 4096) and buf.dtype == torch.bfloat16


@pytest.mark.parametrize("extra_rows, kernel", [(0, "one_shot"), (1, "split")])
def test_reducing_the_buffer_returns_the_sum_in_a_new_tensor(
    monkeypatch, fake_symm_mem, extra_rows, kernel
):
    rows = ar.MULTIMEM_ONESHOT_MAX_BYTES // (4096 * 2) + extra_rows
    calls: list = []
    g = _multimem_group(monkeypatch, fake_symm_mem, calls)
    buf = g.all_reduce_buffer((rows, 4096), torch.bfloat16, torch.device("cpu"))
    buf.fill_(3)  # the producer's partial
    y = g._all_reduce._reduce(buf)
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
        return g._all_reduce._reduce(buf) * 2

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


@pytest.fixture
def symm_group(monkeypatch, fake_symm_mem):
    """A group over a real ``OneShotAllReduce``, whose symmetric path records
    the tensors routed to it."""

    def make(max_bytes: int = 1024) -> tuple[CommGroup, list]:
        g = _group()
        backend = OneShotAllReduce(g.device_group, torch.device("cpu"), max_bytes)
        routed: list = []
        monkeypatch.setattr(backend, "_reduce", lambda t: routed.append(t) or t)
        g._all_reduce = backend
        return g, routed

    return make


def test_small_contiguous_cuda_goes_symm(nccl_calls, symm_group):
    g, routed = symm_group(max_bytes=1024)
    t = _CudaTensor(nbytes=1024)
    assert g.all_reduce(t) is t
    assert routed == [t]
    assert nccl_calls == []


def test_over_cutoff_goes_nccl(nccl_calls, symm_group):
    g, routed = symm_group(max_bytes=1024)
    t = _CudaTensor(nbytes=1025)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


def test_non_contiguous_goes_nccl(nccl_calls, symm_group):
    g, routed = symm_group()
    t = _CudaTensor(nbytes=16, contiguous=False)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


@pytest.mark.parametrize("nbytes", [8186, 20, 1000, 4])
def test_misaligned_byte_size_goes_nccl(nccl_calls, symm_group, nbytes):
    """multimem_all_reduce_ / one_shot_all_reduce reject a message whose
    byte size is not aligned (a (1, 4093) bf16 is 8186 bytes) and raise
    inside the op, past the setup-time fallback; route those to NCCL."""
    assert nbytes % ar.SYMM_ALIGN_BYTES != 0
    g, routed = symm_group(max_bytes=16 * 1024)
    t = _CudaTensor(nbytes=nbytes, esize=2)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_dtypes_the_kernels_lack_go_nccl(nccl_calls, symm_group, dtype):
    g, routed = symm_group()
    t = _CudaTensor(nbytes=16, dtype=dtype)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


@pytest.mark.parametrize("nbytes", [16, 8192, 6144 * 2])
def test_aligned_byte_size_goes_symm(nccl_calls, symm_group, nbytes):
    g, routed = symm_group(max_bytes=16 * 1024)
    t = _CudaTensor(nbytes=nbytes, esize=2)
    g.all_reduce(t)
    assert routed == [t]
    assert nccl_calls == []


def test_fp32_goes_symm(nccl_calls, symm_group):
    g, routed = symm_group()
    t = _CudaTensor(nbytes=16, dtype=torch.float32)
    g.all_reduce(t)
    assert routed == [t]
    assert nccl_calls == []


def test_empty_message_goes_nccl(nccl_calls, symm_group):
    g, routed = symm_group()
    t = _CudaTensor(nbytes=0)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


def test_size_not_in_4_byte_units_goes_nccl(nccl_calls, symm_group):
    g, routed = symm_group()
    t = _CudaTensor(nbytes=6)  # e.g. 3 bf16 elements
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


def test_compiled_one_shot_goes_nccl(monkeypatch, nccl_calls, symm_group):
    """Inductor plans one_shot_all_reduce's input as a symmetric buffer of its own,
    sized by the first call: BAGEL at TP2 failed capture at its second decode size."""
    g, routed = symm_group()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    t = _CudaTensor(nbytes=16)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


def test_cpu_tensor_goes_nccl(nccl_calls, symm_group):
    g, routed = symm_group()
    t = torch.zeros(4)
    g.all_reduce(t)
    assert routed == []
    assert nccl_calls == [(t, g.device_group)]


def test_mode_nccl_routes_everything_to_nccl(nccl_calls, fake_cuda):
    g = _group()
    g.init_all_reduce()  # default mode: no symm path
    t = _CudaTensor(nbytes=16)
    g.all_reduce(t)
    assert nccl_calls == [(t, g.device_group)]
    assert _FakeSymm.instances == []


def test_single_rank_is_a_no_op(nccl_calls):
    g = CommGroup.trivial()
    t = torch.zeros(4)
    assert g.all_reduce(t) is t
    assert nccl_calls == []


# --- the one-shot kernel: a model opts in with init_allreduce_fusion(all_reduce=True) -------


@dataclass
class _Rows:
    """A [rows, hidden] CUDA tensor, as ``all_reduce`` inspects it."""

    shape: tuple
    dtype: torch.dtype = torch.bfloat16
    is_cuda: bool = True

    def is_contiguous(self):
        return True

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return self.dtype.itemsize


def _oneshot_group(monkeypatch, symm_group, calls: list, max_tokens: int = 128) -> CommGroup:
    from mstar.distributed import ar_fusion

    monkeypatch.setattr(ar_fusion, "create_workspace", lambda *a, **k: 0)
    monkeypatch.setattr(ar_fusion, "allreduce", lambda x, ws: calls.append((x, ws)) or x)
    g, _ = symm_group(max_bytes=1 << 20)
    assert g.init_allreduce_fusion(4096, torch.bfloat16, max_tokens, all_reduce=True)
    return g


@pytest.mark.parametrize("shape, dtype, oneshot", [
    ((4, 4096), torch.bfloat16, True),
    ((128, 4096), torch.bfloat16, True),  # the workspace's rows
    ((129, 4096), torch.bfloat16, False),
    ((4, 2048), torch.bfloat16, False),  # another width
    ((4, 4096), torch.float16, False),  # another dtype
    ((4, 2, 4096), torch.bfloat16, False),
])
def test_the_oneshot_kernel_takes_rows_of_its_shape(monkeypatch, nccl_calls, symm_group, shape,
                                                     dtype, oneshot):
    calls: list = []
    g = _oneshot_group(monkeypatch, symm_group, calls)
    t = _Rows(shape, dtype)
    g.all_reduce(t)
    assert calls == ([(t, 0)] if oneshot else [])


def test_the_fused_norm_alone_leaves_all_reduce_as_it_was(monkeypatch):
    from mstar.distributed import ar_fusion

    monkeypatch.setattr(ar_fusion, "create_workspace", lambda *a, **k: 0)
    g = _group()
    assert g.init_allreduce_fusion(4096, torch.bfloat16, 128)
    assert g._ar_oneshot is None
    # a second caller opts in on the same workspace
    assert g.init_allreduce_fusion(4096, torch.bfloat16, 128, all_reduce=True)
    assert g._ar_oneshot == (4096, torch.bfloat16) and g._ar_fusion_ws == 0


def test_the_all_reduce_buffer_is_a_new_tensor_for_the_oneshot_kernel(monkeypatch,
                                                                      fake_symm_mem):
    g = _multimem_group(monkeypatch, fake_symm_mem, [])
    from mstar.distributed import ar_fusion

    monkeypatch.setattr(ar_fusion, "create_workspace", lambda *a, **k: 0)
    g.init_allreduce_fusion(4096, torch.bfloat16, 128, all_reduce=True)
    buf = g.all_reduce_buffer((8, 4096), torch.bfloat16, torch.device("cpu"))
    assert buf.data_ptr() != g._all_reduce._buf.data_ptr()


def test_a_peer_without_a_workspace_keeps_every_rank_off_it(monkeypatch, caplog):
    from mstar.distributed import ar_fusion

    monkeypatch.setattr(ar_fusion, "create_workspace", lambda *a, **k: 0)
    monkeypatch.setattr(comm_mod, "_all_ranks_agree", lambda flag, cpu_group: False)
    g = _group()
    with caplog.at_level(logging.WARNING):
        assert not g.init_allreduce_fusion(4096, torch.bfloat16, 128, all_reduce=True)
    assert g._ar_fusion_ws is None and g._ar_oneshot is None
    assert any("another rank" in r.getMessage() for r in caplog.records)


def test_a_workspace_of_another_shape_takes_no_all_reduce(monkeypatch):
    from mstar.distributed import ar_fusion

    monkeypatch.setattr(ar_fusion, "create_workspace", lambda *a, **k: 0)
    g = _group()
    assert g.init_allreduce_fusion(2048, torch.bfloat16, 128)
    g.init_allreduce_fusion(4096, torch.bfloat16, 128, all_reduce=True)
    assert g._ar_oneshot is None

