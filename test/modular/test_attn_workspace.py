"""A FlashInfer workspace grows to what a plan asks for, until a graph can hold its address.

The planner stands in for FlashInfer's: it takes its allocations off the buffer it was handed
and reports a short one in the words of AlignedAllocator. The GPU tests at the bottom run the
real one, which is what keeps `workspace._OVERFLOW` honest for a new FlashInfer."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from mstar.engine import engine as engine_mod
from mstar.engine.resources.attn import flashinfer as flashinfer_mod
from mstar.engine.resources.attn import workspace as workspace_mod
from mstar.engine.resources.attn import wrappers as wrappers_mod
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.cross import FlashInferCrossManager
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager, FlashInferMLAWrapper
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig
from mstar.engine.resources.attn.ragged.flashinfer import (
    FlashInferRaggedBlockCausalManager,
    FlashInferRaggedCrossManager,
    FlashInferRaggedManager,
)
from mstar.engine.resources.attn.ragged.wrappers import (
    FlashInferRaggedCrossPrefillWrapper,
    FlashInferRaggedPrefillWrapper,
)
from mstar.engine.resources.attn.workspace import Workspace, WorkspacePool, workspace_bytes
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

MIB = 1024 * 1024
CPU = torch.device("cpu")


def _overflow(name: str, size: int, left: int) -> str:
    return (
        f"Buffer overflow when allocating memory for {name} with size {size} and alignment 16, "
        f"but only {left} bytes available in AlignedAllocator. Increase the workspace buffer size."
    )


class _Planner:
    """Allocates a partial output, then its log-sum-exp (``tmp_v``, then the 1/128 of it ``tmp_s``
    takes at head dim 128), from ``tensor``."""

    def __init__(
        self, tensor: torch.Tensor, tmp_v: int, head_dim: int = 128,
        names: tuple[str, str] = ("batch_prefill_tmp_v", "batch_prefill_tmp_s"),
    ):
        self.tensor = tensor
        self.allocations = ((names[0], tmp_v), (names[1], tmp_v // head_dim))
        self.plans = 0

    def plan(self, *args, **kwargs) -> None:
        self.plans += 1
        left = self.tensor.numel()
        for name, size in self.allocations:
            if size > left:
                raise RuntimeError(_overflow(name, size, left))
            left -= size


def test_a_plan_grows_the_workspace_to_what_it_asked_for():
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    planner = _Planner(workspace.tensor, 5 * MIB + 4096)
    workspace.plan(planner.plan)
    # the first allocation, to the MiB, leaves room for the second
    assert planner.plans == 2
    assert workspace.tensor.numel() == 6 * MIB


@pytest.mark.parametrize("head_dim, names", [
    (128, ("batch_prefill_tmp_v", "batch_prefill_tmp_s")),
    # tmp_s is 1/head_dim of tmp_v, so more than 1/32 of it below head dim 32
    (16, ("batch_decode_tmp_v", "batch_decode_tmp_s")),
    (64, ("holistic_partial_o", "holistic_partial_lse")),
], ids=["prefill", "decode-head-dim-16", "holistic"])
def test_a_plan_that_overflows_twice_fits_on_its_third(caplog, head_dim, names):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    # a partial output of whole MiB, so growing by it leaves nothing for its log-sum-exp
    planner = _Planner(workspace.tensor, 5 * MIB, head_dim, names)
    with caplog.at_level("INFO", logger=workspace_mod.logger.name):
        workspace.plan(planner.plan)
    assert planner.plans == 3
    assert workspace.tensor.numel() == 6 * MIB
    assert f"1 -> 5 MiB, a plan asked for {5 * MIB} bytes" in caplog.text
    assert f"5 -> 6 MiB, a plan asked for {5 * MIB + 5 * MIB // head_dim} bytes" in caplog.text


def test_a_plan_that_fits_changes_nothing():
    workspace = WorkspacePool(CPU, size_mb=4).get("main", 0)
    before = workspace.tensor.data_ptr()
    planner = _Planner(workspace.tensor, 3 * MIB)
    workspace.plan(planner.plan)
    assert planner.plans == 1
    assert workspace.tensor.numel() == 4 * MIB
    assert workspace.tensor.data_ptr() == before


def test_a_plan_short_only_of_its_second_allocation_grows_too():
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    planner = _Planner(workspace.tensor, MIB)
    workspace.plan(planner.plan)
    assert planner.plans == 2
    assert workspace.tensor.numel() == 2 * MIB


def test_every_wrapper_on_a_key_moves_with_it():
    pool = WorkspacePool(CPU, size_mb=1)
    small, large = _Planner(pool.get("main", 0).tensor, MIB // 2), _Planner(pool.get("main", 0).tensor, 8 * MIB)
    pool.get("main", 0).plan(small.plan)
    pool.get("main", 0).plan(large.plan)
    # one buffer for the key, which the wrapper planned first sees as well
    assert small.tensor is large.tensor is pool.get("main", 0).tensor
    assert small.tensor.numel() >= 8 * MIB
    assert list(pool._workspaces) == ["main_0"]


def test_each_key_grows_on_its_own():
    pool = WorkspacePool(CPU, size_mb=1)
    grown = pool.get("video", 0)
    grown.plan(_Planner(grown.tensor, 8 * MIB).plan)
    assert pool.get("video", 1).tensor.numel() == MIB
    assert pool.get("audio", 0).tensor.numel() == MIB


def test_a_declared_workspace_mb_wins_over_the_env_var(monkeypatch):
    monkeypatch.delenv("MSTAR_WORKSPACE_BUFFER_MB", raising=False)
    assert workspace_bytes() == 512 * MIB
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "3")
    assert workspace_bytes() == 3 * MIB
    from_env = WorkspacePool(CPU).get("main")
    declared = WorkspacePool(CPU, size_mb=2).get("main")
    assert from_env.tensor.numel() == 3 * MIB
    # the env var is the default for a resource that declares nothing, smaller or larger
    assert declared.tensor.numel() == 2 * MIB
    assert WorkspacePool(CPU, size_mb=5).get("main").tensor.numel() == 5 * MIB
    from_env.plan(_Planner(from_env.tensor, MIB).plan)
    assert from_env.tensor.numel() == 3 * MIB
    # and neither is a cap: a plan grows a declared workspace as it does any other
    declared.plan(_Planner(declared.tensor, 7 * MIB).plan)
    assert declared.tensor.numel() >= 7 * MIB


@pytest.mark.parametrize("size_mb, knob", [
    (None, "It starts at MSTAR_WORKSPACE_BUFFER_MB: set that to at least 6."),
    (1, "It starts at the workspace_mb dit_attn declares: set that to at least 6."),
])
def test_a_sealed_workspace_keeps_its_address_and_names_the_bytes_and_the_knob(monkeypatch, size_mb, knob):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    pool = WorkspacePool(CPU, size_mb=size_mb, owner="dit_attn")
    workspace = pool.get("main", 1)
    before = workspace.tensor.data_ptr()
    pool.seal()
    planner = _Planner(workspace.tensor, 5 * MIB + 4096)
    with pytest.raises(RuntimeError) as err:
        workspace.plan(planner.plan)
    assert workspace.tensor.data_ptr() == before and workspace.tensor.numel() == MIB
    # what a plan asked for, in bytes; which workspace; which setting; how far to raise it
    assert str(err.value) == (
        f"FlashInfer workspace dit_attn/main_1: a plan asks for {5 * MIB + 4096} bytes (6 MiB) and it holds "
        f"{MIB} bytes; it can't grow once a captured graph can hold its address. {knob}"
    )
    assert planner.plans == 1


def test_the_seal_says_what_a_pool_holds(caplog):
    pool = WorkspacePool(CPU, size_mb=1, owner="dit_attn")
    pool.get("main", 0), pool.get("main", 1)
    with caplog.at_level("INFO", logger=workspace_mod.logger.name):
        pool.seal()
    assert "FlashInfer workspaces of dit_attn: 2, 2 MiB" in caplog.text


def test_a_workspace_made_after_the_seal_is_born_sealed():
    """A wrapper made past the seal may still be captured over its key, so no later plan on
    that key may move it."""
    pool = WorkspacePool(CPU, size_mb=1)
    pool.seal()
    workspace = pool.get("eager", 0)
    assert workspace.sealed
    with pytest.raises(RuntimeError, match="captured graph can hold its address"):
        workspace.plan(_Planner(workspace.tensor, 5 * MIB).plan)
    assert workspace.tensor.numel() == MIB


def test_a_plan_still_short_after_three_says_what_each_ran_short_of():
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    calls = []

    def plan():
        calls.append(1)
        raise RuntimeError(_overflow("batch_prefill_tmp_v", (len(calls) * 10) * MIB, workspace.tensor.numel()))

    with pytest.raises(RuntimeError) as err:
        workspace.plan(plan)
    assert len(calls) == 3
    assert str(err.value) == (
        f"FlashInfer workspace attention/main_0: a plan asks for {30 * MIB} bytes (30 MiB) and it holds "
        f"{20 * MIB} bytes; 3 plans ran short, of batch_prefill_tmp_v ({10 * MIB} bytes), "
        f"batch_prefill_tmp_v ({20 * MIB} bytes), batch_prefill_tmp_v ({30 * MIB} bytes). "
        "It starts at the workspace_mb attention declares: set that to at least 30."
    )


def _short_of_memory(monkeypatch, above: int) -> None:
    """``torch.empty`` in the workspace module fails past ``above`` bytes, as a full GPU does."""
    real_empty = torch.empty

    def empty(size, **kwargs):
        if size > above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real_empty(size, **kwargs)

    monkeypatch.setattr(workspace_mod.torch, "empty", empty)


def test_no_room_to_grow_says_what_it_asked_for_and_keeps_what_it_held(monkeypatch):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    planner = _Planner(workspace.tensor, 5 * MIB + 4096)
    _short_of_memory(monkeypatch, above=MIB)
    with pytest.raises(RuntimeError) as err:
        workspace.plan(planner.plan)
    assert str(err.value) == (
        f"FlashInfer workspace attention/main_0: a plan asks for {5 * MIB + 4096} bytes (6 MiB) and it holds "
        f"{MIB} bytes; the GPU has no room to grow it: CUDA out of memory"
    )
    # its bytes are back, so a plan that fitted before still does
    assert workspace.tensor.numel() == MIB
    planner.allocations = (("batch_prefill_tmp_v", MIB // 2),)
    workspace.plan(planner.plan)


def test_no_room_to_take_back_what_it_held_says_so(monkeypatch):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    _short_of_memory(monkeypatch, above=0)
    with pytest.raises(RuntimeError) as err:
        workspace.plan(_Planner(workspace.tensor, 5 * MIB + 4096).plan)
    assert str(err.value).endswith(
        f"and it holds 0 bytes; the GPU has no room to grow it: CUDA out of memory; nor to take back the {MIB} "
        "bytes it held"
    )
    assert workspace.tensor.numel() == 0


@pytest.mark.parametrize("message", [
    "The batch size should be fixed during the lifecycle of the wrapper",
    # FlashInfer's own int workspace, which this buffer has no part in
    _overflow("batch_prefill_request_indices", 9 * MIB, MIB),
])
def test_other_planner_errors_are_not_retried(message):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    calls = []

    def plan():
        calls.append(1)
        raise RuntimeError(message)

    with pytest.raises(RuntimeError) as err:
        workspace.plan(plan)
    assert str(err.value) == message
    assert len(calls) == 1
    assert workspace.tensor.numel() == MIB


class _NativeWrapper:
    """A FlashInfer wrapper as far as a plan goes, asking for ``asks`` of its workspace."""

    asks = 5 * MIB + 4096
    built: list[_NativeWrapper] = []

    def __init__(self, workspace_buffer, *args, **kwargs):
        self.workspace = workspace_buffer
        self.plans = 0
        type(self).built.append(self)

    def plan(self, *args, **kwargs):
        self.plans += 1
        _Planner(self.workspace, type(self).asks).plan()


@pytest.fixture
def native(monkeypatch):
    _NativeWrapper.asks = 5 * MIB + 4096
    _NativeWrapper.built = []
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=_NativeWrapper,
        BatchDecodeWithPagedKVCacheWrapper=_NativeWrapper,
        BatchPrefillWithRaggedKVCacheWrapper=_NativeWrapper,
    ))
    return _NativeWrapper


def _one_row() -> dict:
    return dict(
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([3], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
    )


@pytest.mark.parametrize("wrapper_class, extra", [
    (FlashInferPrefillWrapper, dict(qo_indptr=torch.tensor([0, 1], dtype=torch.int32))),
    (FlashInferDecodeWrapper, {}),
])
def test_the_paged_wrappers_plan_through_their_workspace(native, wrapper_class, extra):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    wrapper = wrapper_class(workspace, num_qo_heads=2, num_kv_heads=2, head_dim=64, page_size=16, device=CPU)
    assert wrapper.attn_wrapper.workspace is workspace.tensor
    wrapper.plan(**_one_row(), **extra)
    assert wrapper.attn_wrapper.plans == 2
    assert workspace.tensor.numel() >= 5 * MIB


def test_the_ragged_wrapper_grows_as_it_primes_its_bucket(native):
    workspace = WorkspacePool(CPU, size_mb=1).get("main", 0)
    wrapper = FlashInferRaggedPrefillWrapper(
        workspace, num_qo_heads=2, num_kv_heads=2, head_dim=64, max_num_segments=2, max_total_tokens=16,
        device=CPU, accelerator_graph=True,
    )
    assert wrapper.attn_wrapper.plans == 2
    assert workspace.tensor.numel() >= 5 * MIB


class _Ragged:
    def __init__(self, **config):
        config = dict(num_qo_heads=2, num_kv_heads=2, head_dim=64, workspace_mb=1, **config)
        self.manager = FlashInferRaggedManager(
            device=CPU, dtype=torch.float32, config=RaggedAttentionConfig(**config), key="dit_attn",
        )

    def plan(self, bs: int, tokens: int, slot: int = 0) -> None:
        rids = list(range(bs))
        segments = tuple(Segment(rid, "main", tokens // bs) for rid in rids)
        bucket = BucketKey(graph_walk="gen", cg_key_info=None, bs=bs, num_tokens=tokens)
        self.manager.plan(AttentionStep(segments=segments, causal=False), StepContext(
            request_ids=tuple(rids), graph_walk="gen", slot=slot, capture=True,
            slot_lease=SlotLease(slot=slot, bucket=bucket),
        ))


def test_a_manager_holds_one_buffer_for_a_key_at_its_largest_plan(native):
    """Buckets lead with batch size at capture, so a later one can ask for more than the first."""
    ragged = _Ragged()
    native.asks = MIB // 2
    ragged.plan(bs=4, tokens=4)
    first = native.built[0]
    native.asks = 9 * MIB
    ragged.plan(bs=1, tokens=64)
    assert len(native.built) == 2
    workspaces = ragged.manager._workspaces
    assert list(workspaces._workspaces) == ["main_0"]
    # the decode-sized bucket's wrapper reads the grown buffer, not a second one kept for it
    assert first.workspace is native.built[1].workspace is workspaces.get("main", 0).tensor
    assert first.workspace.numel() >= 9 * MIB


def test_a_manager_refuses_to_move_a_workspace_past_the_seal(native):
    ragged = _Ragged()
    native.asks = MIB // 2
    ragged.plan(bs=1, tokens=8)
    ragged.manager.seal_capture_buffers()
    native.asks = 3 * MIB
    with pytest.raises(RuntimeError, match="dit_attn/main_0.*the workspace_mb dit_attn declares"):
        ragged.plan(bs=2, tokens=16)
    # nor one a key's first wrapper makes after the seal
    with pytest.raises(RuntimeError, match="dit_attn/main_1"):
        ragged.plan(bs=2, tokens=16, slot=1)
    assert ragged.manager._workspaces.get("main", 1).tensor.numel() == MIB


_FLASHINFER_MANAGERS = [
    FlashInferManager, FlashInferCrossManager, FlashInferMLAManager,
    FlashInferRaggedManager, FlashInferRaggedCrossManager, FlashInferRaggedBlockCausalManager,
]


@pytest.mark.parametrize("manager_class", _FLASHINFER_MANAGERS, ids=lambda cls: cls.__name__)
def test_every_flashinfer_manager_seals_its_pool_through_the_base(manager_class):
    """The seal is the base's, so a manager can't hold a pool the engine doesn't seal."""
    assert manager_class.seal_capture_buffers is AttentionResource.seal_capture_buffers
    manager = manager_class.__new__(manager_class)
    manager._workspaces = WorkspacePool(CPU, size_mb=1)
    made = manager._workspaces.get("main", 0)
    manager.seal_capture_buffers()
    assert made.sealed and manager._workspaces.get("main", 1).sealed


def test_a_captured_wrapper_made_after_the_seal_gets_a_sealed_workspace(monkeypatch):
    """`FlashInferManager` as the engine builds it: a key its `_cg_wrapper` first reaches past
    the seal can't grow either."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    kv_config = PagedKVConfig(
        num_layers=1, head_dim=64, num_qo_heads=2, num_kv_heads=2, page_size=16, max_num_pages=4, max_seq_len=64,
    )
    manager = FlashInferManager("kv", CPU, torch.float32, kv_config)
    built = []
    monkeypatch.setattr(flashinfer_mod, "FlashInferPrefillWrapper", lambda **kwargs: built.append(kwargs))
    bucket = BucketKey(graph_walk="prefill", cg_key_info=None, bs=1, num_tokens=16)
    manager._cg_wrapper(SlotLease(slot=0, bucket=bucket), "main", 1, force_prefill=False)
    manager.seal_capture_buffers()
    manager._cg_wrapper(SlotLease(slot=1, bucket=bucket), "main", 1, force_prefill=False)
    workspaces = [kwargs["workspace_buffer"] for kwargs in built]
    assert workspaces == [manager._workspaces.get("main", 0), manager._workspaces.get("main", 1)]
    assert all(workspace.sealed for workspace in workspaces)


def test_the_engine_seals_between_sizing_and_capture(monkeypatch):
    """A growth during sizing moves wrappers whose only graphs were thrown away; one during
    capture would free memory a recorded graph reads."""
    events = []

    class _Runner:
        any_graphs = False
        dropped_buckets = ()

        def __init__(self, **kwargs):
            pass

        def prepare_for_capture(self):
            events.append("prepare")

        def warmup_and_capture(self, planned):
            events.append("capture")

    class _Resource:
        def seal_capture_buffers(self):
            events.append("seal")

        def post_warmup_validate(self):
            pass

    engine = engine_mod.Engine.__new__(engine_mod.Engine)
    engine._device = CPU
    engine._enable_nvtx = False
    engine._resources = {"attn": _Resource()}
    engine._runner = None
    engine._side_stream_nodes = frozenset()
    engine._submodules = {"dit": SimpleNamespace(
        submodule=SimpleNamespace(warmup_compiles=lambda: None, resident_walks=frozenset), resources={},
        joint_comm_group=None, num_slots=2,
        fp32_precision=None, node_dtype=None, lora_pools=(),
    )}
    monkeypatch.setattr(engine_mod, "AcceleratorGraphRunner", _Runner)
    monkeypatch.setattr(engine, "_compile_regions", lambda: None, raising=False)
    monkeypatch.setattr(engine, "_build_piecewise_runners", lambda *args: {}, raising=False)
    monkeypatch.setattr(engine, "_plan_captures", lambda *args: events.append("size"), raising=False)
    monkeypatch.setattr(engine, "_compile_submodules", lambda: None, raising=False)
    engine.warmup()
    assert events == ["prepare", "size", "seal", "capture"]


# ---------------------------------------------------------------------------------------- real planner

needs_flashinfer_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and FlashInfer")
GPU = torch.device("cuda") if torch.cuda.is_available() else CPU
HEADS, HEAD_DIM, SEGMENTS, TOKENS = 24, 128, 4, 4096
ROOMY = 2048 * MIB


def _ragged_gpu(workspace) -> FlashInferRaggedPrefillWrapper:
    return FlashInferRaggedPrefillWrapper(
        workspace, num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, max_num_segments=SEGMENTS,
        max_total_tokens=TOKENS, device=GPU, accelerator_graph=True, backend="fa2",
    )


def _small_workspace() -> Workspace:
    return Workspace(torch.empty(MIB, dtype=torch.uint8, device=GPU), "test")


def _qkv():
    gen = torch.Generator(device=GPU).manual_seed(0)
    return [torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=GPU, generator=gen) for _ in range(3)]


def _layout() -> torch.Tensor:
    return torch.tensor([0, 512, 2048, 2048, TOKENS], dtype=torch.int32)


@needs_flashinfer_gpu
def test_flashinfers_overflow_reads_as_the_bytes_it_needs():
    pytest.importorskip("flashinfer")
    from flashinfer.prefill import get_batch_prefill_module

    workspace = _small_workspace()
    _ragged_gpu(workspace)
    module = get_batch_prefill_module(
        "fa2", torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, HEAD_DIM, HEAD_DIM, 0, False, False, False,
    )
    per_segment = TOKENS // SEGMENTS
    cu = torch.arange(SEGMENTS + 1, dtype=torch.int32) * per_segment
    lens = torch.full((SEGMENTS,), per_segment, dtype=torch.int32)
    need, _ = module.workspace_size(
        workspace.tensor, cu, cu, lens, TOKENS, SEGMENTS, HEADS, HEADS, 1, True, HEAD_DIM, HEAD_DIM,
        False, -1, -1, False, 0, 0,
    )
    # at least what FlashInfer's own query says, and less than a MiB over it
    assert need <= workspace.tensor.numel() < need + MIB


@needs_flashinfer_gpu
def test_a_sealed_buffer_reports_flashinfers_number():
    pytest.importorskip("flashinfer")
    workspace = Workspace(torch.empty(MIB, dtype=torch.uint8, device=GPU), "test", sealed=True)
    with pytest.raises(RuntimeError, match=r"a plan asks for \d+ bytes .*MSTAR_WORKSPACE_BUFFER_MB"):
        _ragged_gpu(workspace)


@needs_flashinfer_gpu
def test_a_gpu_with_no_room_leaves_the_workspace_what_it_held():
    workspace = _small_workspace()
    planner = _Planner(workspace.tensor, 2 * torch.cuda.get_device_properties(GPU).total_memory)
    with pytest.raises(RuntimeError, match=r"the GPU has no room to grow it: CUDA out of memory") as err:
        workspace.plan(planner.plan)
    assert f"and it holds {MIB} bytes" in str(err.value)
    assert workspace.tensor.numel() == MIB and workspace.tensor.device.type == "cuda"


@needs_flashinfer_gpu
def test_a_grown_workspace_attends_as_a_roomy_one_and_a_graph_replays_it():
    pytest.importorskip("flashinfer")
    q, k, v = _qkv()
    grown = _ragged_gpu(_small_workspace())
    roomy = _ragged_gpu(Workspace(torch.empty(ROOMY, dtype=torch.uint8, device=GPU), "test"))
    for wrapper in (grown, roomy):
        wrapper.plan(_layout())
    want = roomy.run(q, k, v)
    assert torch.equal(grown.run(q, k, v), want)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = grown.run(q, k, v)
    out.zero_()
    graph.replay()
    assert torch.equal(out, want)


@needs_flashinfer_gpu
def test_a_captured_paged_wrapper_grows_on_its_first_plan_and_its_later_plans_go_direct(monkeypatch):
    pytest.importorskip("flashinfer")
    # the direct path reads the workspace off the planner arguments, whatever the installed version
    monkeypatch.setattr(wrappers_mod, "_direct_plan_supported", lambda: True)
    workspace = _small_workspace()
    # a kv length that takes the split path; a one-row bucket of 4096 tokens at 24 heads wants tens of MiB
    wrapper = FlashInferPrefillWrapper(
        workspace, num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, page_size=128, batch_size=1,
        max_total_tokens=TOKENS, max_num_pages=64, device=GPU, accelerator_graph=True, backend="fa2",
    )
    layout = dict(
        qo_indptr=torch.tensor([0, TOKENS], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 32], dtype=torch.int32),
        paged_kv_indices=torch.arange(32, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([128], dtype=torch.int32),
        kv_lens=torch.tensor([TOKENS], dtype=torch.int32),
    )
    wrapper.plan(dtype=torch.bfloat16, **layout)
    size = workspace.tensor.numel()
    assert size > MIB
    assert wrapper._planner_args is not None and wrapper._planner_args[0] is workspace.tensor
    plans = []
    real_direct = wrapper._plan_direct
    monkeypatch.setattr(wrapper, "_plan_direct", lambda *args: (plans.append(1), real_direct(*args)))
    wrapper.plan(dtype=torch.bfloat16, **layout)
    assert plans and workspace.tensor.numel() == size
    torch.cuda.synchronize()


def _kibibyte() -> Workspace:
    return Workspace(torch.empty(1024, dtype=torch.uint8, device=GPU), "test")


def _paged_rows() -> dict:
    return dict(
        paged_kv_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([16, 16], dtype=torch.int32),
    )


def _plan_prefill(workspaces):
    wrapper = FlashInferPrefillWrapper(
        workspaces[0], num_qo_heads=8, num_kv_heads=8, head_dim=64, page_size=16, batch_size=2, max_total_tokens=32,
        max_num_pages=8, device=GPU, accelerator_graph=True, backend="fa2",
    )
    return lambda: wrapper.plan(qo_indptr=torch.tensor([0, 16, 32], dtype=torch.int32), **_paged_rows())


def _plan_decode(workspaces):
    wrapper = FlashInferDecodeWrapper(
        workspaces[0], num_qo_heads=8, num_kv_heads=8, head_dim=64, page_size=16, batch_size=2,
        max_num_pages=8, device=GPU, accelerator_graph=True, backend="fa2",
    )
    return lambda: wrapper.plan(**_paged_rows())


def _plan_mixed(workspaces):
    wrapper = FlashInferMixedWrapper(
        decode_workspace=workspaces[0], prefill_workspace=workspaces[1], num_qo_heads=8, num_kv_heads=8, head_dim=64,
        page_size=16, batch_size=2, max_total_tokens=32, max_num_pages=8, max_prefill_rows=1, device=GPU,
        backend="fa2",
    )
    return lambda: wrapper.plan(qo_indptr=torch.tensor([0, 1, 32], dtype=torch.int32), **_paged_rows())


def _plan_ragged(workspaces):
    wrapper = FlashInferRaggedPrefillWrapper(
        workspaces[0], num_qo_heads=8, num_kv_heads=8, head_dim=64, max_num_segments=2, max_total_tokens=32,
        device=GPU, accelerator_graph=True, backend="fa2",
    )
    return lambda: wrapper.plan(torch.tensor([0, 8, 32], dtype=torch.int32))


def _plan_ragged_cross(workspaces):
    wrapper = FlashInferRaggedCrossPrefillWrapper(
        workspaces[0], num_qo_heads=8, num_kv_heads=8, head_dim=64, max_num_segments=2, max_total_tokens=32,
        device=GPU, accelerator_graph=True, backend="fa2",
    )
    return lambda: wrapper.plan(
        torch.tensor([0, 8, 16], dtype=torch.int32), torch.tensor([0, 16, 32], dtype=torch.int32),
    )


def _plan_block_causal(workspaces):
    wrapper = RaggedBlockCausalWrapper(
        workspaces[0], num_qo_heads=8, num_kv_heads=8, head_dim=64, block_size=8, max_num_segments=2,
        max_total_tokens=32, device=GPU, accelerator_graph=True, backend="fa2",
    )
    return lambda: wrapper.plan([16, 16])


def _plan_mla(workspaces):
    wrapper = FlashInferMLAWrapper(
        workspaces[0], num_qo_heads=8, kv_lora_rank=512, qk_rope_head_dim=64, page_size=64, sm_scale=192 ** -0.5,
        batch_size=2, max_num_pages=8, device=GPU, use_cuda_graph=True,
    )
    return lambda: wrapper.plan(
        qo_indptr=torch.tensor([0, 16, 32], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 2], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([16, 16], dtype=torch.int32),
    )


@needs_flashinfer_gpu
@pytest.mark.parametrize("build, count", [
    (_plan_prefill, 1), (_plan_decode, 1), (_plan_mixed, 2), (_plan_ragged, 1), (_plan_ragged_cross, 1),
    (_plan_block_causal, 1), (_plan_mla, 1),
], ids=lambda value: getattr(value, "__name__", None) if callable(value) else "")
def test_every_wrapper_grows_on_the_real_planner_and_its_next_plan_fits(build, count):
    pytest.importorskip("flashinfer")
    workspaces = [_kibibyte() for _ in range(count)]
    plan = build(workspaces)
    plan()
    sizes = [workspace.tensor.numel() for workspace in workspaces]
    assert all(size > 1024 for size in sizes)
    # under a graph a plan's need doesn't move with its layout, so nothing later grows it
    plan()
    assert [workspace.tensor.numel() for workspace in workspaces] == sizes
    torch.cuda.synchronize()
