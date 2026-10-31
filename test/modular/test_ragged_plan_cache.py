"""Skipping a plan a wrapper already holds, for every manager on ``_RaggedPrefillBase``.

CPU only: FlashInfer is a stand-in that counts native plans, and FA3's wrappers
never reach their kernel here. What is counted is ``_plan_kernel``, the step that
costs a device kernel and a stream sync.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.dense import DenseAttentionManager
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.attn.ragged.dense import DenseRaggedCrossManager, DenseRaggedManager
from mstar.engine.resources.attn.ragged.dense_wrappers import DensePrefixedPrefillWrapper
from mstar.engine.resources.attn.ragged.flashinfer import (
    FlashInferRaggedBlockCausalManager,
    FlashInferRaggedCrossManager,
    FlashInferRaggedManager,
)
from mstar.engine.resources.attn.ragged.wrappers import (
    FlashInferPrefixedPrefillWrapper,
    FlashInferRaggedPrefillWrapper,
    _RaggedPrefillBase,
)
from mstar.engine.resources.base import BucketKey
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView
from mstar.engine.resources.step import Segment, SlotLease, StepContext

DEVICE = torch.device("cpu")
HEADS, HEAD_DIM = 2, 64


class _NativeWrapper:
    """FlashInfer's ragged wrapper as far as a plan goes."""

    def __init__(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def fake_flashinfer(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=_NativeWrapper),
    )


@pytest.fixture
def plans(monkeypatch) -> list[tuple[_RaggedPrefillBase, list[int], list[int]]]:
    """Every kernel plan: (wrapper, query layout, key layout)."""
    calls = []
    for cls in {c for c in _subclasses(_RaggedPrefillBase) if "_plan_kernel" in vars(c)}:
        real = cls._plan_kernel

        def counting(self, cu, kv_cu, total_tokens, causal, _real=real):
            calls.append((self, cu.tolist(), kv_cu.tolist()))
            return _real(self, cu, kv_cu, total_tokens, causal)

        monkeypatch.setattr(cls, "_plan_kernel", counting)
    return calls


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _ragged_config() -> RaggedAttentionConfig:
    return RaggedAttentionConfig(num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, workspace_mb=1)


def _ctx(rids, slot: int = 0, lease: SlotLease | None = None, is_preplan: bool = False, **kwargs) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="gen", slot=slot, capture=False,
        slot_lease=lease, is_preplan=is_preplan, **kwargs,
    )


class _Self:
    """A ragged self-attention manager, planned from ``(span, ...)``."""

    def __init__(self, manager_class):
        self.manager = manager_class(device=DEVICE, dtype=torch.float32, config=_ragged_config())

    def plan(self, spans, causal=False, **ctx):
        segments = tuple(Segment(f"r{i}", "main", span) for i, span in enumerate(spans))
        self.manager.plan(AttentionStep(segments=segments, causal=causal), _ctx(segments_rids(spans), **ctx))

    def wrapper(self):
        return self.manager._current_plan_states["main"]


class _Cross:
    """A ragged cross-attention manager; each query span attends a key span twice its size."""

    def __init__(self, manager_class):
        self.manager = manager_class(device=DEVICE, dtype=torch.float32, config=_ragged_config())

    def plan(self, spans, causal=False, **ctx):
        assert not causal
        segments = tuple(
            Segment(f"r{i}", label, span * scale)
            for label, scale in (("video", 1), ("text", 2))
            for i, span in enumerate(spans)
        )
        step = RaggedCrossAttentionStep(segments=segments, pairs=(("video", "text"),))
        self.manager.plan(step, _ctx(segments_rids(spans), **ctx))

    def wrapper(self):
        return self.manager._current_plan_states[cross_label("video", "text")]


class _Dense:
    """The KV-backed dense manager; each query span follows a frozen prefix of 3."""

    def __init__(self, wrapper_class):
        config = PagedKVConfig(
            num_layers=1, num_kv_heads=HEADS, head_dim=HEAD_DIM, max_seq_len=64, max_num_pages=8, page_size=4,
        )
        self.manager = DenseAttentionManager(
            kv_cache="kv", device=DEVICE, dtype=torch.float32, kv_config=config, wrapper_class=wrapper_class,
        )

    def plan(self, spans, causal=False, slot=0):
        views = [
            SequenceView(request_id=f"r{i}", label="main", page_idxs=[i], length=3 + span, to_compute=span)
            for i, span in enumerate(spans)
        ]
        outputs = {"kv": {"main": KVPlanOutput(cpu_indptrs=None, cuda_indptrs=None, views=views)}}
        self.manager.plan(AttentionStep(causal=causal), _ctx(segments_rids(spans), slot=slot, plan_results=outputs))

    def wrapper(self):
        return self.manager._current_plans["main"].wrapper


def segments_rids(spans) -> list[str]:
    return [f"r{i}" for i in range(len(spans))]


EVERY_MANAGER = [
    pytest.param(lambda: _Self(FlashInferRaggedManager), id="flashinfer_self"),
    pytest.param(lambda: _Self(DenseRaggedManager), id="fa3_self"),
    pytest.param(lambda: _Cross(FlashInferRaggedCrossManager), id="flashinfer_cross"),
    pytest.param(lambda: _Cross(DenseRaggedCrossManager), id="fa3_cross"),
    pytest.param(lambda: _Dense(DensePrefixedPrefillWrapper), id="fa3_dense"),
    pytest.param(lambda: _Dense(FlashInferPrefixedPrefillWrapper), id="flashinfer_dense"),
]
CAPTURED_MANAGERS = EVERY_MANAGER[:4]


@pytest.mark.parametrize("build", EVERY_MANAGER)
def test_a_layout_the_wrapper_holds_is_not_replanned(build, plans):
    res = build()
    res.plan([8, 4])
    res.plan([8, 4])
    assert len(plans) == 1
    assert res.wrapper() is plans[0][0], "the skipped plan still hands the layer the planned wrapper"


@pytest.mark.parametrize("build", EVERY_MANAGER)
def test_another_layout_plans_again(build, plans):
    res = build()
    res.plan([8, 4])
    res.plan([8, 8])
    assert len(plans) == 2 and plans[1][1] != plans[0][1]
    res.plan([8, 8])
    assert len(plans) == 2
    res.plan([8, 4])
    assert len(plans) == 3, "only the last plan is held"


@pytest.mark.parametrize("build", EVERY_MANAGER)
def test_each_slot_owns_its_schedule(build, plans):
    res = build()
    res.plan([8, 4], slot=0)
    res.plan([8, 4], slot=1)
    assert len(plans) == 2 and plans[0][0] is not plans[1][0], "a slot's first step plans its own wrapper"
    res.plan([8, 4], slot=0)
    res.plan([8, 4], slot=1)
    assert len(plans) == 2
    # slot 0 moving on says nothing about what slot 1 holds
    res.plan([4, 4], slot=0)
    res.plan([8, 4], slot=1)
    assert len(plans) == 3
    res.plan([4, 4], slot=1)
    assert len(plans) == 4


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _Self(FlashInferRaggedManager), id="flashinfer_self"),
        pytest.param(lambda: _Self(DenseRaggedManager), id="fa3_self"),
        pytest.param(lambda: _Dense(DensePrefixedPrefillWrapper), id="fa3_dense"),
    ],
)
def test_causal_is_part_of_the_layout(build, plans):
    res = build()
    res.plan([8], causal=False)
    res.plan([8], causal=True)
    assert len(plans) == 2


def _lease(res, slot: int = 0, bs: int = 2, tokens: int = 64) -> SlotLease:
    return SlotLease(slot=slot, bucket=BucketKey(graph_walk="gen", bs=bs, num_tokens=tokens))


@pytest.mark.parametrize("build", CAPTURED_MANAGERS)
def test_a_captured_wrapper_keeps_its_static_layout_on_a_skip(build, plans):
    res = build()
    res.plan([8, 4], lease=_lease(res))
    wrapper = res.wrapper()
    assert wrapper.accelerator_graph
    primed = sum(1 for w, *_ in plans if w is wrapper) - 1
    res.plan([8, 4], lease=_lease(res))
    assert sum(1 for w, *_ in plans if w is wrapper) == primed + 1
    assert wrapper.num_segments == 2


@pytest.mark.parametrize("build", CAPTURED_MANAGERS[:2])
def test_padding_to_the_same_layout_is_still_another_plan(build, plans):
    """``[0, 8]`` and ``[0, 8, 8]`` pad to one static layout and differ in
    ``num_segments``, which the plan sets."""
    res = build()
    res.plan([8, 0], lease=_lease(res))
    assert res.wrapper().num_segments == 2
    res.plan([8], lease=_lease(res))
    assert res.wrapper().num_segments == 1


@pytest.mark.parametrize("build", CAPTURED_MANAGERS)
def test_a_promoted_preplan_is_kept(build, plans):
    res = build()
    res.plan([8, 4], lease=_lease(res), is_preplan=True)
    wrapper = res.manager._preplan_states[next(iter(res.manager._preplan_states))]
    planned = len(plans)
    res.plan([8, 4], lease=_lease(res))
    assert res.wrapper() is wrapper and len(plans) == planned
    # the next step's pre-plan onto this wrapper finds its layout in place
    res.plan([8, 4], lease=_lease(res), is_preplan=True)
    assert len(plans) == planned


@pytest.mark.parametrize("build", CAPTURED_MANAGERS)
def test_a_dropped_preplan_is_staged_again(build, plans):
    """Its H2Ds were on the plan stream, and the step that replaces it waits on
    no event from that stream."""
    res = build()
    res.plan([8, 4], lease=_lease(res), is_preplan=True)
    res.manager.clear_preplan()
    planned = len(plans)
    res.plan([8, 4], lease=_lease(res))
    assert len(plans) == planned + 1


def test_a_dropped_block_causal_preplan_has_nothing_to_forget(monkeypatch):
    """Its wrapper plans every step, and shares the manager's rollback."""
    monkeypatch.setattr(RaggedBlockCausalWrapper, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(RaggedBlockCausalWrapper, "plan", lambda self, spans: None)
    manager = FlashInferRaggedBlockCausalManager(
        device=DEVICE, dtype=torch.float32, config=_ragged_config(), block_size=4,
    )
    segments = (Segment("r0", "main", 8),)
    manager.plan(
        AttentionStep(segments=segments, causal=False),
        _ctx(["r0"], lease=_lease(None, bs=1), is_preplan=True),
    )
    manager.clear_preplan()
    assert manager._preplan_states == {}


def test_a_plan_that_raises_leaves_nothing_to_reuse(plans):
    """It may have overwritten the schedule of the layout before it."""
    res = _Self(FlashInferRaggedManager)
    res.plan([8, 4])
    wrapper = res.wrapper()

    def fail(cu, kv_cu, total_tokens, causal):
        raise RuntimeError("plan failed partway")

    wrapper._plan_kernel = fail
    with pytest.raises(RuntimeError, match="plan failed"):
        res.plan([4, 4])
    del wrapper._plan_kernel
    res.plan([8, 4])
    assert len(plans) == 2


def test_a_device_layout_plans_without_being_read_back(monkeypatch):
    """Comparing it would read it on the host: a stream sync per plan."""
    kernel = []
    monkeypatch.setattr(
        FlashInferRaggedPrefillWrapper, "_plan_kernel",
        lambda self, cu, kv_cu, total_tokens, causal: kernel.append(cu),
    )
    wrapper = FlashInferRaggedPrefillWrapper(
        torch.empty(8, dtype=torch.uint8), HEADS, HEADS, HEAD_DIM, device=DEVICE,
    )
    layout = torch.tensor([0, 8], dtype=torch.int32, device="meta")  # .tolist() raises on meta
    wrapper.plan(layout)
    wrapper.plan(layout)
    assert len(kernel) == 2
