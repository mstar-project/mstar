"""A capture that writes a resource more tokens than its inputs (an encoder
context written whole at prefill) names it in ``resource_write_tokens``, and
the engine sizes that resource's capture buffers by those writes.

The decoder's prefill bucket of a Whisper-like model carries 4 prompt tokens a
row and writes a 1500-token context a row into the context cache; sized by the
bucket's input tokens, the cache's static write addressing holds 4 a row.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.model.submodule_base import NodeInputs

PROMPT, CONTEXT = 4, 1500


class _Sized(Resource):
    """Records the length each sweep handed it."""

    def __init__(self):
        self.lengths: list[int] = []

    @classmethod
    def build(cls, spec, info):
        return cls()

    def build_accelerator_graph_buffers(self, slots, max_bs, max_seq_len):
        self.lengths.append(max_seq_len)


def test_only_the_named_resource_is_sized_past_the_input_tokens():
    ctx, kv = _Sized(), _Sized()
    runner = StepRunner({"ctx": ctx, "kv": kv})

    runner.build_accelerator_graph_buffers(
        slots=[], max_bs=16, max_seq_len=64, write_tokens={"ctx": 16 * CONTEXT},
    )

    assert ctx.lengths == [16 * CONTEXT]
    assert kv.lengths == [64]


def test_a_write_below_the_input_tokens_never_shrinks_a_resource():
    ctx = _Sized()
    StepRunner({"ctx": ctx}).build_accelerator_graph_buffers(
        slots=[], max_bs=4, max_seq_len=64, write_tokens={"ctx": 8},
    )
    assert ctx.lengths == [64]


def _prefill_config(sizes) -> BatchedAcceleratorGraphConfig:
    return BatchedAcceleratorGraphConfig(
        capture_graph_walk="prefill",
        single_request_inputs=NodeInputs(input_seq_len=PROMPT),
        capture_batch_sizes=sizes,
        resource_write_tokens={"ctx": CONTEXT},
    )


def test_the_runner_sizes_a_written_resource_by_its_largest_bucket():
    """The real `prepare_for_capture` over a decode config and a prefill
    config that writes a context per row."""
    decode = BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode",
        single_request_inputs=NodeInputs(input_seq_len=1),
        capture_batch_sizes=[1, 2, 4, 8, 16, 32, 64],
    )
    calls = []
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._capture_configs = [decode, _prefill_config([1, 2, 4, 8, 16])]
    runner._num_slots = 2
    runner._prepared_slot_specs = None
    runner._resources = {}
    runner._submodule_name = "decoder"
    runner._submodule = SimpleNamespace(declare_step=lambda **kwargs: None)
    runner._dummy_rows = SimpleNamespace(handles=lambda key, bs: list(range(-bs, 0)))
    runner._step_runner = SimpleNamespace(
        build_accelerator_graph_buffers=lambda slots, **kwargs: calls.append(kwargs),
    )
    runner.max_bs = 64

    runner.prepare_for_capture()

    (call,) = calls
    # the input tokens of the widest bucket: 64 decode rows, 16 prefill rows of 4
    assert call["max_seq_len"] == 64
    assert call["write_tokens"] == {"ctx": 16 * CONTEXT}


def test_write_tokens_scale_with_the_bucket():
    config = _prefill_config([1, 2])
    assert config.write_tokens("ctx", 2) == 2 * CONTEXT
    assert config.write_tokens("other", 2) == 0


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def cleanup(self):
        pass


@pytest.fixture
def context_cache(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=CONTEXT,
            max_num_pages=1 + 2 * 12, page_size=128,
        ),
        name="ctx", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _capture(kv: KVManager, bs: int):
    """A prefill bucket's capture: each row writes its whole context."""
    bucket = BucketKey(graph_walk="prefill", bs=bs, num_tokens=PROMPT * bs)
    rids = tuple(range(-1, -bs - 1, -1))
    for rid in rids:
        kv.ingest_capture_row(rid)
    step = KVStep(segments=tuple(Segment(rid, "main", CONTEXT) for rid in rids))
    ctx = StepContext(
        request_ids=rids, graph_walk="prefill", slot=0, capture=True,
        slot_lease=SlotLease(slot=0, bucket=bucket),
    )
    assert kv.admit(step, ctx).ok
    return step, ctx


def test_a_context_cache_sized_by_its_writes_stages_the_prefill(context_cache):
    kv = context_cache
    kv.build_accelerator_graph_buffers([], max_bs=2, max_seq_len=2 * CONTEXT)
    step, ctx = _capture(kv, bs=2)

    kv.plan(step, ctx)

    state = kv._current_plan_states["main"]
    assert state.total_tokens == 2 * CONTEXT
    assert set(state.token_to_page[: 2 * CONTEXT].tolist()) == {SINK_PAGE}


def test_a_context_cache_sized_by_the_input_tokens_says_so(context_cache):
    kv = context_cache
    kv.build_accelerator_graph_buffers([], max_bs=2, max_seq_len=2 * PROMPT)
    step, ctx = _capture(kv, bs=2)

    with pytest.raises(RuntimeError, match="resource_write_tokens"):
        kv.plan(step, ctx)
