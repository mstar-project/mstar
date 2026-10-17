"""A packed speculative step pre-plans from its rows' spans measured ahead.

Before ``prepare_inputs`` a packed step has no token count, so its rows are
measured: a row in hand from its inputs, a row continuing from the step in
flight from the submodule's estimate, which starts from its span there. The step leases and declares from
those spans; once the rows are prepared, a stage they do not match (a cached
prefix cut a prompt, a row left) is dropped with its lease, so the step plans
inline rather than replaying a plan for other spans.
"""

from types import SimpleNamespace

import torch

from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.step import StepContext
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_5.config import LLM_MIXED
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, InputSeqLenInfo, NodeSubmodule


class _Submodule(NodeSubmodule):
    """Decode rows estimate one token, others the base rule; a prompt
    measures its ids."""

    def __init__(self):
        pass

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def speculative_input_seq_len(self, graph_walk, fwd_info, prev_seq_len):
        if graph_walk == "decode":
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)

    def get_input_sequence_len(self, graph_walk, fwd_info, inputs):
        if "text_inputs" not in inputs:
            return None
        return InputSeqLenInfo(inputs["text_inputs"][0].shape[0])


def _engine(submodule=None, packed=True):
    released = []
    runner = SimpleNamespace(
        is_packed=lambda lease: packed,
        release=lambda lease, bs: released.append(lease),
    )
    fake = SimpleNamespace(
        _enable_nvtx=False,
        _submodules={"LLM": SimpleNamespace(
            submodule=submodule or _Submodule(), cuda_graph_runner=runner,
        )},
        _runner=SimpleNamespace(clear_preplan=lambda: None),
    )
    for name in (
        "measure_ahead_spans", "preplan_is_stale", "settle_preplan",
        "reset_pre_plan_for_batch", "_ahead_lease", "_drop_ahead_lease",
    ):
        setattr(fake, name, getattr(Engine, name).__get__(fake))
    return fake, released


def _batch(rids, walks, inputs=None):
    ctx = StepContext(
        request_ids=rids, graph_walk=LLM_MIXED, slot=0, capture=False,
        request_walks=dict(zip(rids, walks, strict=True)),
    )
    return SimpleNamespace(
        node_name="LLM", request_ids=rids, step_context=ctx,
        per_request_info={rid: None for rid in rids},
        per_request_input_tensors=inputs or {},
        inputs=None, ahead_spans=None, preplanned_spans=None,
        preplanned_rids=None, preplan_event=None, step=None,
    )


def test_a_continuing_row_takes_the_estimate_and_a_new_prompt_its_ids():
    engine, _ = _engine()
    batch = _batch([1, 2], ["decode", "prefill_text"], {2: {"text_inputs": [torch.zeros(9)]}})

    # row 1 ran its prompt's last 40 tokens in the step in flight
    engine.measure_ahead_spans(batch, prev_seq_lens={1: 40})

    assert batch.ahead_spans == (1, 9)


def test_a_continuing_row_repeats_its_last_span_on_this_node():
    engine, _ = _engine()
    batch = _batch([1, 2], ["prefill_text", "decode"])

    engine.measure_ahead_spans(batch, prev_seq_lens={1: 128, 2: 1})

    assert batch.ahead_spans == (128, 1)


def test_a_row_from_another_node_has_no_estimate():
    engine, _ = _engine()
    batch = _batch([1, 2], ["prefill_text", "decode"])

    engine.measure_ahead_spans(batch, prev_seq_lens={1: -1, 2: 1})

    assert batch.ahead_spans is None


def test_a_prepared_batch_reports_each_rows_tokens():
    batch = ExecutingBatch(
        node_name="LLM", per_request_info={},
        step_context=StepContext(request_ids=[1, 2], graph_walk=LLM_MIXED, slot=0, capture=False),
    )
    assert batch.seq_len_of(1) == -1
    batch.register_prepare_batch([ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=7)])
    batch.prepared_rids = (1, 2)

    assert (batch.seq_len_of(1), batch.seq_len_of(2), batch.seq_len_of(3)) == (1, 7, -1)


def test_a_prepared_span_that_differs_drops_the_stage_and_its_lease():
    engine, released = _engine()
    batch = _batch([1, 2], ["decode", "prefill_text"])
    batch.ahead_spans = batch.preplanned_spans = (1, 9)
    batch.preplanned_rids = (1, 2)
    batch.step_context.slot_lease = lease = object()
    # a cached prefix cut the prompt to 3 tokens
    batch.inputs = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=3)]

    engine.settle_preplan(batch)

    assert batch.preplanned_rids is None and batch.preplanned_spans is None
    assert batch.step_context.slot_lease is None and released == [lease]


def test_matching_spans_keep_the_stage():
    engine, released = _engine()
    batch = _batch([1, 2], ["decode", "prefill_text"])
    batch.ahead_spans = batch.preplanned_spans = (1, 9)
    batch.preplanned_rids = (1, 2)
    batch.step_context.slot_lease = object()
    batch.inputs = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=9)]

    engine.settle_preplan(batch)

    assert batch.preplanned_rids == (1, 2) and released == []


def test_a_lease_taken_on_spans_that_no_longer_hold_goes_without_a_stage():
    # the pre-plan never staged, but the packed lease was picked for 10 tokens
    engine, released = _engine()
    batch = _batch([1, 2], ["decode", "prefill_text"])
    batch.ahead_spans = (1, 9)
    batch.step_context.slot_lease = lease = object()
    batch.inputs = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_seq_len=30)]

    engine.settle_preplan(batch)

    assert batch.step_context.slot_lease is None and released == [lease]


def test_qwen_declares_a_mixed_step_from_spans_as_from_its_rows():
    llm = object.__new__(LLMSubmodule)
    real = [
        ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([3]), graph_walk="decode"),
        ARNodeInputs(input_seq_len=4, input_ids=torch.arange(4), graph_walk="prefill_text"),
    ]
    ahead = llm.declare_inputs_from_spans(LLM_MIXED, (1, 4), ("decode", "prefill_text"), (None, None))

    from_rows = llm.declare_step(LLM_MIXED, [0, 1], real)
    from_spans = llm.declare_step(LLM_MIXED, [0, 1], ahead)

    assert from_spans.segments == from_rows.segments
    assert llm.speculative_input_seq_len("decode", None, 40) == 1
    assert llm.speculative_input_seq_len("prefill_text", None, 64) == 64


def test_orpheus_declares_a_mixed_step_from_spans_as_from_its_rows():
    llm = object.__new__(OrpheusLLMSubmodule)
    real = [
        ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([3]), graph_walk="decode"),
        ARNodeInputs(input_seq_len=5, input_ids=torch.arange(5), graph_walk="prefill"),
    ]
    ahead = llm.declare_inputs_from_spans("mixed", (1, 5), ("decode", "prefill"), (None, None))

    assert llm.declare_step("mixed", [0, 1], ahead).segments == (
        llm.declare_step("mixed", [0, 1], real).segments
    )
    assert llm.speculative_input_seq_len("decode", None, 12) == 1
