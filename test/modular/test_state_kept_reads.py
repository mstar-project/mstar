"""A read of every entry the row keeps plus this step's adds, and a cross read over a kept context.

Pi0.5's prefix, V-JEPA 2's rollout and Canary's context all plan non-causal attention over a kept pool. Nothing in a
kernel notices a flag that's wrong, or a context read under the wrong label: the outputs just move. A cross read's plans
are keyed by the query side's label, and both models that read one today use one label for both sides, so a slip there
would pass every run until a model gives its context a label of its own. Here the context's label differs from the
query side's.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.attn.config import AttentionStep, CrossAttentionConfig, CrossAttentionSpec
from mstar.engine.resources.attn.cross import FlashInferCrossManager
from mstar.engine.resources.attn.dense import DenseAttentionManager
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.ragged.base import _RaggedCrossManagerBase
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedCrossAttentionStep
from mstar.engine.resources.attn.ragged.wrappers import PrefixedPrefillWrapper
from mstar.engine.resources.kv import manager as kv_manager_module
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Block,
    Causal,
    KVEntry,
    PagedNHD,
    Positions,
    Read,
    Value,
    Window,
    at_last_token,
)
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.engine.state.calls import ArrayHandle
from mstar.engine.state.contract import Entry
from mstar.engine.state.within_step import WithinStepHandle
from mstar.model.components.distributed.attention import ParallelCrossAttention

state_op_module = sys.modules["mstar.engine.state.state_op"]

HEADS, DIM = 2, 8
LAYERS = frozenset({0, 1})
CONTEXT = "context"
ENTRY = KVEntry(heads=HEADS, head_dim=DIM)


class _StubKV:
    """Records writes and hands back a named view per layer."""

    def __init__(self, device="cpu"):
        self.writes = []
        self.kv_cache = SimpleNamespace(layout=KVLayout.NHD, dtype=torch.bfloat16, device=torch.device(device))

    def layer_view(self, i):
        return f"view{i}"

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.writes.append((layer_idx, label, k.shape[0]))

    def opens_label(self, rid, label):
        return True


class _StubSelfRead:
    """A self read over the pages it is handed."""

    requires_kv_write = True

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return q

    def select_last_hidden(self, x, label="main"):
        return x


class _StubCross:
    """What the cross resource exposes: ``run`` with no k or v, no ``requires_kv_write``, no ``select_last_hidden``."""

    def __init__(self):
        self.runs = []

    def run(self, q, label=None, kv_cache_layer=None):
        self.runs.append((label, kv_cache_layer))
        return q


def _self_array() -> Array:
    return Array("kv", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=64, positions=Positions(),
                 reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))})


def _context_array(**overrides) -> Array:
    return Array("context", **{
        "nodes": {"n"}, "layers": LAYERS, "entry": ENTRY, "max_len": 32, "labels": {CONTEXT},
        "reads": {"cross": Read(mask=Bidirectional(), op=Attend(q_heads=HEADS), queries="kv")}, **overrides,
    })


def _bound(device="cpu") -> dict[str, ArrayHandle]:
    self_array = _self_array()
    return {
        "kv": ArrayHandle(self_array, {"kv": _StubKV(), "attn": _StubSelfRead(), "rope": object()}, "rope"),
        "context": ArrayHandle(
            _context_array(), {"context": _StubKV(device), "cross": _StubCross()}, None, queried=self_array,
        ),
    }


def _ctx(capture=False, rows=("a", "b")) -> StepContext:
    return StepContext(request_ids=rows, graph_walk="w", slot=0, capture=capture)


def _opened(handles, query_spans=(1, 1), context_spans=(0, 0), context_rows="ab", capture=False) -> SubmoduleStep:
    step = state_steps(SubmoduleStep(
        segments=[Segment(rid, "main", span) for rid, span in zip("ab", query_spans, strict=True)],
        state={"kv": ArrayStep(), "context": ArrayStep(segments=tuple(
            Segment(rid, CONTEXT, span) for rid, span in zip(context_rows, context_spans, strict=True)
        ))},
    ), handles)
    step.set_ctx(_ctx(capture))
    step.begin_calls()
    return step


def _add_and_read(handle, layer, tokens=3) -> None:
    q = k = v = torch.zeros(tokens, HEADS, DIM)
    handle.add(layer, k=k, v=v)
    handle.read(layer, q)


# Which flag each mask plans


@pytest.mark.parametrize("mask, causal", [(Causal(), True), (Window(4), True), (Bidirectional(), False)])
def test_a_kept_reads_mask_picks_the_steps_flag(mask, causal):
    array = Array("kv", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=64,
                  reads={"attn": Read(mask=mask, op=Attend(q_heads=HEADS))})
    handle = ArrayHandle(array, {"kv": _StubKV(), "attn": _StubSelfRead()}, None)
    step = state_steps(SubmoduleStep(segments=[Segment("a", "main", 1)], state={"kv": ArrayStep()}), {"kv": handle})
    assert step.steps["attn"].causal is causal, f"{mask} plans the other flag"


def test_a_kept_cross_read_plans_over_its_arrays_segments_with_no_causal_flag():
    handles = _bound()
    context = (Segment("a", CONTEXT, 13), Segment("b", CONTEXT, 7))
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 9), Segment("b", "main", 9)],
        state={"kv": ArrayStep(), "context": ArrayStep(segments=context)},
    ), handles)
    assert step.steps["cross"] == AttentionStep(segments=context, causal=False)
    assert step.steps["context"] == KVStep(segments=context)
    assert step.steps["attn"].causal, "the self read's flag follows its own mask"


# The specs a kept cross read builds


def test_a_kept_cross_read_builds_todays_spec_over_its_arrays_one_label():
    specs = {spec.resource_key: spec for spec in build_specs([_self_array(), _context_array()])}
    assert specs["cross"] == CrossAttentionSpec(
        resource_key="cross", nodes={"n"},
        config=CrossAttentionConfig(kv_cache="context", query_kv_cache="kv", context_label=CONTEXT),
    )
    assert isinstance(specs["context"], KVSpec) and specs["context"].config.num_qo_heads == HEADS


@pytest.mark.parametrize("arrays, cause", [
    ([_self_array(), _context_array(labels={CONTEXT, "other"})], "a cross read's array declares one label"),
    (
        [_self_array(), _context_array(
            reads={
                "cross": Read(mask=Bidirectional(), op=Attend(q_heads=HEADS), queries="kv", layers={0}),
                "own": Read(mask=Causal(), op=Attend(q_heads=HEADS), layers={1}),
            },
        )],
        "a cross read's array takes no other read yet",
    ),
    (
        [
            Array("kv", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=0,
                  reads={"attn": Read(mask=Block(), op=Attend(q_heads=HEADS))}),
            _context_array(),
        ],
        "reads the queries of 'kv', which isn't an append-only array that keeps entries",
    ),
    (
        [
            Value("kv", nodes={"n"}, layers=LAYERS, entry=Entry({}, host={"sent": bool})),
            _context_array(),
        ],
        "reads the queries of 'kv', which isn't an append-only array that keeps entries",
    ),
], ids=["two_labels", "second_read", "queries_over_max_len_0", "queries_over_a_value"])
def test_a_kept_cross_read_with_no_layout_yet_is_refused_by_name(arrays, cause):
    with pytest.raises(ValueError, match=f"Array 'context': no layout yet: .*{cause}"):
        build_specs(arrays)


def test_a_kept_cross_read_takes_only_decode_tensor_cores():
    arrays = [_self_array(), _context_array()]
    with pytest.raises(ValueError, match=r"kernel_settings\['cross'\] sets \['decode_fmha'\]; it may set only "
                                         r"\['decode_tensor_cores'\]"):
        build_specs(arrays, kernel_settings={"cross": {"decode_fmha": True}})
    built = build_specs(arrays, kernel_settings={"cross": {"decode_tensor_cores": False}})
    assert next(spec for spec in built if spec.resource_key == "cross").config.decode_tensor_cores is False
    built = build_specs(arrays, kernel_settings={"context": {"max_num_pages": 9}})
    assert next(spec for spec in built if spec.resource_key == "context").config.max_num_pages == 9


def test_a_max_len_0_read_over_a_kept_array_plans_todays_cross_step():
    """The other form of a kept cross read: the queries come from the kept array, and the context stays per step."""
    chunk = _self_array()
    text = Array("text", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=0, labels={"text"},
                 reads={"text_attn": Read(mask=Block(), op=Attend(q_heads=HEADS), queries="kv")})
    handles = {
        "kv": ArrayHandle(chunk, {"kv": _StubKV(), "attn": _StubSelfRead(), "rope": object()}, "rope"),
        "text": WithinStepHandle(text, {"text_attn": _StubCross()}, None, resource_key="text_attn", queried=chunk),
    }
    queries = [Segment("a", "main", 30), Segment("b", "main", 20)]
    context = (Segment("a", "text", 512), Segment("b", "text", 512))
    step = state_steps(SubmoduleStep(
        segments=queries, state={"kv": ArrayStep(), "text": ArrayStep(segments=context)},
    ), handles)
    assert step.steps["text_attn"] == RaggedCrossAttentionStep(segments=(*queries, *context), pairs=(("main", "text"),))


class _PlannedWrapper:
    """Takes the plan the ragged manager makes and returns q, so a read runs on the CPU."""

    def __init__(self, **kwargs):
        pass

    def plan(self, q_cu, kv_cu):
        pass

    def run(self, q, k, v):
        return q


class _CpuRaggedCross(_RaggedCrossManagerBase):
    """The ragged cross manager's planning and label lookup, over a wrapper that runs nothing."""

    wrapper_class = _PlannedWrapper


def _form_b(labels=frozenset({"main"})) -> list[Array]:
    """A max_len 0 cross read whose queries are a kept array's segments, as LingBot's text read."""
    chunk = Array("kv", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=64, labels=labels,
                  reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))})
    text = Array("text", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=0, labels={"text"},
                 reads={"text_attn": Read(mask=Block(), op=Attend(q_heads=HEADS), queries="kv")})
    return [chunk, text]


def test_a_max_len_0_read_over_a_kept_array_reads_through_node_handles():
    """The handle fixes its plan label at bind, and the step's plan holds the same one."""
    arrays = _form_b()
    config = RaggedAttentionConfig(num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM)
    ragged = _CpuRaggedCross(torch.device("cpu"), torch.float32, config, "text_attn")
    resources = {"kv": _StubKV(), "attn": _StubSelfRead(), "text_attn": ragged}
    assert {spec.resource_key for spec in build_specs(arrays)} == set(resources)
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "n", resources)
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 3), Segment("b", "main", 2)],
        state={"kv": ArrayStep(), "text": ArrayStep(segments=(Segment("a", "text", 4), Segment("b", "text", 4)))},
    ), handles)
    ctx = _ctx()
    ragged.plan(step.steps["text_attn"], ctx)
    step.set_ctx(ctx)
    step.begin_calls()
    for layer in sorted(LAYERS):
        _add_and_read(handles["kv"], layer, tokens=5)
        handles["text"].add(layer, k=torch.zeros(8, HEADS, DIM), v=torch.zeros(8, HEADS, DIM))
        out = handles["text"].read(layer, torch.ones(5, HEADS, DIM))
        assert out.shape == (5, HEADS, DIM), "the read should find the plan the step made"
    step.check_calls()
    step.end_calls()


def test_a_max_len_0_read_over_queries_on_two_labels_has_no_layout_yet():
    """A step on the second label would plan a label the handle never reads."""
    with pytest.raises(ValueError, match=r"Array 'text': no layout yet: reads\['text_attn'\] reads the queries of "
                                         r"'kv', which has labels \['cond', 'uncond'\]"):
        build_specs(_form_b(labels=frozenset({"cond", "uncond"})))


# The handle's cross read


def test_a_cross_add_writes_the_pages_under_its_label_and_holds_nothing():
    handles = _bound()
    context = handles["context"]
    step = _opened(handles, context_spans=(13, 7))
    k = v = torch.zeros(20, HEADS, DIM)
    context.add(torch.tensor(1), k=k, v=v)
    assert context._kv.writes == [(1, CONTEXT, 20)], "the context's K/V should be on its own label's pages"
    assert context._held is None, "a cross read attends the pages, so nothing waits for the read"
    step.end_calls()


def test_a_cross_read_runs_under_the_query_sides_label_with_no_k_or_v():
    """The cross resource keys its plans by the query side's label: under the context's it would read another plan."""
    handles = _bound()
    context = handles["context"]
    step = _opened(handles, context_spans=(13, 7))
    context.add(0, k=torch.zeros(20, HEADS, DIM), v=torch.zeros(20, HEADS, DIM))
    q = torch.ones(3, HEADS, DIM)
    assert context.read(0, q) is q
    assert context.first_read.runs == [("main", "view0")]
    step.end_calls()


def test_a_step_that_adds_no_context_reads_the_kept_entries():
    """A decode step lists the context at span 0, so every layer reads and none adds, and no check minds."""
    handles = _bound()
    context = handles["context"]
    step = _opened(handles)
    for layer in sorted(LAYERS):
        _add_and_read(handles["kv"], layer)
        context.read(layer, torch.zeros(3, HEADS, DIM))
    step.check_calls()
    step.end_calls()
    assert context._kv.writes == [] and len(context.first_read.runs) == 2


def test_a_prefill_adds_every_layer_then_reads_each():
    handles = _bound()
    context = handles["context"]
    step = _opened(handles, context_spans=(13, 7))
    tokens = torch.zeros(20, HEADS, DIM)
    for layer in sorted(LAYERS):
        context.add(layer, k=tokens, v=tokens)
    for layer in sorted(LAYERS):
        _add_and_read(handles["kv"], layer)
        context.read(layer, torch.zeros(3, HEADS, DIM))
    step.check_calls()
    step.end_calls()
    assert context._kv.writes == [(0, CONTEXT, 20), (1, CONTEXT, 20)], "every layer's context belongs on its own pages"


def test_a_context_added_on_one_layer_is_refused_at_the_steps_end():
    handles = _bound()
    step = _opened(handles, context_spans=(13, 7))
    handles["context"].add(0, k=torch.zeros(20, HEADS, DIM), v=torch.zeros(20, HEADS, DIM))
    for layer in sorted(LAYERS):
        _add_and_read(handles["kv"], layer)
    with pytest.raises(ValueError, match=r"Array 'context': layers \[1\] have no add in a step that keeps"):
        step.check_calls()
    step.end_calls()


def test_a_step_listing_the_cross_array_without_its_queries_is_refused():
    with pytest.raises(ValueError, match="Array 'context': a step lists it without 'kv'"):
        state_steps(SubmoduleStep(
            state={"context": ArrayStep(segments=(Segment("a", CONTEXT, 0),))},
        ), _bound())


def test_mismatched_requests_between_the_two_arrays_are_refused_when_they_open():
    handles = _bound()
    with pytest.raises(ValueError, match="its segments and those of 'kv'.*same order"):
        _opened(handles, context_rows="ba")


def test_a_replay_runs_no_request_check():
    handles = _bound()
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 1), Segment("b", "main", 1)],
        state={"kv": ArrayStep(), "context": ArrayStep(segments=(Segment("b", CONTEXT, 0), Segment("a", CONTEXT, 0)))},
    ), handles)
    step.set_ctx(StepContext(
        request_ids=("a", "b"), graph_walk="w", slot=0, capture=False, slot_lease=SimpleNamespace(captured=True),
    ))
    step.begin_calls()
    assert handles["context"]._open, "a replay's rows follow what its capture checked"
    step.end_calls()


def test_an_empty_query_side_leaves_the_query_label_alone():
    context = _bound()["context"]
    context._query_label = "kept"
    step = SubmoduleStep(segments=[], state={"kv": ArrayStep(segments=()), "context": ArrayStep(segments=())})
    step.set_ctx(_ctx(rows=()))
    context.begin_step(step)
    assert context._query_label == "kept"
    context.end_step(step)


def test_at_last_token_on_a_cross_array_is_refused_by_name():
    handles = _bound()
    step = _opened(handles)
    with pytest.raises(ValueError, match="Array 'context': at_last_token lays x out.*'kv'; call it on that array"):
        at_last_token(handles["context"], torch.zeros(1, 3))
    assert at_last_token(handles["kv"], torch.ones(1, 3)).shape == (1, 3), "its queried array serves the call"
    step.end_calls()


def test_a_handles_queried_array_is_keyword_only():
    with pytest.raises(TypeError):
        ArrayHandle(_context_array(), {"context": _StubKV(), "cross": _StubCross()}, None, None, _self_array())


# State ops and the pages


CUDA_ONLY = SimpleNamespace(formats=(PagedNHD(devices={"cuda"}),))
ANY_DEVICE = SimpleNamespace(formats=(PagedNHD(),))


def test_a_cross_handle_isnt_addressed_by_an_op_whose_formats_dont_list_its_pool():
    """Grouped wrong, ``queried or requires_kv_write and formats`` addresses every op, and `_call` would then run a
    fused kernel on a format it doesn't list."""
    context = _bound(device="cpu")["context"]
    assert not state_op_module._addresses(CUDA_ONLY, context)
    assert state_op_module._addresses(ANY_DEVICE, context), "a listed format may write the pages a cross read attends"


def test_state_ops_bind_on_a_node_with_a_cross_array(monkeypatch):
    """Reading ``requires_kv_write`` off the cross resource would raise at load, since it has none."""
    monkeypatch.setattr(state_op_module, "STATE_OPS", {"any_device": ANY_DEVICE, "cuda_only": CUDA_ONLY})
    handles = _bound(device="cpu")
    state_op_module.bind_state_ops(handles)
    assert handles["context"].fused_ops == {"any_device"}


class _StubDense(_StubSelfRead):
    requires_kv_write = False


def test_a_backend_that_takes_k_and_v_at_its_read_is_never_addressed():
    array = _self_array()
    handle = ArrayHandle(array, {"kv": _StubKV(), "attn": _StubDense(), "rope": object()}, "rope")
    assert not state_op_module._addresses(ANY_DEVICE, handle)


# ParallelCrossAttention


class _StubContext:
    """Records the handle calls a layer makes."""

    def __init__(self):
        self.added, self.read_at = [], []

    def add(self, layer, *, k, v):
        self.added.append((int(layer), k, v))

    def read(self, layer, q):
        self.read_at.append(int(layer))
        return q


def _cross_attention(**kwargs) -> ParallelCrossAttention:
    layer = ParallelCrossAttention(
        comm_group=CommGroup.trivial(), hidden_size=8, num_heads=2, head_dim=4, k_bias=True, **kwargs,
    )
    for parameter in layer.parameters():
        parameter.data.normal_(0, 0.02)
    return layer


@torch.no_grad()
def test_the_handle_path_adds_the_projected_context_and_reads_through_the_handle():
    layer = _cross_attention(context_kv_key="context", layer=3)
    handle = _StubContext()
    layer.bind_state({"context": handle})
    encoder_states = torch.randn(5, 8)
    layer.add_context(encoder_states)
    k, v = layer.compute_kv(encoder_states)
    assert handle.added[0][0] == 3
    assert torch.equal(handle.added[0][1], k) and torch.equal(handle.added[0][2], v)
    assert layer(torch.randn(2, 8)).shape == (2, 8)
    assert handle.read_at == [3], "the layer id reaches the handle"


# Through real resources


class _CpuVarlenSdpa(PrefixedPrefillWrapper):
    """The dense wrapper's interface over SDPA, since its kernels need a GPU."""

    def _plan_kernel(self, cu, kv_cu, total_tokens, causal):
        self._cu_q, self._cu_k, self._causal = cu.tolist(), kv_cu.tolist(), causal

    def run(self, q, k, v, return_lse=False):
        out = []
        for i in range(len(self._cu_q) - 1):
            row_q = q[self._cu_q[i]:self._cu_q[i + 1]]
            row_k = k[self._cu_k[i]:self._cu_k[i + 1]]
            row_v = v[self._cu_k[i]:self._cu_k[i + 1]]
            mask = None if not self._causal else torch.ones(
                row_q.shape[0], row_k.shape[0], dtype=torch.bool,
            ).tril(diagonal=row_k.shape[0] - row_q.shape[0])
            out.append(_sdpa(row_q, row_k, row_v, mask))
        return torch.cat(out)


def _sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, enable_gqa=True,
    ).transpose(0, 1)


@pytest.fixture
def dense(monkeypatch):
    """A real pool and a real dense manager over it, on the CPU, read through a real handle."""
    monkeypatch.setattr(kv_manager_module, "KVTransferManager", lambda info, kv_cache, **kwargs: None)
    device, dtype = torch.device("cpu"), torch.float64
    config = PagedKVConfig(
        num_layers=len(LAYERS), num_qo_heads=4, num_kv_heads=HEADS, head_dim=DIM, max_seq_len=64, max_num_pages=16,
        page_size=4,
    )
    kv = kv_manager_module.KVManager(
        cfg=config, name="kv", joint_comm_group=None, transfer_engine_info=None, device=device, dtype=dtype,
    )
    attn = DenseAttentionManager(
        kv_cache="kv", device=device, dtype=dtype, kv_config=config, wrapper_class=_CpuVarlenSdpa,
    )
    array = Array("kv", nodes={"n"}, layers=LAYERS, entry=ENTRY, max_len=64,
                  reads={"attn": Read(mask=Bidirectional(), op=Attend(q_heads=4))})
    for rid in ("a", "b"):
        kv.ingest_request(rid)
    return kv, attn, ArrayHandle(array, {"kv": kv, "attn": attn}, None)


def _run_step(kv, attn, handle, spans, keep):
    """One step's plan, calls and commit. Returns each layer's q, k, v and what its read returned."""
    declared = SubmoduleStep(
        segments=[Segment(rid, "main", span) for rid, span in spans.items()],
        state={"kv": ArrayStep(keep=keep)},
    )
    step = state_steps(declared, {"kv": handle})
    ctx = _ctx(rows=tuple(spans))
    assert kv.admit(step.steps["kv"], ctx).ok
    ctx.plan_results["kv"] = kv.plan(step.steps["kv"], ctx)
    attn.plan(step.steps["attn"], ctx)
    step.set_ctx(ctx)
    step.begin_calls()
    total = sum(spans.values())
    tensors, outputs = {}, {}
    for layer in sorted(LAYERS):
        q = torch.randn(total, 4, DIM, dtype=torch.float64)
        k = torch.randn(total, HEADS, DIM, dtype=torch.float64)
        v = torch.randn(total, HEADS, DIM, dtype=torch.float64)
        if keep:
            # The dense backend takes K/V at its read and never pages them, so this seeds the prefix a later step reads.
            kv.write_kv(k, v, layer_idx=layer, label="main")
        handle.add(layer, k=k, v=v)
        outputs[layer] = handle.read(layer, q)
        tensors[layer] = (q, k, v)
    step.check_calls()
    step.end_calls()
    kv.commit(step.steps["kv"], ctx)
    return tensors, outputs


def _expected(tensors, spans, kept, mask=False):
    """Each layer's float64 SDPA over ``[kept | this step's adds]`` per row, the reach of ``Bidirectional``."""
    expected = {}
    for layer, (q, k, v) in tensors.items():
        parts, offset = [], 0
        for rid, span in spans.items():
            past_k, past_v = kept.get((rid, layer), (k[:0], v[:0]))
            row_k = torch.cat([past_k, k[offset:offset + span]])
            row_v = torch.cat([past_v, v[offset:offset + span]])
            causal = torch.ones(span, row_k.shape[0], dtype=torch.bool).tril(diagonal=row_k.shape[0] - span)
            parts.append(_sdpa(q[offset:offset + span], row_k, row_v, causal if mask else None))
            offset += span
        expected[layer] = torch.cat(parts)
    return expected


def _kept(tensors, spans) -> dict:
    kept, offset = {}, 0
    for rid, span in spans.items():
        for layer, (_, k, v) in tensors.items():
            kept[(rid, layer)] = (k[offset:offset + span], v[offset:offset + span])
        offset += span
    return kept


def test_a_bidirectional_read_sees_the_row_kept_and_this_steps_adds_whole(dense):
    """A kept step whose tokens see each other, as Pi0.5's prefix and a V-JEPA 2 frame do, then a scratch step that sees
    the prefix and itself, as Pi0.5's action tokens do."""
    kv, attn, handle = dense
    torch.manual_seed(0)
    spans = {"a": 5, "b": 3}
    tensors, outputs = _run_step(kv, attn, handle, spans, keep=True)
    expected = _expected(tensors, spans, {})
    for layer in LAYERS:
        assert torch.allclose(outputs[layer], expected[layer], atol=1e-12), "a row's kept step should see itself whole"
    wrong = _expected(tensors, spans, {}, mask=True)
    assert not torch.allclose(outputs[0], wrong[0], atol=1e-6), "this comparison should tell the two flags apart"

    kept = _kept(tensors, spans)
    scratch = {"a": 2, "b": 4}
    tensors, outputs = _run_step(kv, attn, handle, scratch, keep=False)
    expected = _expected(tensors, scratch, kept)
    for layer in LAYERS:
        assert torch.allclose(outputs[layer], expected[layer], atol=1e-12), "scratch should see the prefix and itself"


def test_a_scratch_step_leaves_the_rows_kept_length_unchanged(dense):
    kv, attn, handle = dense
    torch.manual_seed(0)
    _run_step(kv, attn, handle, {"a": 5, "b": 3}, keep=True)
    for _ in range(2):
        _run_step(kv, attn, handle, {"a": 2, "b": 4}, keep=False)
        segments = attn._current_plans["main"].segments
        assert [segment.prefix_len for segment in segments] == [5, 3], "a denoise step would see another prefix"


# On the kernels that serve


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's kernels need a GPU")
_PAGED = dict(num_qo_heads=4, num_kv_heads=2, head_dim=64, page_size=16, max_num_pages=16)


def _pool(name: str, max_seq_len: int, label: str = "main") -> tuple[PagedKVConfig, kv_manager_module.KVManager]:
    config = PagedKVConfig(num_layers=len(LAYERS), max_seq_len=max_seq_len, **_PAGED)
    pool = kv_manager_module.KVManager(
        cfg=config, name=name, joint_comm_group=None, transfer_engine_info=None, device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    for rid in ("a", "b"):
        pool.ingest_request(rid, overrides=KVReqConfig(needed_labels=[label]))
    return config, pool


def _tokens(total: int, heads: int) -> torch.Tensor:
    return torch.randn(total, heads, _PAGED["head_dim"], device="cuda", dtype=torch.bfloat16)


def _float_sdpa(q, k, v) -> torch.Tensor:
    return _sdpa(q.float(), k.float(), v.float())


@requires_cuda
@torch.no_grad()
def test_a_bidirectional_read_on_flashinfers_kernel_matches_sdpa(monkeypatch):
    monkeypatch.setattr(kv_manager_module, "KVTransferManager", lambda info, kv_cache, **kwargs: None)
    torch.manual_seed(0)
    config, pool = _pool("kv", 64)
    attn = FlashInferManager(kv_cache="kv", device=torch.device("cuda"), dtype=torch.bfloat16, kv_config=config)
    array = Array("kv", nodes={"n"}, layers=LAYERS, entry=KVEntry(heads=2, head_dim=64), max_len=64,
                  reads={"attn": Read(mask=Bidirectional(), op=Attend(q_heads=4))})
    handle = ArrayHandle(array, {"kv": pool, "attn": attn}, None)
    kept: dict = {}
    for spans, keep in (({"a": 18, "b": 3}, True), ({"a": 2, "b": 4}, False)):
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "main", span) for rid, span in spans.items()], state={"kv": ArrayStep(keep=keep)},
        ), {"kv": handle})
        ctx = _ctx(rows=tuple(spans))
        assert pool.admit(step.steps["kv"], ctx).ok
        ctx.plan_results["kv"] = pool.plan(step.steps["kv"], ctx)
        attn.plan(step.steps["attn"], ctx)
        step.set_ctx(ctx)
        step.begin_calls()
        for layer in sorted(LAYERS):
            total = sum(spans.values())
            q, k, v = _tokens(total, 4), _tokens(total, 2), _tokens(total, 2)
            handle.add(layer, k=k, v=v)
            actual = handle.read(layer, q)
            offset, parts = 0, []
            for rid, span in spans.items():
                past_k, past_v = kept.get((rid, layer), (k[:0], v[:0]))
                row_k = torch.cat([past_k, k[offset:offset + span]])
                row_v = torch.cat([past_v, v[offset:offset + span]])
                parts.append(_float_sdpa(q[offset:offset + span], row_k, row_v))
                if keep:
                    kept[(rid, layer)] = (row_k, row_v)
                offset += span
            assert torch.allclose(actual.float(), torch.cat(parts), atol=2e-2, rtol=2e-2), f"layer {layer}, keep={keep}"
        step.check_calls()
        step.end_calls()
        pool.commit(step.steps["kv"], ctx)


@requires_cuda
@torch.no_grad()
def test_a_kept_cross_read_on_flashinfers_kernel_matches_sdpa_with_a_label_of_its_own(monkeypatch):
    """A prefill that adds each row's context under its own label, then a decode that reads what it kept."""
    monkeypatch.setattr(kv_manager_module, "KVTransferManager", lambda info, kv_cache, **kwargs: None)
    torch.manual_seed(0)
    self_config, self_pool = _pool("kv", 64)
    context_config, context_pool = _pool("context", 48, label=CONTEXT)
    attn = FlashInferManager(kv_cache="kv", device=torch.device("cuda"), dtype=torch.bfloat16, kv_config=self_config)
    cross = FlashInferCrossManager(
        kv_cache="context", query_kv_cache="kv", context_label=CONTEXT, device=torch.device("cuda"),
        dtype=torch.bfloat16, kv_config=context_config,
    )
    self_array = Array("kv", nodes={"n"}, layers=LAYERS, entry=KVEntry(heads=2, head_dim=64), max_len=64,
                       reads={"attn": Read(mask=Causal(), op=Attend(q_heads=4))})
    cross_array = Array("context", nodes={"n"}, layers=LAYERS, entry=KVEntry(heads=2, head_dim=64), max_len=48,
                        labels={CONTEXT},
                        reads={"cross": Read(mask=Bidirectional(), op=Attend(q_heads=4), queries="kv")})
    handles = {
        "kv": ArrayHandle(self_array, {"kv": self_pool, "attn": attn}, None),
        "context": ArrayHandle(cross_array, {"context": context_pool, "cross": cross}, None, queried=self_array),
    }
    contexts: dict = {}
    for query_spans, context_spans in (((5, 3), (20, 7)), ((1, 1), (0, 0))):
        segments = (Segment("a", CONTEXT, context_spans[0]), Segment("b", CONTEXT, context_spans[1]))
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "main", span) for rid, span in zip("ab", query_spans, strict=True)],
            state={"kv": ArrayStep(), "context": ArrayStep(segments=segments)},
        ), handles)
        ctx = _ctx()
        assert self_pool.admit(step.steps["kv"], ctx).ok and context_pool.admit(step.steps["context"], ctx).ok
        ctx.plan_results["kv"] = self_pool.plan(step.steps["kv"], ctx)
        ctx.plan_results["context"] = context_pool.plan(step.steps["context"], ctx)
        attn.plan(step.steps["attn"], ctx)
        cross.plan(step.steps["cross"], ctx)
        step.set_ctx(ctx)
        step.begin_calls()
        for layer in sorted(LAYERS):
            if sum(context_spans):
                k, v = _tokens(sum(context_spans), 2), _tokens(sum(context_spans), 2)
                handles["context"].add(layer, k=k, v=v)
                contexts[("a", layer)] = (k[:context_spans[0]], v[:context_spans[0]])
                contexts[("b", layer)] = (k[context_spans[0]:], v[context_spans[0]:])
        for layer in sorted(LAYERS):
            total = sum(query_spans)
            q = _tokens(total, 4)
            handles["kv"].add(layer, k=_tokens(total, 2), v=_tokens(total, 2))
            handles["kv"].read(layer, _tokens(total, 4))
            actual = handles["context"].read(layer, q)
            expected = torch.cat([
                _float_sdpa(q[:query_spans[0]], *contexts[("a", layer)]),
                _float_sdpa(q[query_spans[0]:], *contexts[("b", layer)]),
            ])
            assert torch.allclose(actual.float(), expected, atol=2e-2, rtol=2e-2), f"layer {layer}, {query_spans}"
        step.check_calls()
        step.end_calls()
        self_pool.commit(step.steps["kv"], ctx)
        context_pool.commit(step.steps["context"], ctx)
