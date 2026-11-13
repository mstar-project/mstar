"""Steps over several labels, reads that differ by walk, and the node that admits for a pool on several nodes.

Chatterbox and Bagel's guided steps put every row's main and every row's guidance branch into one plan, and Bagel reads
one cache causally on its text walks and as a whole on its image walks. A plan packed in another order, a label's rows
sliced one off, or a causal flag taken from the wrong walk changes outputs with no error, so each rule is checked here
over CPU stand-ins, and each refusal names the array and the cause.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_layout_intake import _Model as _IntakeModel
from test_kv_layout_intake import _preprocess
from test_state_calls import (
    HEADS,
    TOKENS,
    _add,
    _array,
    _begin,
    _ctx,
    _kv_handle,
    _refused,
    _step,
    _StubAttn,
    _StubDense,
    _StubExchange,
    _StubKV,
    _tensors,
    _value_handle,
)

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.conductor.conductor import Conductor
from mstar.engine.resources.attn.config import AttentionSpec, AttentionStep
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    PagedKVConfig,
    RingKVConfig,
    RingKVLayerConfig,
    Span,
    refuse_undeclared_labels,
)
from mstar.engine.resources.kv.plan import group_by_plan_label
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Causal,
    KVEntry,
    Layout,
    Positions,
    Read,
    Window,
    at_last_token,
)
from mstar.engine.state.adapters import (
    PAGED,
    PAGED_BY_WALK,
    StateSetup,
    Walks,
    _leader,
    _resolve,
    build_handle,
    build_specs,
    check_capture,
    node_handles,
    state_steps,
    storage_of,
)
from mstar.engine.state.calls import ArrayHandle
from mstar.model.base import KVLabel, LabelLayout, PrefixKeys
from mstar.model.utils import check_kv_pools_fit

LABELS = ("main", "uncond")
TEXT, IMAGE = frozenset({"prefill_text", "decode"}), frozenset({"prefill_vit", "image_gen"})
COMBINED = {"mixed_text": frozenset(TEXT)}


def _label_major(spans=(3, 5), labels=LABELS) -> list[Segment]:
    return [Segment(row, label, span) for label in labels for row, span in enumerate(spans, start=1)]


class _LastTokens(_StubAttn):
    """Selects each segment's last token as FlashInfer's plan does, from the spans it was handed."""

    spans: list[int] = []

    def select_last_hidden(self, x, label="main"):
        ends = torch.tensor(self.spans).cumsum(0) - 1
        return x.index_select(0, ends)


def _two_label_handle(attn=None) -> ArrayHandle:
    return _kv_handle(_array(labels=set(LABELS)), attn=attn)


# Steps over several labels


def test_a_step_over_two_labels_binds_one_plan_named_for_both():
    h = _two_label_handle()
    step = state_steps(_step(h, segments=_label_major()), {"kv": h}, graph_walk="decode")
    assert step.steps["kv"].combined_labels == {LABELS: "main+uncond"}, "the KV resource would plan each label alone"
    assert all(built.combined_labels == step.combined_labels for built in step.steps.values()), \
        "a resource that plans the step's rows would pack them by their own labels"
    step.set_ctx(_ctx(rows=(1, 2)))
    h.begin_step(step)
    assert h.label == "main+uncond", "add, read and rope would plan under one label's rows"


def test_a_step_that_packs_two_labels_on_one_array_refuses_another_array_listed_on_one():
    """Every array on KV pages plans the labels a step packs together, so an array the step lists on one of them would
    plan a label it has no rows on, a ``KeyError`` in the KV plan."""
    array = replace(_array(), name="draft", reads={"draft_attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))})
    one = build_handle(array, {"draft": _StubKV(), "draft_attn": _StubAttn()}, _resolve([array], StateSetup()))
    step = SubmoduleStep(segments=_label_major(), state={
        "kv": ArrayStep(), "draft": ArrayStep(segments=[Segment(row, "main", TOKENS) for row in (1, 2)]),
    })
    _refused(
        r"Array 'draft': no layout yet: a step lists it on labels \['main'\] and Array 'kv' on \['main', 'uncond'\]",
        lambda: state_steps(step, {"kv": _two_label_handle(), "draft": one}, graph_walk="decode"),
    )


def test_a_step_over_one_label_plans_as_before():
    h = _two_label_handle()
    step = _begin(h, segments=[Segment(1, "uncond", TOKENS)])
    assert h.label == "uncond" and state_steps(step, {"kv": h}, graph_walk="decode").steps["kv"].combined_labels == {}


def test_a_combined_plan_keeps_a_label_major_steps_order_and_reorders_any_other():
    """The step's segment order is the layout of the model's tensors, and the plan packs each label's rows together."""
    label_major = tuple(_label_major())
    row_major = tuple(Segment(row, label, span) for row, span in ((1, 3), (2, 5)) for label in LABELS)
    assert tuple(group_by_plan_label(label_major, {LABELS: "main+uncond"})["main+uncond"]) == label_major
    assert tuple(group_by_plan_label(row_major, {LABELS: "main+uncond"})["main+uncond"]) != row_major, (
        "a row-major step would come back reordered, which is why the handle refuses it"
    )


def test_at_last_token_gives_every_row_and_label_or_one_labels_rows():
    attn = _LastTokens()
    h = _two_label_handle(attn=attn)
    segments = _label_major(spans=(3, 5))
    attn.spans = [segment.span for segment in segments]
    _begin(h, segments=segments, ctx=_ctx(rows=(1, 2)))
    x = torch.arange(16.0)[:, None]
    assert at_last_token(h, x)[:, 0].tolist() == [2.0, 7.0, 10.0, 15.0]
    uncond = at_last_token(h, x, label="uncond")
    attn.spans = [3, 5]
    own_plan = attn.select_last_hidden(x[8:])
    assert torch.equal(uncond, own_plan), "a label's rows sliced from the combined plan differ from its own plan's"


def test_at_last_token_on_a_label_the_step_doesnt_touch_raises():
    h = _two_label_handle()
    _begin(h, segments=_label_major(labels=("main",)), ctx=_ctx(rows=(1, 2)))
    _refused(r"Array 'kv': at_last_token on label 'uncond', and the step's segments name \['main'\]",
             lambda: at_last_token(h, torch.zeros(8, 1), label="uncond"))


def test_at_last_token_refuses_a_segment_with_no_token():
    """A span-0 segment adds nothing to the plan, so its row would be the previous segment's last token."""
    h = _two_label_handle()
    segments = [Segment(1, "main", 3), Segment(2, "main", 2), Segment(1, "uncond", 0), Segment(2, "uncond", 2)]
    _begin(h, segments=segments, ctx=_ctx(rows=(1, 2)))
    _refused(r"Array 'kv': at_last_token on row 1, label 'uncond', whose segment has span 0",
             lambda: at_last_token(h, torch.zeros(7, 1), label="uncond"))


def test_a_row_with_two_segments_on_one_label_of_a_combined_step_is_refused():
    h = _two_label_handle()
    segments = [Segment(1, "main", 2), Segment(1, "main", 1), Segment(1, "uncond", 3)]
    _refused(r"Array 'kv': row 1 has two segments on label 'main' in one step", lambda: _begin(h, segments=segments))


def test_a_step_over_two_labels_on_a_value_has_no_layout_yet():
    h = _value_handle(labels=set(LABELS))
    _refused(r"Array 'gdn': no layout yet: a step touches labels \['main', 'uncond'\], and a plan holds one per call",
             lambda: _begin(h, segments=_label_major(), ctx=_ctx(rows=(1, 2))))


def test_a_declared_label_with_a_plus_is_refused_at_load():
    """The plan of a step over several labels takes their names joined by '+', and a label of that name would replace
    that plan's rows with its own."""
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: labels \['main\+uncond'\] hold a '\+'"):
        build_specs([_array(labels={"main", "uncond", "main+uncond"})])


def test_a_cross_read_refuses_queries_laid_out_over_two_labels():
    """The cross resource keys its plans by the query side's label, and a combined query side has no such label."""
    queried = _array(labels=set(LABELS))
    context = Array("ctx", nodes={"LLM"}, layers=queried.layers, entry=queried.entry, max_len=16,
                    reads={"cross": Read(mask=Bidirectional(), op=Attend(q_heads=2), queries="kv")})
    resolved = _resolve([queried, context], StateSetup())
    h = build_handle(context, {"ctx": _StubKV(), "cross": _StubAttn()}, resolved=resolved)
    step = SubmoduleStep(segments=[Segment(1, "main", 0), Segment(2, "main", 0)], state={
        "ctx": ArrayStep(), "kv": ArrayStep(segments=tuple(_label_major())),
    })
    step.set_ctx(_ctx(rows=(1, 2)))
    _refused(r"Array 'ctx': no layout yet: a step over labels \['main', 'uncond'\] on 'kv'", lambda: h.begin_step(step))


# The labels a request opens


def _conductor(kv_labels: dict, returned: dict | None = None) -> Conductor:
    conductor = Conductor.__new__(Conductor)
    conductor.model = SimpleNamespace(get_request_resource_configs=lambda *args: dict(returned or {}))
    conductor._kv_labels = kv_labels
    conductor._node_resources_cache = []
    return conductor


def test_a_request_opens_the_labels_its_layout_names():
    conductor = _conductor({"t3_kv": LABELS})
    layout = {"t3_kv": {"main": LabelLayout(spans=()).row()}}
    configs = conductor._get_resource_configs([], [], {"kv_layout": layout})
    assert configs["t3_kv"].needed_labels == ["main"], "an unguided request would open uncond, which it doesn't lay out"


def test_a_request_opens_every_label_of_an_array_its_layout_doesnt_name():
    configs = _conductor({"t3_kv": LABELS})._get_resource_configs([], [], {"kv_layout": {}})
    assert configs["t3_kv"].needed_labels == list(LABELS)


def test_a_config_the_model_returns_is_left_as_it_returned_it():
    """Bagel names its labels per node and walk, and an unguided request touches no branch label."""
    returned = KVReqConfig(needed_labels_per_node_walk={("LLM", "decode"): ["main"]})
    configs = _conductor({"kv": ("main", "cfg_img")}, {"kv": returned})._get_resource_configs([], [], {"kv_layout": {}})
    assert configs["kv"].needed_labels is None, "the request would open branch labels its unguided steps never touch"


def test_an_array_the_layout_doesnt_name_needs_a_row_for_each_label_it_opens():
    configs = {"t3_kv": KVReqConfig(needed_labels=list(LABELS))}
    with pytest.raises(RuntimeError, match=r"returned no kv_layout for 't3_kv'/'main', which the request opens"):
        refuse_undeclared_labels("Stub", {"t3_kv": LABELS}, {}, configs)


# Reads that differ by walk


def _per_walk(**reads) -> Array:
    return _array(labels=set(LABELS), reads=reads or {
        "attn": Read(mask=Causal(), op=Attend(q_heads=2), walks=TEXT),
        "attn_image": Read(mask=Bidirectional(), op=Attend(q_heads=2), walks=IMAGE),
    })


def _per_walk_handle(array=None, attn=None) -> ArrayHandle:
    """Built as a node builds it, since `node_handles` reads the model's combined walks."""
    array = array or _per_walk()
    model = SimpleNamespace(declare_state=lambda: [array], get_combined_graph_walks=lambda: {"LLM": COMBINED})
    keys = set(_resolve([array], StateSetup()).resource_of.values())
    resources = {"kv": _StubKV(), **{key: attn(key) if attn else _StubAttn() for key in keys}}
    return node_handles(model, "LLM", resources)["kv"]


def _causal(h: ArrayHandle, walk: str | None) -> dict[str, bool]:
    step = _step(h, segments=[Segment(1, "main", TOKENS)])
    return {key: s.causal for key, s in state_steps(step, {"kv": h}, graph_walk=walk).steps.items()
            if isinstance(s, AttentionStep)}


def test_reads_that_differ_by_walk_share_one_spec_and_plan_causal_per_walk():
    specs = build_specs([_per_walk()])
    assert [spec.resource_key for spec in specs if isinstance(spec, AttentionSpec)] == ["attn"], (
        "each read would get its own FlashInfer manager over one pool"
    )
    h = _per_walk_handle()
    assert {walk: _causal(h, walk) for walk in ("prefill_text", "decode", "prefill_vit", "image_gen")} == {
        "prefill_text": {"attn": True}, "decode": {"attn": True},
        "prefill_vit": {"attn": False}, "image_gen": {"attn": False},
    }


def test_an_array_whose_reads_differ_by_walk_has_a_storage_of_its_own():
    """Its handle holds each walk's reads, which an array whose reads serve every walk has no use for."""
    assert storage_of(_per_walk()) is PAGED_BY_WALK
    assert storage_of(_array()) is PAGED, "every paged array's handle would carry a walk table"


def test_an_array_whose_reads_differ_by_walk_steps_its_positions_on_every_walk():
    """A walk's step plans only that walk's reads, and the array's positions on any walk."""
    h = _per_walk_handle(_array(positions=Positions(), reads=_per_walk().reads))
    for walk in ("decode", "prefill_vit", "mixed_text", "vae_decode"):
        steps = state_steps(_step(h), {"kv": h}, graph_walk=walk).steps
        assert isinstance(steps.get("rope"), PositionStep), f"a step on {walk!r} would leave the positions unplanned"


def test_a_combined_walk_plans_the_mask_its_walks_read():
    assert _causal(_per_walk_handle(), "mixed_text") == {"attn": True}


def test_shared_reads_take_the_first_reads_key_in_declaration_order():
    image_first = _per_walk(
        attn_image=Read(mask=Bidirectional(), op=Attend(q_heads=2), walks=IMAGE),
        attn=Read(mask=Causal(), op=Attend(q_heads=2), walks=TEXT),
    )
    assert _resolve([image_first], StateSetup()).resource_of == {"attn_image": "attn_image", "attn": "attn_image"}


def test_reads_with_two_masks_on_one_walk_keep_two_specs():
    """With equal configs they would share a spec, whose step plans one causal flag for every layer."""
    array = _array(layers={0, 1, 2, 3}, reads={
        "causal": Read(mask=Causal(), op=Attend(q_heads=2), layers={0, 1}),
        "whole": Read(mask=Bidirectional(), op=Attend(q_heads=2), layers={2, 3}),
    })
    assert [spec.resource_key for spec in build_specs([array]) if isinstance(spec, AttentionSpec)] == [
        "causal", "whole",
    ]


def test_two_specs_over_one_layer_on_different_walks_have_no_layout_yet():
    """A handle reads each layer through one resource, so the other walk would read stale plans."""
    array = _per_walk(attn=Read(mask=Causal(), op=Attend(q_heads=2), walks={"a"}),
                      local=Read(mask=Window(16), op=Attend(q_heads=2), walks={"b"}))
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: reads\['attn'\] and reads\['local'\] build "
                                         r"different specs and both cover layer 0"):
        build_specs([array])


def test_a_step_with_no_walk_on_reads_that_differ_by_walk_has_no_layout_yet():
    """A piecewise region steps under no walk of the model."""
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: a step with no walk"):
        _causal(_per_walk_handle(), None)


def test_a_step_with_no_walk_on_reads_of_every_walk_plans_as_before():
    assert _causal(_kv_handle(), None) == {"attn": True}


@pytest.mark.parametrize("walks, refused", [
    ({"prefill_text", "mixed_text"}, False),
    ({"prefill_text", "prefill_vit"}, True),
])
def test_a_capture_replayed_on_walks_of_two_masks_is_refused(walks, refused):
    """FlashInfer fixes the mask when a capture plans, so every replay would run the capture walk's."""
    h = _per_walk_handle()
    if not refused:
        check_capture({"kv": h}, walks)
        return
    with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: a capture replayed on walks \['prefill_text', "
                                         r"'prefill_vit'\] runs reads \['attn', 'attn_image'\] on 'attn'"):
        check_capture({"kv": h}, walks)


class _Named(_StubAttn):
    """Returns its own name from `select_last_hidden`, so a test sees which resource's plan was read."""

    def __init__(self, name):
        self.name = name

    def select_last_hidden(self, x, label="main"):
        return self.name


def _two_spec_handle() -> ArrayHandle:
    """Two reads that build different specs over different layers, each on its own walk, and walk ``c`` with none."""
    array = _per_walk(causal=Read(mask=Causal(), op=Attend(q_heads=2), walks={"a"}, layers={0, 1}),
                      local=Read(mask=Window(16), op=Attend(q_heads=2), walks={"b"}, layers={2, 3}))
    return _per_walk_handle(array, attn=_Named)


def _at_last_token_on(h: ArrayHandle, walk: str):
    step = _step(h, segments=[Segment(1, "main", TOKENS)])
    step.set_ctx(StepContext(request_ids=(1,), graph_walk=walk, slot=0, capture=False))
    h.begin_step(step)
    return at_last_token(h, torch.zeros(TOKENS, 1))


@pytest.mark.parametrize("walk", ["a", "b"])
def test_at_last_token_selects_through_the_read_its_walk_planned(walk):
    """A walk plans only its own reads' resources, so the other resource holds another step's plan."""
    expected = {"a": "causal", "b": "local"}[walk]
    assert _at_last_token_on(_two_spec_handle(), walk) == expected, "at_last_token read a plan its walk didn't make"


def test_at_last_token_on_a_walk_that_runs_no_read_is_refused():
    _refused(r"Array 'kv': at_last_token on walk 'c', which runs no read of this array",
             lambda: _at_last_token_on(_two_spec_handle(), "c"))


def test_a_read_on_a_layer_no_read_covers_on_the_walk_raises():
    h = _per_walk_handle(_per_walk(attn=Read(mask=Causal(), op=Attend(q_heads=2), walks=TEXT, layers={0, 1})))
    step = _step(h, segments=[Segment(1, "main", TOKENS)])
    step.set_ctx(StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False))
    h.begin_step(step)
    q, k, v = _tensors()
    h.add(3, k=k, v=v)
    _refused(r"Array 'kv': a read on layer 3, which no read covers on walk 'decode'", lambda: h.read(3, q))


def test_kernel_settings_reach_the_shared_spec_and_two_values_for_one_field_are_refused():
    shared = {"attn": {"decode_tensor_cores": False}, "attn_image": {"decode_tensor_cores": False}}
    specs = build_specs([_per_walk()], kernel_settings=shared)
    assert next(spec for spec in specs if spec.resource_key == "attn").config.decode_tensor_cores is False
    with pytest.raises(ValueError, match=r"kernel_settings\['attn'\] and kernel_settings\['attn_image'\] set "
                                         r"decode_tensor_cores to False and True, and both reads run on the resource "
                                         r"'attn', which holds one value"):
        build_specs([_per_walk()], kernel_settings={**shared, "attn_image": {"decode_tensor_cores": True}})


# An add a read takes k and v from


class _DenseExchange(_StubExchange):
    """Takes k and v at the read, through its exchange, as the dense backend does on a node whose shard splits. Its
    exchange writes nothing, so an add it never reads reaches no page."""

    requires_kv_write = False


@pytest.mark.parametrize("attn", [_StubDense, _DenseExchange], ids=["direct", "exchange"])
def test_an_add_while_another_layers_add_waits_for_its_read_is_refused(attn):
    h = _kv_handle(attn=attn())
    _begin(h)
    _add(h, 0)
    _refused(r"Array 'kv': an add on layer 1 while layer 0's add waits for its read", lambda: _add(h, 1))


@pytest.mark.parametrize("attn", [_StubDense, _DenseExchange], ids=["direct", "exchange"])
def test_a_kept_step_that_ends_with_an_add_unread_is_refused(attn):
    h = _kv_handle(attn=attn())
    step = _begin(h)
    q, _, _ = _tensors()
    for layer in range(4):
        _add(h, layer)
        if layer < 3:
            h.read(layer, q)
    _refused(r"Array 'kv': the step ends with layer 3's add unread", lambda: h.check_step(step))


# Keyed spans on a walk read as a whole


def _keyed_worker(mask) -> SimpleNamespace:
    return SimpleNamespace(
        _keyed_labels={"kv": {"main": None}},
        _whole_reads={"kv": {"prefill_vit": "attn_image"}} if mask is Bidirectional else {},
    )


def _layout(*spans) -> dict:
    return {"kv": {"main": LabelLayout(spans=spans)}}


_TEXT = Span("ids", "prefill_text", 4, 4, "text_inputs")


@pytest.mark.parametrize("spans, refusal", [
    ((_TEXT, Span("digest", "prefill_vit", 6, 1, ("image", 0))), None),
    ((_TEXT, Span("ids", "prefill_vit", 4, 4, "caption_ids")), "span 1 of 'main' is an ids span"),
    ((_TEXT, Span("digest", "prefill_vit", 6, 1, ("image", 0)), Span("digest", "prefill_vit", 6, 1, ("image", 1),
                                                                     opens_step=False)),
     "span 1 of 'main' is a digest span that shares its step"),
], ids=["one_item_a_step", "ids", "two_items_a_step"])
def test_a_keyed_span_a_hit_could_end_inside_a_whole_step_of_fails_the_request(spans, refusal):
    """A prefix hit ends at a page, so a step whose adds are read as a whole has to be one keyed item."""
    check = lambda: PreprocessWorkerThread._check_whole_steps(_keyed_worker(Bidirectional), _layout(*spans))  # noqa: E731
    if refusal is None:
        check()
        return
    with pytest.raises(RuntimeError, match=rf"Array 'kv'.reads\['attn_image'\] reads walk 'prefill_vit' as a whole, "
                                           rf"and {refusal}, so a prefix hit could end inside the step"):
        check()


def test_a_keyed_ids_span_on_a_causal_walk_passes():
    PreprocessWorkerThread._check_whole_steps(_keyed_worker(Causal), _layout(_TEXT, Span("ids", "decode", 4, 4, "x")))


class _WholeReadModel(_IntakeModel):
    """Reads its declared cache as a whole and returns its own ``kv_layout``."""

    def __init__(self):
        super().__init__(layout={"kv": {"main": LabelLayout(spans=(Span("ids", "prefill", 100, 100, "text_inputs"),))}})

    def get_node_resources(self):
        return []

    def declare_state(self):
        return [Array("kv", nodes={"llm"}, layers=frozenset({0}), entry=KVEntry(heads=1, head_dim=8), max_len=4096,
                      reads={"attn": Read(mask=Bidirectional(), op=Attend(q_heads=1))})]

    def kv_labels(self):
        return {"kv": {"main": KVLabel(keys=PrefixKeys("caption_ids", ("prefill",)))}}


def test_a_model_that_lays_out_its_own_prompt_is_checked_per_request_too():
    """The load check leaves a keyed walk whose node doesn't take the keyed tensor to the preprocess worker, which
    checked only a ``state_layout``: a model that returned ``kv_layout`` itself had neither check."""
    with pytest.raises(RuntimeError, match=r"Array 'kv'\.reads\['attn'\] reads walk 'prefill' as a whole, and span 0 "
                                           r"of 'main' is an ids span"):
        _preprocess(_WholeReadModel(), {})


# The node that admits for a pool on several nodes

# Bagel under CFG parallel: its text walks run on LLM alone, and a guided window on LLM and the branches
BAGEL_WALKS = {
    "prefill_text": {"LLM"}, "decode": {"LLM"}, "prefill_vit": {"LLM", "vit_encoder"}, "image_gen": {"LLM", "vae"},
    "image_gen_cfg2": {"LLM", "LLM_cfg_text", "combine_cfg"},
    "image_gen_cfg3": {"LLM", "LLM_cfg_text", "LLM_cfg_img", "combine_cfg"}, "image_decode": {"vae"},
}


def _pool(*nodes) -> Array:
    return _array(nodes=set(nodes))


def test_the_leader_is_the_one_node_that_runs_every_walk_its_pool_runs():
    assert _leader(_pool("LLM", "LLM_cfg_text", "LLM_cfg_img"), Walks(nodes=BAGEL_WALKS)) == "LLM"


def test_a_pool_where_no_node_runs_every_walk_has_no_leader():
    """Cosmos3's DiT and reasoner each miss the other's walks, so each admits for itself."""
    assert _leader(_pool("dit", "reasoner"), Walks(nodes={"generate": {"dit"}, "reason": {"reasoner"}})) is None


def test_a_pool_on_one_node_has_no_leader():
    """Another array on several nodes fills the walks' nodes, and this pool's one node admits for itself."""
    setup = StateSetup(walks=Walks(nodes=BAGEL_WALKS))
    (kv,) = [spec for spec in build_specs([_pool("LLM")], setup=setup) if isinstance(spec, KVSpec)]
    assert kv.leader is None


def test_a_kept_arrays_handle_is_the_same_whichever_of_its_nodes_builds_it():
    array = _array(nodes={"LLM", "LLM_cfg_img"}, labels=set(LABELS), reads=_per_walk().reads)
    model = SimpleNamespace(declare_state=lambda: [array], get_combined_graph_walks=lambda: {"LLM": COMBINED},
                            get_graph_walk_graphs=dict)
    resources = {"kv": _StubKV(), **{key: _StubAttn() for key in _resolve([array], StateSetup()).resource_of.values()}}
    llm, cfg = (node_handles(model, node, resources)["kv"] for node in ("LLM", "LLM_cfg_img"))
    assert set(COMBINED) <= set(cfg.walk_reads.reads)
    assert cfg.walk_reads == llm.walk_reads, (
        "the engine keeps whichever node's handle it builds first, so it has to plan every node's combined walks"
    )


def test_build_specs_puts_the_leader_on_the_pool():
    setup = StateSetup(walks=Walks(nodes=BAGEL_WALKS))
    (kv,) = [spec for spec in build_specs([_pool("LLM", "LLM_cfg_img")], setup=setup) if isinstance(spec, KVSpec)]
    assert kv.leader == "LLM"


# The pool check counts each of a request's streams


def test_a_recurrent_pool_holds_a_slot_for_each_label_of_each_request(monkeypatch):
    """A request reserves a slot of a recurrent pool for each label it opens there, so a guided request takes two."""
    state = RecurrentStateConfig(num_layers=1, blocks={"h": RecurrentBlockConfig(shape=(4,), dtype=torch.float32)},
                                 max_slots=1 + 2 * 2, labels=("main", "uncond"))
    monkeypatch.setattr("mstar.model.base.serving_resources",
                        lambda model, config: [RecurrentStateSpec(resource_key="state", nodes={"t3"}, config=state)])
    check_kv_pools_fit(None, {"max_concurrent_requests": 2}, "cfg.yaml", logging.getLogger(__name__))
    with pytest.raises(ValueError, match=r"3 concurrent requests \(max_concurrent_requests\) need 6 slots of "
                                         r"resources.state \(2 per request\), which leaves 4 of its 5"):
        check_kv_pools_fit(None, {"max_concurrent_requests": 3}, "cfg.yaml", logging.getLogger(__name__))


def test_two_streams_of_a_request_need_the_pages_of_both(monkeypatch):
    """Chatterbox's guided requests hold a main and an uncond stream on one pool."""
    pool = PagedKVConfig(num_layers=2, num_qo_heads=2, num_kv_heads=2, head_dim=8, max_seq_len=256,
                         page_size=64, max_num_pages=1 + 2 * 4)
    monkeypatch.setattr("mstar.model.base.serving_resources",
                        lambda model, config: [KVSpec(resource_key="t3_kv", nodes={"t3"}, config=pool)])
    array = Array("t3_kv", nodes={"t3"}, layers=frozenset({0, 1}), entry=KVEntry(heads=2, head_dim=8), max_len=256,
                  labels=frozenset(LABELS), reads={"attn": Read(mask=Causal(), op=Attend(q_heads=2))})
    model = SimpleNamespace(declare_state=lambda: [array])
    stream = Layout(spans=(Span("ids", "prefill", 192, 192, "text_inputs"),), growth=(("decode_loop", 1),))
    config = {"max_concurrent_requests": 2}
    check_kv_pools_fit(model, config, "cfg.yaml", logging.getLogger(__name__))
    with pytest.raises(ValueError, match=r"2 concurrent requests \(max_concurrent_requests\) need 17 pages of "
                                         r"resources.t3_kv \(8 per request and the sink page\), which has 9"):
        check_kv_pools_fit(model, config, "cfg.yaml", logging.getLogger(__name__),
                           request_layouts={"t3_kv": dict.fromkeys(LABELS, stream)}, loop_caps={"decode_loop": 64})
    with pytest.raises(ValueError, match=r"request_layouts\['t3_kv'\] grows loops \['decode_loop'\], which loop_caps "
                                         r"doesn't cap"):
        check_kv_pools_fit(model, config, "cfg.yaml", logging.getLogger(__name__),
                           request_layouts={"t3_kv": dict.fromkeys(LABELS, stream)})


def test_a_placed_pool_counts_the_ranks_of_the_node_groups_that_run_its_nodes(monkeypatch):
    """A pool placed by position over 2 SP ranks pages its window layer in half pages, so a request holds 8 full and
    16 window pages rather than 8 and 8. The check hands the pool the ranks of the node groups that run its nodes,
    so SP ranks under another node leave its count alone."""
    pool = PagedKVConfig(num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=20, page_size=8,
                         context_parallel=True)
    pool.keep_window(8, (1,))
    monkeypatch.setattr("mstar.model.base.serving_resources",
                        lambda model, config: [KVSpec(resource_key="kv", nodes={"llm"}, config=pool)])
    split = {"ranks": [0, 1], "sp_size": 2}
    config = {"max_concurrent_requests": 2, "node_groups": [{"node_names": ["llm"]}, {"node_names": ["vae"], **split}]}
    check_kv_pools_fit(None, config, "cfg.yaml", logging.getLogger(__name__))
    config["node_groups"] = [{"node_names": ["llm"], **split}]
    with pytest.raises(ValueError, match=r"need 25 pages of resources.kv \(12 per request and the sink page\)"):
        check_kv_pools_fit(None, config, "cfg.yaml", logging.getLogger(__name__))


def test_at_last_token_over_every_row_of_one_label_takes_a_row_that_pads_the_step():
    """A padded step lists each row that pads it at span 0, and a call over every row returns what the plan selects
    for it, which nothing reads."""
    attn = _LastTokens()
    h = _kv_handle(attn=attn)
    segments = [Segment(1, "main", 3), Segment(2, "main", 0)]
    attn.spans = [segment.span for segment in segments]
    _begin(h, segments=segments, ctx=_ctx(rows=(1, 2)))
    assert at_last_token(h, torch.arange(3.0)[:, None])[:, 0].tolist() == [2.0, 2.0], (
        "the padding row would not get what the plan selects for it"
    )


def test_at_last_token_over_every_row_of_a_combined_plan_refuses_a_segment_with_no_token():
    """Chatterbox's guided T3 prefill takes every (row, label)'s last token, and a span-0 one would be another's."""
    h = _two_label_handle()
    segments = [Segment(1, "main", 3), Segment(2, "main", 2), Segment(1, "uncond", 0), Segment(2, "uncond", 2)]
    step = state_steps(_step(h, segments=segments), {"kv": h}, graph_walk="decode")
    step.set_ctx(_ctx(rows=(1, 2)))
    h.begin_step(step)
    _refused(r"Array 'kv': at_last_token on row 1, label 'uncond', whose segment has span 0",
             lambda: at_last_token(h, torch.zeros(7, 1)))


def test_a_node_takes_the_handle_another_node_of_its_process_built():
    array = _array(nodes={"LLM", "LLM_cfg_img"})
    model = SimpleNamespace(declare_state=lambda: [array], get_graph_walk_graphs=dict)
    resources, built = {"kv": _StubKV(), "attn": _StubAttn()}, {}
    llm, cfg = (node_handles(model, node, resources, built=built)["kv"] for node in ("LLM", "LLM_cfg_img"))
    assert cfg is llm, "a layer both nodes run would stay on the handle of the node bound last"


def test_a_pool_its_window_sizes_holds_any_cap():
    """The pool check asks every spec, so a pool that requests don't size says so itself."""
    ring = KVSpec(resource_key="ring", nodes={"t3"}, config=RingKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=8, tokens_per_frame=4,
        layers=(RingKVLayerConfig(ring_frames=8, ring_buckets=1, pinned_dilation=1),),
    ))
    assert ring.capacity_problem(1000, None) is None, "a ring buffer's window sizes it, not its requests"


def test_a_paged_pool_with_no_layout_holds_one_stream_of_its_max_len_a_request():
    paged = KVSpec(resource_key="t3_kv", nodes={"t3"}, config=PagedKVConfig(
        num_layers=2, num_qo_heads=2, num_kv_heads=2, head_dim=8, max_seq_len=256, page_size=64, max_num_pages=9,
    ))
    assert paged.capacity_problem(2, None) is None, "two streams of 4 pages and the sink page fit in 9"
    assert paged.capacity_problem(3, None) == (
        "3 concurrent requests (max_concurrent_requests) need 13 pages of resources.t3_kv (4 per request and the sink "
        "page), which has 9; raise max_num_pages, or lower max_concurrent_requests or the tokens a request may hold"
    )


def test_a_setting_refused_on_a_shared_spec_names_the_read_it_was_set_under():
    with pytest.raises(ValueError, match=r"Array 'kv': kernel_settings\['attn_image'\]: decode_tensor_cores must be "
                                         r"true or false"):
        build_specs([_per_walk()], kernel_settings={"attn_image": {"decode_tensor_cores": "false"}})
