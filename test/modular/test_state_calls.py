"""A layer reaches declared state through a handle.

The order of a step's calls is the contract: a read sees this layer's add, and an add after a read would change what an
earlier layer read. Nothing in a kernel would notice a broken order, so each rule raises here, naming the array and the
cause, in eager steps and at capture. A replay runs no Python, so it skips all of them.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

sys.path.insert(0, ".")

from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    CausalConv,
    Entry,
    Fork,
    GatedDeltaNet,
    KVEntry,
    Positions,
    Read,
    Rope,
    Sampler,
    SampleStep,
    TensorType,
    Value,
    at_last_token,
    causal_conv,
    gated_delta_net,
    kernel_setting,
    last_sampled,
    position,
    positions,
    rope,
    sample,
)
from mstar.engine.state.calls import ArrayHandle, SamplerHandle
from mstar.engine.state.within_step import WithinStepHandle

LAYERS, SLOTS, HEADS, DIM = 4, 64, 2, 8
TOKENS = 3


class _StubKV:
    """Holds real pages per layer and records writes instead of touching a plan."""

    def __init__(self, layers=LAYERS):
        self.pages = torch.zeros(layers, SLOTS, 2, HEADS, DIM)
        self.writes = []
        self.unopened = set()

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.writes.append((layer_idx, label))
        self.pages[layer_idx, :k.shape[0], 0] = k
        self.pages[layer_idx, :k.shape[0], 1] = v

    def opens_label(self, rid, label):
        return (rid, label) not in self.unopened


class _StubAttn:
    """Reads the pages it is handed, and records who asked."""

    requires_kv_write = True
    max_segments_per_step = 7

    def __init__(self):
        self.runs = []

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.runs.append((layer_idx, label, k is not None))
        return q + kv_cache_layer[:q.shape[0], 0]

    def select_last_hidden(self, x, label="main"):
        return x[-1:]


class _StubDense(_StubAttn):
    """Takes k and v at the read, as the dense backend does."""

    requires_kv_write = False


class _StubRagged:
    """A within-step kernel: only ever sees k and v at the read."""

    max_segments_per_step = 7

    def __init__(self):
        self.runs = []

    def run(self, q, k, v, label=None):
        self.runs.append(label)
        return q + k + v


class _StubPool:
    """Holds a real block per field and records scatters."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.scatters = []

    def block(self, name, i):
        return self.blocks[name][i]

    def scatter_(self, name, value, layer_idx, *, label):
        self.scatters.append((name, layer_idx, label, value.shape))


class _StubLinear:
    """Writes its state block in place, as the kernels do."""

    def __init__(self):
        self.labels = []

    def run_conv(self, x, conv_layer, weight, bias, label):
        self.labels.append(label)
        conv_layer[0] = 1
        return x * 2

    def run(self, q, k, v, a, b, state_layer, a_log, dt_bias, label):
        self.labels.append(label)
        state_layer[0] = 1
        return v + 1


class _StubPositions:
    def __init__(self):
        self.ids = torch.arange(TOKENS)
        self.seen = []

    def position(self, rid, label):
        return 5

    def pos_ids(self, label):
        self.seen.append(label)
        return self.ids

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        self.seen.append((label, rope_dtype))
        return q, k


def _ctx(capture=False, leased=False, rows=(1,)) -> StepContext:
    lease = SlotLease(slot=0, bucket=BucketKey("decode", len(rows), TOKENS)) if leased else None
    return StepContext(request_ids=rows, graph_walk="decode", slot=0, capture=capture, slot_lease=lease)


def _step(*handles, segments=None, ctx=None, **declared) -> SubmoduleStep:
    step = SubmoduleStep(
        segments=[Segment(1, "main", TOKENS)] if segments is None else segments,
        state={h.name: declared.get(h.name, SampleStep() if isinstance(h, SamplerHandle) else ArrayStep())
               for h in handles},
    )
    step.set_ctx(ctx or _ctx())
    return step


def _begin(*handles, **kwargs) -> SubmoduleStep:
    step = _step(*handles, **kwargs)
    for h in handles:
        h.begin_step(step)
    return step


def _array(**overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)), "entry": KVEntry(heads=HEADS, head_dim=DIM),
        "max_len": SLOTS, "reads": {"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))}, **overrides,
    })


def _kv_handle(array=None, kv=None, attn=None) -> ArrayHandle:
    array = array or _array()
    resources = {"kv": kv or _StubKV(len(array.layers)), **{key: attn or _StubAttn() for key in array.reads}}
    return ArrayHandle(array, resources, None)


def _value_handle(labels=frozenset({"main"})) -> ArrayHandle:
    entry = Entry({
        "state": TensorType((HEADS, DIM, DIM), dtype=torch.float32, shard=0),
        "conv": TensorType((3 * HEADS * DIM, 3), dtype=None, shard=0),
    })
    updates = {"c": CausalConv(), "d": GatedDeltaNet()}
    array = replace(Value("gdn", nodes={"LLM"}, layers={1, 5}, entry=entry, updates=updates), labels=frozenset(labels))
    pool = _StubPool({
        "state": torch.zeros(2, SLOTS, HEADS, DIM, DIM),
        "conv": torch.zeros(2, SLOTS, 3 * HEADS * DIM, 3),
    })
    return ArrayHandle(array, {"gdn": pool, "c": None, "d": _StubLinear()}, None)


def _within_handle(ragged=None) -> WithinStepHandle:
    array = _array(max_len=0, reads={"vision": Read(mask=Block(), op=Attend(q_heads=HEADS))})
    return WithinStepHandle(array, {"vision": ragged or _StubRagged()}, None)


def _sampler_handle(draws=1) -> SamplerHandle:
    resource = SamplerResource(vocab_size=8, enable_repetion_penalty=True, device=torch.device("cpu"))
    resource.ingest_request(1, None)
    return SamplerHandle(Sampler("sampler", nodes={"LLM"}, vocab_size=8, draws=draws), resource)


def _tensors(n=TOKENS):
    return torch.randn(n, HEADS, DIM), torch.randn(n, HEADS, DIM), torch.randn(n, HEADS, DIM)


# The calls


def test_a_read_sees_this_layers_add_through_the_layers_own_pages():
    kv, attn = _StubKV(), _StubAttn()
    h = _kv_handle(kv=kv, attn=attn)
    _begin(h)
    q, k, v = _tensors()
    h.add(2, k=k, v=v)
    out = h.read(2, q)
    assert kv.writes == [(2, "main")] and attn.runs == [(2, "main", False)]
    assert torch.equal(out, q + k), "the read should see the k this layer just added"
    assert kv.pages[0].abs().sum() == 0 and kv.pages[3].abs().sum() == 0, "the add should touch only its own layer"


def test_a_layer_id_is_the_models_and_the_pool_counts_among_the_arrays_layers():
    kv, attn = _StubKV(layers=2), _StubAttn()
    h = _kv_handle(_array(layers={3, 7}), kv=kv, attn=attn)
    _begin(h)
    q, k, v = _tensors()
    h.add(7, k=k, v=v)
    h.read(7, q)
    assert kv.writes == [(1, "main")] and attn.runs == [(1, "main", False)]


def test_a_label_comes_from_the_steps_segments():
    kv, attn = _StubKV(), _StubAttn()
    h = _kv_handle(_array(labels={"main", "cfg"}), kv=kv, attn=attn)
    _begin(h, segments=[Segment(1, "cfg", TOKENS)])
    q, k, v = _tensors()
    h.add(0, k=k, v=v)
    h.read(0, q)
    assert kv.writes == [(0, "cfg")] and attn.runs == [(0, "cfg", False)]


def test_an_array_steps_own_segments_win_over_the_steps():
    h = _kv_handle(_array(labels={"main", "cfg"}))
    _begin(h, kv=ArrayStep(segments=(Segment(1, "cfg", TOKENS),)))
    assert h.label == "cfg"


def test_a_kernel_that_takes_no_writes_gets_k_and_v_at_the_read():
    kv, attn = _StubKV(), _StubDense()
    h = _kv_handle(kv=kv, attn=attn)
    _begin(h)
    q, k, v = _tensors()
    h.add(1, k=k, v=v)
    h.read(1, q)
    assert kv.writes == [], "a dense read writes its own KV"
    assert attn.runs == [(1, "main", True)]


def test_a_within_step_read_passes_the_add_to_the_kernel_and_keeps_no_hold():
    ragged = _StubRagged()
    h = _within_handle(ragged)
    _begin(h)
    q, k, v = _tensors()
    h.add(0, k=k, v=v)
    assert torch.equal(h.read(0, q), q + k + v)
    assert ragged.runs == ["main"]
    assert h._held is None, "a hold would pin the whole projection k and v are views of"


def test_a_within_step_add_nobody_read_goes_at_the_end_of_the_step():
    h = _within_handle()
    step = _begin(h)
    _, k, v = _tensors()
    h.add(0, k=k, v=v)
    assert h._held is not None
    h.end_step(step)
    assert h._held is None, "a hold after the step would pin its tensors until the node's next step"


def test_scratch_adds_can_repeat_before_the_kept_add():
    kv = _StubKV()
    h = _kv_handle(kv=kv)
    _begin(h)
    _, k, v = _tensors()
    h.add(0, keep=False, k=k, v=v)
    h.add(0, keep=False, k=k, v=v)
    h.add(0, k=k, v=v)
    assert len(kv.writes) == 3


def test_a_step_that_keeps_nothing_makes_every_add_scratch():
    h = _kv_handle()
    _begin(h, kv=ArrayStep(keep=False))
    _, k, v = _tensors()
    h.add(0, k=k, v=v)
    h.add(0, k=k, v=v)


def test_write_overwrites_a_value_field_in_its_stored_layout():
    h = _value_handle()
    _begin(h)
    h.write(5, state=torch.zeros(TOKENS, HEADS, DIM, DIM + 1))
    assert h._pool.scatters == [("state", 1, "main", (TOKENS, HEADS, DIM + 1, DIM))], "state is stored K-last"


def test_updates_run_on_the_layers_own_field_and_write_it_in_place():
    h = _value_handle()
    _begin(h)
    x = torch.randn(TOKENS, 3 * HEADS * DIM)
    out = causal_conv(h, 5, x, weight=torch.ones(3 * HEADS * DIM, 4))
    assert torch.equal(out, x * 2)
    assert h._pool.blocks["conv"][1].abs().sum() > 0 and h._pool.blocks["conv"][0].abs().sum() == 0
    q, k, v = _tensors()
    core = gated_delta_net(h, 1, q, k, v, q, q, q, q)
    assert torch.equal(core, v + 1)
    assert h._pool.blocks["state"][0].abs().sum() > 0 and h._pool.blocks["state"][1].abs().sum() == 0


def test_positions_read_the_resource_at_the_steps_label():
    pos = _StubPositions()
    array = _array(labels={"main", "cfg"}, positions=Positions(rope=Rope(dtype=torch.bfloat16)))
    h = ArrayHandle(array, {"kv": _StubKV(), "attn": _StubAttn(), "rope": pos}, "rope")
    _begin(h, segments=[Segment(1, "cfg", TOKENS)])
    assert position(h, 1, "cfg") == 5
    assert torch.equal(positions(h), pos.ids)
    q, k, _ = _tensors()
    assert rope(h, q, k) == (q, k)
    assert pos.seen == ["cfg", ("cfg", torch.bfloat16)], "each should read the step's label, and a declared dtype"


def test_rope_with_no_declared_dtype_rotates_in_qs():
    pos = _StubPositions()
    resources = {"kv": _StubKV(), "attn": _StubAttn(), "rope": pos}
    h = ArrayHandle(_array(positions=Positions(rope=Rope())), resources, "rope")
    _begin(h)
    q, k, _ = _tensors()
    rope(h, q.double(), k)
    assert pos.seen == [("main", torch.float64)]


def test_at_last_token_and_kernel_setting_read_the_reads_resource():
    h = _kv_handle()
    _begin(h)
    x = torch.arange(3.0)
    assert at_last_token(h, x).tolist() == [2.0]
    assert kernel_setting(h, "attn", "max_segments_per_step") == 7


def test_sample_takes_the_steps_rows_padding_included():
    h = _sampler_handle()
    seen = []
    h.resource.sample = lambda rows, logits: seen.append(rows) or logits.argmax(-1)
    ctx = _ctx(rows=(1,))
    ctx.set_padded_rids((1, -1))
    _begin(h, ctx=ctx)
    assert sample(h, torch.eye(2, 8)).tolist() == [0, 1]
    assert seen == [[1, -1]], "a padding row must stay in the rows, since the logits have a row for it"


def test_last_sampled_reads_the_resource_with_the_steps_rows():
    h = _sampler_handle()
    h.resource.loopback_tokens = torch.tensor
    _begin(h, ctx=_ctx(rows=(4, 5)))
    assert last_sampled(h).tolist() == [4, 5]


def test_two_draws_in_a_step_draw_at_different_offsets_and_a_third_raises(monkeypatch):
    offsets = []

    def draw(logits, *, rand_offset, **kwargs):
        offsets.append(int(rand_offset[0]))
        return torch.zeros(logits.shape[0], dtype=torch.long)

    monkeypatch.setattr(sampler_utils, "sample_tokens", draw)
    h = _sampler_handle(draws=2)
    h.resource.plan(SamplerStep(), _ctx())
    step = _begin(h)
    logits = torch.randn(1, 8)
    sample(h, logits)
    sample(h, logits)
    assert offsets[0] != offsets[1], "a second draw at the first's offset would repeat its randomness"
    with pytest.raises(ValueError, match=r"Sampler 'sampler': sample number 3 in one step, and draws is 2"):
        sample(h, logits)


def test_a_step_that_samples_too_few_times_is_refused_by_its_check():
    h = _sampler_handle(draws=2)
    h.resource.sample = lambda rows, logits: logits.argmax(-1)
    step = _begin(h)
    sample(h, torch.randn(1, 8))
    with pytest.raises(ValueError, match=r"Sampler 'sampler': a step listing it sampled 1 times, and draws is 2"):
        h.check_step(step)


def test_a_replay_checks_no_draws_and_inherits_none_from_a_failed_step():
    h = _sampler_handle()
    h.resource.sample = lambda rows, logits: logits.argmax(-1)
    failed = _begin(h)
    sample(h, torch.randn(1, 8))
    h.end_step(failed)
    replay = _begin(h, ctx=_ctx(leased=True, rows=(7,)))
    h.check_step(replay)
    assert h.rows == (7,) and h._drawn == 0, "the replay's rows, and no draws carried over"


@pytest.mark.parametrize("make", [_kv_handle, _sampler_handle], ids=["array", "sampler"])
def test_a_begin_on_an_open_handle_raises(make):
    h = make()
    step = _begin(h)
    _refused(rf"'{h.name}': a step began while another is open", lambda: h.begin_step(step))
    h.end_step(step)
    h.begin_step(step)


def test_a_capture_row_passes_the_opened_label_check_on_any_label():
    h = _kv_handle()
    h._kv.opens_label = lambda rid, label: False
    _begin(h, ctx=_ctx(capture=True))
    h.end_step(_step(h))
    _refused(r"Array 'kv': a step touches label 'main' of request 1, which it didn't open", lambda: _begin(h))


# Refusals: one per rule, each naming the array and the cause


def _refused(pattern: str, call) -> None:
    with pytest.raises(ValueError, match=pattern):
        call()


def test_an_add_on_an_array_the_step_does_not_list_raises():
    h = _kv_handle()
    _, k, v = _tensors()
    _refused(r"Array 'kv': add in a step that doesn't list the array", lambda: h.add(0, k=k, v=v))


def test_a_read_on_an_array_the_step_does_not_list_raises():
    h = _kv_handle()
    q, _, _ = _tensors()
    _refused(r"Array 'kv': read in a step that doesn't list the array", lambda: h.read(0, q))


def test_an_update_on_an_array_the_step_does_not_list_raises():
    h = _value_handle()
    q, k, v = _tensors()
    _refused(r"Array 'gdn': the update of state in a step that doesn't list",
             lambda: gated_delta_net(h, 1, q, k, v, q, q, q, q))


def test_a_sample_on_a_sampler_the_step_does_not_list_raises():
    h = _sampler_handle()
    _refused(r"Sampler 'sampler': sample in a step that doesn't list", lambda: sample(h, torch.ones(1, 8)))


def test_a_call_after_the_steps_end_raises():
    h = _kv_handle()
    h.end_step(_begin(h))
    _, k, v = _tensors()
    _refused(r"doesn't list the array", lambda: h.add(0, k=k, v=v))


def test_a_call_on_an_undeclared_layer_raises():
    h = _kv_handle()
    _begin(h)
    q, k, v = _tensors()
    _refused(r"Array 'kv': add on layer 9, which the array doesn't declare", lambda: h.add(9, k=k, v=v))
    _refused(r"Array 'kv': read on layer 9, which the array doesn't declare", lambda: h.read(9, q))


def test_a_within_step_call_on_an_undeclared_layer_raises():
    h = _within_handle()
    _begin(h)
    _, k, v = _tensors()
    _refused(r"Array 'kv': add on layer 9, which the array doesn't declare", lambda: h.add(9, k=k, v=v))


def test_an_update_on_an_undeclared_layer_raises():
    h = _value_handle()
    _begin(h)
    _refused(r"Array 'gdn': no field 'conv' on layer 3",
             lambda: causal_conv(h, 3, torch.ones(TOKENS, 6), weight=torch.ones(6, 4)))


def test_an_add_after_a_kept_add_raises():
    h = _kv_handle()
    _begin(h)
    _, k, v = _tensors()
    h.add(0, k=k, v=v)
    _refused(r"Array 'kv': an add on layer 0 after its kept add", lambda: h.add(0, k=k, v=v))


def test_an_add_after_a_read_in_the_same_layer_raises():
    h = _kv_handle()
    _begin(h)
    q, k, v = _tensors()
    h.add(0, keep=False, k=k, v=v)
    h.read(0, q)
    _refused(r"Array 'kv': an add on layer 0 after a read in the same layer", lambda: h.add(0, keep=False, k=k, v=v))


def test_write_on_an_append_only_array_raises():
    h = _kv_handle()
    _begin(h)
    _refused(r"Array 'kv': write on an append-only array", lambda: h.write(0, k=torch.ones(1)))


def test_write_at_positions_has_no_layout_yet():
    h = _value_handle()
    _begin(h)
    _refused(r"Array 'gdn': no layout yet: write at positions",
             lambda: h.write(1, at=torch.tensor([0]), state=torch.ones(1)))


def test_keep_true_on_max_len_zero_raises():
    h = _within_handle()
    _begin(h)
    _, k, v = _tensors()
    _refused(r"Array 'kv': keep=True, and max_len 0 keeps nothing", lambda: h.add(0, keep=True, k=k, v=v))


def test_a_within_step_read_with_no_add_raises():
    h = _within_handle()
    _begin(h)
    q, _, _ = _tensors()
    _refused(r"Array 'kv': a read on layer 0 with no add before it", lambda: h.read(0, q))


def test_a_second_update_of_a_field_in_a_layer_raises():
    h = _value_handle()
    _begin(h)
    x, weight = torch.ones(TOKENS, 6), torch.ones(6, 4)
    causal_conv(h, 1, x, weight=weight)
    _refused(r"Array 'gdn': a second update of conv on layer 1", lambda: causal_conv(h, 1, x, weight=weight))


def test_a_read_with_no_op_or_with_mask_arguments_has_no_layout_yet():
    h = _kv_handle()
    _begin(h)
    q, _, _ = _tensors()
    _refused(r"Array 'kv': no layout yet: a read with no op", lambda: h.read(0))
    _refused(r"Array 'kv': no layout yet: mask arguments \['top_k'\]", lambda: h.read(0, q, top_k=3))


def test_a_step_on_an_undeclared_label_raises():
    h = _kv_handle()
    _refused(r"Array 'kv': a step touches labels \['cfg'\], and the array declares \['main'\]",
             lambda: _begin(h, segments=[Segment(1, "cfg", TOKENS)]))


def test_a_step_on_two_labels_of_one_array_has_no_layout_yet():
    h = _kv_handle(_array(labels={"main", "cfg"}))
    _refused(r"Array 'kv': no layout yet: a step touches labels \['cfg', 'main'\]",
             lambda: _begin(h, segments=[Segment(1, "main", TOKENS), Segment(1, "cfg", TOKENS)]))


def test_a_fork_on_an_undeclared_label_raises():
    h = _kv_handle()
    fork = Fork(src="main", dst="cfg", at="start")
    _refused(r"Array 'kv': a fork names labels \['cfg'\]", lambda: _begin(h, kv=ArrayStep(forks=(fork,))))


def test_a_row_on_a_label_its_request_did_not_open_raises():
    kv = _StubKV()
    kv.unopened.add((1, "main"))
    h = _kv_handle(kv=kv)
    _refused(r"Array 'kv': a step touches label 'main' of request 1, which it didn't open", partial(_begin, h))


def test_each_replay_starts_clean_for_an_add_that_runs_outside_the_graph():
    h = _kv_handle()
    for _ in range(2):
        step = _begin(h, ctx=_ctx(leased=True), kv=ArrayStep(eager_add=True))
        _add(h, 1)
        h.end_step(step)


@pytest.mark.parametrize("make, declared, cause", [
    (_kv_handle, ArrayStep(advance=(1,)), "advance or position_ids"),
    (_kv_handle, ArrayStep(position_ids=torch.zeros(1)), "advance or position_ids"),
    (_value_handle, ArrayStep(placeholders=frozenset({(1, "main")})), "placeholders or eager_add"),
    (_value_handle, ArrayStep(eager_add=True), "placeholders or eager_add"),
    (_within_handle, ArrayStep(forks=(Fork(src="main", dst="main", at="start"),)), "forks an array with max_len 0"),
], ids=["advance", "position_ids", "placeholders", "eager_add", "forks"])
def test_a_step_field_the_array_cannot_use_is_refused(make, declared, cause):
    h = make()
    _refused(cause, lambda: _begin(h, **{h.name: declared}))


@pytest.mark.parametrize("call", [
    positions,
    lambda h: rope(h, torch.zeros(1), torch.zeros(1)),
    lambda h: at_last_token(h, torch.zeros(1)),
], ids=["positions", "rope", "at_last_token"])
def test_a_read_of_this_steps_plan_on_an_unlisted_array_raises(call):
    _refused(r"Array 'kv': .* in a step that doesn't list the array", lambda: call(_kv_handle()))


def test_last_sampled_on_an_unlisted_sampler_raises():
    _refused("last_sampled in a step that doesn't list the sampler", lambda: last_sampled(_sampler_handle()))


def test_a_placeholder_row_may_touch_a_label_its_request_did_not_open():
    kv = _StubKV()
    kv.unopened.add((1, "main"))
    h = _kv_handle(kv=kv)
    _begin(h, kv=ArrayStep(placeholders=frozenset({(1, "main")})))


def test_keep_false_on_a_value_with_updates_raises():
    h = _value_handle()
    _refused(r"Array 'gdn': keep=False on an array with updates", lambda: _begin(h, gdn=ArrayStep(keep=False)))


def test_a_replay_skips_every_step_check():
    kv = _StubKV()
    kv.unopened.add((1, "main"))
    h = _kv_handle(kv=kv)
    step = _begin(h, segments=[Segment(1, "cfg", TOKENS)], ctx=_ctx(leased=True))
    assert h.label == "cfg", "preprocess runs on a replay too, and reads the step's label"
    h.end_step(step)


def test_a_replay_doesnt_answer_for_the_scratch_adds_of_a_failed_step():
    h = _kv_handle()
    failed = _begin(h)
    _add(h, 1, keep=False)
    h.end_step(failed)
    h.check_step(_begin(h, ctx=_ctx(leased=True)))


def test_capture_runs_the_checks():
    h = _kv_handle()
    _refused(r"a step touches labels \['cfg'\]",
             lambda: _begin(h, segments=[Segment(1, "cfg", TOKENS)], ctx=_ctx(capture=True, leased=True)))


# Under compile


class _Block(torch.nn.Module):
    def __init__(self, layer, h):
        super().__init__()
        self.layer, self.h = layer, h

    def forward(self, x):
        self.h.add(self.layer, k=x * 2, v=x + 1)
        return self.h.read(self.layer, x)


@torch.no_grad()
def test_blocks_compiled_one_by_one_share_a_graph_and_leave_no_hold():
    """The ViT compiles each block alone so they reuse one traced graph, which a guard on the layer would undo."""
    torch._dynamo.reset()
    calls = []

    @torch.library.custom_op("mstar_test::ragged", mutates_args=())
    def ragged(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return q + k + v

    @ragged.register_fake
    def _(q, k, v):
        return q.new_empty(q.shape)

    class _Op(_StubRagged):
        def run(self, q, k, v, label=None):
            return ragged(q, k, v)

    h = _within_handle(_Op())
    _begin(h)
    blocks = [_Block(i, h) for i in range(LAYERS)]
    for block in blocks:
        block.forward = torch.compile(block.forward, dynamic=True)
    x = torch.randn(TOKENS, HEADS, DIM)
    blocks[0](x)
    with torch.compiler.set_stance("fail_on_recompile"):
        for block in blocks[1:]:
            block(x)
        for block in blocks:
            block(torch.randn(TOKENS + 2, HEADS, DIM))
    assert len(calls) == 2 * LAYERS
    assert h._held is None, "a compiled block should not return k and v from its graph"


# What a step keeps, which the resources decide for the step as a whole


def _add(h, layer, **kwargs) -> None:
    _, k, v = _tensors()
    h.add(layer, k=k, v=v, **kwargs)


def test_a_scratch_add_then_a_kept_add_passes_the_check():
    h = _kv_handle()
    step = _begin(h)
    _add(h, 1, keep=False)
    for layer in range(LAYERS):
        _add(h, layer)
    h.check_step(step)


def test_a_step_that_keeps_adds_on_only_some_layers_is_refused():
    h = _kv_handle()
    step = _begin(h)
    _add(h, 1)
    _refused(r"Array 'kv': layers \[0, 2, 3\] have no add in a step that keeps adds on others",
             lambda: h.check_step(step))


def test_a_step_that_adds_on_no_layer_only_reads_and_passes():
    h = _kv_handle()
    h.check_step(_begin(h))


def test_a_layer_ending_on_a_scratch_add_in_a_step_that_keeps_is_refused():
    h = _kv_handle()
    step = _begin(h)
    _add(h, 1, keep=False)
    _refused(r"Array 'kv': layers \[1\] end on a scratch add in a step that keeps its adds", lambda: h.check_step(step))


def test_keep_true_in_a_step_with_keep_false_is_refused():
    h = _kv_handle()
    _begin(h, kv=ArrayStep(keep=False))
    _refused(r"Array 'kv': keep=True on layer 1 in a step with keep=False", lambda: _add(h, 1, keep=True))


def test_a_step_with_keep_false_passes_its_check_with_scratch_adds():
    h = _kv_handle()
    step = _begin(h, kv=ArrayStep(keep=False))
    _add(h, 1)
    _add(h, 1, keep=False)
    h.check_step(step)


def test_a_read_cannot_take_the_add_held_for_another_layer():
    h = _within_handle()
    _begin(h)
    q, k, v = _tensors()
    h.add(0, k=k, v=v)
    h.add(1, k=k, v=v)
    _refused(r"Array 'kv': a read on layer 0 would take the add of layer 1", lambda: h.read(0, q))


# The layer's two forms


def test_every_call_takes_the_layer_as_the_int_or_its_cpu_tensor():
    kv = _kv_handle()
    _begin(kv)
    q, k, v = _tensors()
    kv.add(torch.tensor(1), k=k, v=v)
    kv.read(torch.tensor(1), q)
    kv.add(2, k=k, v=v)
    kv.read(2, q)
    assert kv._kv.writes == [(1, "main"), (2, "main")], "the tensor form should name the layer the int does"
    assert kv._read == {1, 2}

    gdn = _value_handle()
    _begin(gdn)
    x = torch.randn(TOKENS, 3 * HEADS * DIM)
    causal_conv(gdn, torch.tensor(1), x, weight=torch.ones(3 * HEADS * DIM, 4))
    gated_delta_net(gdn, torch.tensor(5), q, k, v, q, q, q, q)
    assert gdn._updated == {("conv", 1), ("state", 5)}
    gdn._pool.scatter_ = Mock()
    gdn.write(torch.tensor(5), conv=torch.ones(1, 3 * HEADS * DIM, 3))
    assert gdn._pool.scatter_.call_args.args[2] == 1, "layer 5 is the second of the Value's declared layers"


@pytest.mark.parametrize("layer", [
    torch.tensor([1]), torch.tensor(1.0), torch.tensor(1, dtype=torch.int32), torch.tensor(1, device="meta"),
], ids=["not_0d", "a_float", "not_int64", "off_the_cpu"])
def test_every_call_refuses_any_other_layer_tensor_naming_the_array(layer):
    cause = r"a layer tensor must be 0-d int64 on the CPU"
    kv = _kv_handle()
    _begin(kv)
    q, k, v = _tensors()
    _refused(rf"Array 'kv': {cause}", lambda: kv.add(layer, k=k, v=v))
    _refused(rf"Array 'kv': {cause}", lambda: kv.read(layer, q))
    gdn = _value_handle()
    _begin(gdn)
    _refused(rf"Array 'gdn': {cause}", lambda: causal_conv(gdn, layer, q, weight=q))
    _refused(rf"Array 'gdn': {cause}", lambda: gated_delta_net(gdn, layer, q, k, v, q, q, q, q))
    _refused(rf"Array 'gdn': {cause}", lambda: gdn.write(layer, conv=q))
    within = _within_handle()
    _begin(within)
    _refused(rf"Array 'kv': {cause}", lambda: within.add(layer, k=k, v=v))
    _refused(rf"Array 'kv': {cause}", lambda: within.read(layer, q))


def test_every_call_reads_the_steps_label():
    q, k, v = _tensors()
    gdn = _value_handle(labels={"main", "cfg"})
    _begin(gdn, segments=[Segment(1, "cfg", TOKENS)])
    causal_conv(gdn, 1, torch.randn(TOKENS, 3 * HEADS * DIM), weight=torch.ones(3 * HEADS * DIM, 4))
    gated_delta_net(gdn, 1, q, k, v, q, q, q, q)
    assert gdn._linear.labels == ["cfg", "cfg"]
    ragged = _StubRagged()
    array = _array(max_len=0, labels={"main", "cfg"}, reads={"vision": Read(mask=Block(), op=Attend(q_heads=HEADS))})
    within = WithinStepHandle(array, {"vision": ragged}, None)
    _begin(within, segments=[Segment(1, "cfg", TOKENS)])
    within.add(0, k=k, v=v)
    within.read(0, q)
    assert ragged.runs == ["cfg"]


# Where the graph breaks


def _compiling_at_each_kernel(h_calls, record) -> list[bool]:
    """What each kernel saw of `is_compiling` when a compiled caller made ``h_calls``, which a traced call sees as True
    and a call behind a break as False."""
    torch._dynamo.reset()
    seen = []
    record(lambda: seen.append(torch.compiler.is_compiling()))

    def caller(x):
        return h_calls(x * 2)

    torch._dynamo.optimize("eager", dynamic=False)(caller)(torch.randn(TOKENS, HEADS, DIM))
    return seen


@torch.no_grad()
def test_every_call_on_a_kept_array_runs_behind_a_graph_break():
    """A model writes no `torch.compiler.disable`, so a call that lost its break would trace into the model's frame."""
    kv = _kv_handle()
    _begin(kv)
    _, k, v = _tensors()

    def attend(x):
        kv.add(1, k=k, v=v)
        return kv.read(1, x)

    def record(note):
        write, run = kv._kv.write_kv, kv.first_read.run
        kv._kv.write_kv = lambda *args, **kwargs: (note(), write(*args, **kwargs))[1]
        kv.first_read.run = lambda *args, **kwargs: (note(), run(*args, **kwargs))[1]

    assert _compiling_at_each_kernel(attend, record) == [False, False]


@torch.no_grad()
def test_every_update_runs_behind_a_graph_break():
    h = _value_handle()
    _begin(h)
    width = 3 * HEADS * DIM
    _, k, v = _tensors()

    def updates(x):
        causal_conv(h, 1, torch.zeros(TOKENS, width), weight=torch.ones(width, 4))
        h.write(1, conv=torch.ones(1, width, 3))
        return gated_delta_net(h, 5, x, k, v, x, x, x, x)

    def record(note):
        conv, delta, scatter = h._linear.run_conv, h._linear.run, h._pool.scatter_
        h._linear.run_conv = lambda *args, **kwargs: (note(), conv(*args, **kwargs))[1]
        h._linear.run = lambda *args, **kwargs: (note(), delta(*args, **kwargs))[1]
        h._pool.scatter_ = lambda *args, **kwargs: (note(), scatter(*args, **kwargs))[1]

    assert _compiling_at_each_kernel(updates, record) == [False, False, False]


@torch.no_grad()
def test_a_sample_runs_behind_a_graph_break():
    h = _sampler_handle()
    _begin(h)
    resource = SimpleNamespace(sample=None)

    def record(note):
        resource.sample = lambda rows, logits: (note(), logits.argmax(-1))[1]
        h.resource = resource

    assert _compiling_at_each_kernel(lambda x: sample(h, x), record) == [False]


@torch.no_grad()
def test_a_within_step_handle_adds_no_break_of_its_own():
    torch._dynamo.reset()
    h = _within_handle()
    _begin(h)
    _, k, v = _tensors()
    graphs = []

    def backend(graph_module, inputs):
        graphs.append(graph_module)
        return graph_module.forward

    block = torch._dynamo.optimize(backend, dynamic=True)(_Block(0, h))
    block(torch.randn(TOKENS, HEADS, DIM))
    assert len(graphs) == 1, "a break in the add or the read would split a block compiled alone"
