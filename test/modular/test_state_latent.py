"""An MLA latent is one row per token that every query head reads, not k and v per head.

GLM-5.2 and GLM-5.3 keep it in FlashInfer's MLA layout, ``ckv`` then the rope key ``kpe`` in one row, and attend it
with the absorbed query and the rope query. The latent entry and `LatentAttend` declare that, and the latent storage
builds the pool and the attention GLM writes by hand, joins an add's fields into the row and refuses what the MLA
layout can't hold.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttnBackend
from mstar.engine.resources.kv.config import KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Causal,
    Entry,
    LatentAttend,
    LatentEntry,
    PagedLatent,
    PagedLatentTarget,
    PagedNHD,
    Positions,
    Read,
    Rope,
    TensorType,
    Window,
    at_last_token,
    state_op,
)
from mstar.engine.state.adapters import LATENT, StateSetup, _resolve, build_handle, build_specs, storage_of
from mstar.engine.state.calls import ArrayHandle

state_op_module = sys.modules["mstar.engine.state.state_op"]

LAYERS, SLOTS, HEADS, RANK, ROPE = 3, 32, 4, 16, 8
TOKENS = 3
SCALE = 0.125


class _StubLatentKV:
    """Holds real MLA pages per layer and records writes instead of touching a plan."""

    def __init__(self, width=RANK + ROPE):
        self.pages = torch.zeros(LAYERS, SLOTS, width)
        self.writes = []

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v=None, layer_idx=None, label=None, tokens=None):
        self.writes.append((layer_idx, label, v))
        self.pages[layer_idx, :k.shape[0]] = k

    def opens_label(self, rid, label):
        return True


class _StubMLA:
    """Records each absorbed read and returns the rows it read."""

    requires_kv_write = True

    def __init__(self):
        self.runs = []

    def run(self, q, label=None, kv_cache_layer=None, q_pe=None):
        self.runs.append((label, q_pe.shape))
        return kv_cache_layer[:q.shape[0], :RANK].unsqueeze(1).expand_as(q)

    def select_last_hidden(self, x, label="main"):
        return x[-1:]


def _latent(rope_dim=ROPE, **overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)), "entry": LatentEntry(rank=RANK, rope_dim=rope_dim),
        "max_len": SLOTS, "reads": {"attn": Read(mask=Causal(), op=LatentAttend(HEADS, scale=SCALE))}, **overrides,
    })


def _handle(array=None, kv=None, attn=None):
    array = array or _latent()
    width = sum(t.shape[0] for t in array.entry.tensors.values())
    resources = {"kv": kv or _StubLatentKV(width), "attn": attn or _StubMLA()}
    return build_handle(array, resources, _resolve([array], StateSetup()))


def _begin(h):
    step = SubmoduleStep(segments=[Segment(1, "main", TOKENS)], state={h.name: ArrayStep()})
    step.set_ctx(StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False))
    h.begin_step(step)
    return step


def _refused(pattern: str, call) -> None:
    with pytest.raises(ValueError, match=pattern):
        call()


def test_a_latent_entry_refuses_bad_sizes():
    _refused("rank must be a positive int", lambda: LatentEntry(rank=0, rope_dim=ROPE))
    _refused("rope_dim must be an int >= 0", lambda: LatentEntry(rank=RANK, rope_dim=-1))
    assert set(LatentEntry(rank=RANK, rope_dim=0).tensors) == {"ckv"}, "a latent with no rope key has no kpe"


def test_latent_attend_refuses_its_own_bad_fields():
    def latent(op, entry=None):
        return lambda: _latent(entry=entry or LatentEntry(rank=RANK, rope_dim=ROPE),
                               reads={"attn": Read(mask=Causal(), op=op)})

    _refused("q_heads must be a positive int", latent(LatentAttend(0, scale=SCALE)))
    _refused("scale must be a positive float", latent(LatentAttend(HEADS, scale=0.0)))
    _refused("scale must be a positive float", latent(LatentAttend(HEADS, scale=1)))
    flat = TensorType((RANK,), dtype=None)
    _refused("ckv 1-D and unsharded", latent(LatentAttend(HEADS, scale=SCALE),
                                             Entry({"ckv": TensorType((2, RANK), dtype=None)})))
    _refused("kpe 1-D and unsharded", latent(LatentAttend(HEADS, scale=SCALE),
                                             Entry({"ckv": flat, "kpe": TensorType((ROPE,), dtype=None, shard=0)})))
    _refused("ckv and kpe of one dtype", latent(LatentAttend(HEADS, scale=SCALE),
                                                Entry({"ckv": flat, "kpe": TensorType((ROPE,), dtype=torch.float32)})))


def test_the_latent_pool_holds_only_a_latent_entry():
    extra = Entry({**LatentEntry(rank=RANK, rope_dim=ROPE).tensors, "k": TensorType((RANK,), dtype=None)})
    assert storage_of(_latent(entry=extra)) is LATENT
    _refused("entry must be a LatentEntry", lambda: build_specs([_latent(entry=extra)]))
    _refused("labels \\['a\\+b'\\] hold a '\\+'", lambda: build_specs([_latent(labels={"a+b"})]))
    two = {"a": Read(mask=Causal(), op=LatentAttend(HEADS, scale=SCALE), layers={0}),
           "b": Read(mask=Causal(), op=LatentAttend(2 * HEADS, scale=SCALE), layers={1, 2})}
    _refused("q_heads \\[4, 8\\]", lambda: build_specs([_latent(reads=two)]))


def test_a_latent_read_takes_causal_rows():
    def read(**fields):
        return lambda: build_specs([_latent(reads={"attn": Read(op=LatentAttend(HEADS, scale=SCALE), **fields)})])

    _refused("mask must be Causal", read(mask=Window(4)))
    _refused("walks must be None", read(mask=Causal(), walks={"decode"}))
    _refused("packed=False", read(mask=Causal(), packed=False))


def test_a_latent_array_builds_the_mla_pool_and_attention_glm_writes():
    specs = {spec.resource_key: spec for spec in build_specs([_latent()])}
    assert specs["kv"] == KVSpec(resource_key="kv", nodes=frozenset({"LLM"}), config=PagedKVConfig(
        num_layers=LAYERS, num_kv_heads=1, head_dim=RANK + ROPE, max_seq_len=SLOTS, num_qo_heads=HEADS,
        layout=KVLayout.MLA, kv_lora_rank=RANK, qk_rope_head_dim=ROPE,
    )), "the pool should be the MLA config GLM-5.2 writes by hand"
    assert specs["attn"] == AttentionSpec(resource_key="attn", nodes=frozenset({"LLM"}), config=AttentionConfig(
        kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=SCALE,
    )), "the read should run FlashInfer's MLA attention at the declared scale"


def test_a_latent_add_writes_one_row_per_token():
    kv = _StubLatentKV()
    h = _handle(kv=kv)
    _begin(h)
    ckv, kpe = torch.randn(TOKENS, RANK), torch.randn(TOKENS, ROPE)
    h.add(1, ckv=ckv, kpe=kpe)
    assert kv.writes == [(1, "main", None)], "a latent add should be one write of the row, with no v"
    assert torch.equal(kv.pages[1, :TOKENS], torch.cat([ckv, kpe], -1)), "the row should be ckv then kpe"

    plain = _StubLatentKV(RANK)
    h = _handle(_latent(rope_dim=0), kv=plain)
    _begin(h)
    h.add(0, ckv=ckv)
    assert torch.equal(plain.pages[0, :TOKENS], ckv), "a latent with no kpe should write ckv alone"


def test_a_latent_add_takes_exactly_the_entry_s_fields():
    h = _handle()
    _begin(h)
    ckv = torch.randn(TOKENS, RANK)
    _refused("an add of fields \\['ckv'\\], and the latent entry holds \\['ckv', 'kpe'\\]", lambda: h.add(0, ckv=ckv))
    _refused("an add of fields \\['k', 'v'\\]", lambda: h.add(0, k=ckv, v=ckv))


def test_a_latent_read_takes_the_keywords_its_op_defines():
    attn = _StubMLA()
    h = _handle(attn=attn)
    _begin(h)
    q, q_pe = torch.randn(TOKENS, HEADS, RANK), torch.randn(TOKENS, HEADS, ROPE)
    h.add(0, ckv=torch.randn(TOKENS, RANK), kpe=torch.randn(TOKENS, ROPE))
    _refused("a read without q_pe, on a latent of rope_dim 8", lambda: h.read(0, q))
    _refused("read arguments \\['weights'\\]", lambda: h.read(0, q, q_pe=q_pe, weights=q))
    _refused("selected, on a read whose mask picks no positions", lambda: h.read(0, q, q_pe=q_pe, selected=q))
    h.read(0, q, q_pe=q_pe)
    assert attn.runs == [("main", q_pe.shape)], "the read should pass its rope query to the MLA kernel"

    attn = _StubMLA()
    h = _handle(_latent(rope_dim=0), attn=attn)
    _begin(h)
    h.add(0, ckv=torch.randn(TOKENS, RANK))
    _refused("a read with q_pe, on a latent of rope_dim 0", lambda: h.read(0, q, q_pe=q_pe))
    h.read(0, q)
    assert attn.runs == [("main", (TOKENS, HEADS, 0))], "a latent with no kpe should hand the kernel an empty q_pe"


def test_a_latent_step_over_two_labels_has_no_layout_yet():
    h = _handle(_latent(labels={"main", "other"}))
    step = SubmoduleStep(segments=[Segment(1, "main", 1), Segment(1, "other", 1)], state={"kv": ArrayStep()})
    step.set_ctx(StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False))
    _refused("an MLA plan holds one label", lambda: h.begin_step(step))


def test_at_last_token_on_a_latent_array_asks_the_mla_plan():
    h = _handle()
    _begin(h)
    x = torch.randn(TOKENS, 2)
    assert torch.equal(at_last_token(h, x), x[-1:]), "the MLA resource's plan should select each row's last token"


# The latent state-op format


class _StubMLAPages(_StubLatentKV):
    """Pages in the MLA layout, (pages, page_size, rank + rope_dim) per layer, where a step writes token t to page 0,
    slot t."""

    def __init__(self):
        super().__init__()
        self.pages = torch.zeros(LAYERS, 2, SLOTS, RANK + ROPE)
        self.kv_cache = SimpleNamespace(layout=KVLayout.MLA, dtype=self.pages.dtype, device=self.pages.device,
                                        tensor=self.pages)

    def write_kv(self, k, v=None, layer_idx=None, label=None, tokens=None):
        self.pages[layer_idx, 0, :k.shape[0]] = k

    def write_slots(self, label=None):
        return torch.zeros(TOKENS, dtype=torch.long), torch.arange(TOKENS)


class _StubRope:
    """The table and pairing of a position resource, which the fused prep rotates with."""

    def __init__(self, interleave=True, table=True):
        self.interleave = interleave
        self.cos_sin_table = torch.ones(SLOTS, ROPE) if table else None


class _NoPages(ArrayHandle):
    """A handle over a storage that keeps no pages."""

    def _check_storage(self, declared, segments, ctx):
        pass


@pytest.fixture
def _op(monkeypatch):
    """Makes a state op in ``formats`` that writes the latent rows itself, in an empty registry."""
    monkeypatch.setattr(state_op_module, "STATE_OPS", {})

    def make(*formats, kernel=None):
        def write_latent(kv, layer, ckv, kpe, *, target):
            target.cache[target.page_of_token, target.slot_of_token] = torch.cat([ckv, kpe], -1)

        def reference(kv, layer, ckv, kpe):
            kv.add(layer, ckv=ckv, kpe=kpe)

        return state_op(adds=("kv",), formats=formats, reference=reference)(kernel or write_latent)

    return make


def _rows():
    return torch.ones(TOKENS, RANK), torch.ones(TOKENS, ROPE)


def test_the_latent_handle_takes_the_fused_prep_on_mla_pages(_op):
    write = _op(PagedLatent())
    h = _handle(kv=_StubMLAPages())
    assert h.takes_fused(write.op), "a latent op would run its reference on the pages it addresses"
    _begin(h)
    target = h.fused_target(1)
    assert isinstance(target, PagedLatentTarget) and target.count == TOKENS and target.rope_table is None
    assert target.cache.data_ptr() == h._kv.pages[1].data_ptr() and 1 in h._kept, (
        "the fused op is layer 1's add, into layer 1's pages"
    )


def test_a_latent_handle_takes_no_nhd_format(_op):
    assert not _handle(kv=_StubMLAPages()).takes_fused(_op(PagedNHD()).op), (
        "an NHD kernel would write MLA pages as (tokens, heads, head_dim)"
    )


def test_a_latent_array_with_no_positions_takes_no_rope_table_op(_op):
    assert not _handle(kv=_StubMLAPages()).takes_fused(_op(PagedLatent(rope_table=True)).op), (
        "the kernel would rotate with a table no position resource holds"
    )


@pytest.mark.parametrize("rope, fused", [
    (_StubRope(), True), (_StubRope(interleave=False), False), (_StubRope(table=False), False),
])
def test_a_rope_table_op_needs_the_position_resource_s_interleaved_table(_op, rope, fused):
    array = _latent(positions=Positions(rope=Rope()))
    resources = {"kv": _StubMLAPages(), "attn": _StubMLA(), "rope": rope}
    h = build_handle(array, resources, _resolve([array], StateSetup()))
    assert h.takes_fused(_op(PagedLatent(rope_table=True)).op) is fused


def _fused_call(make, kernel=None):
    write = make(PagedLatent(), kernel=kernel)
    h = _handle(kv=_StubMLAPages())
    h.fused_ops = frozenset({write.op.name})
    _begin(h)
    return lambda: write(h, 0, *_rows())


def test_the_debug_check_reads_latent_slots(_op, monkeypatch):
    monkeypatch.setattr(state_op_module, "_DEBUG_ASSERTS", True)
    _fused_call(_op)()

    def write_past(kv, layer, ckv, kpe, *, target):
        target.cache[1, 0] = 1.0

    _refused("changed 24 elements outside this step's slots", _fused_call(_op, write_past))


def test_conformance_compares_a_latent_kernel_on_the_step_s_slots(_op):
    write = _op(PagedLatent())
    h = _handle(kv=_StubMLAPages())
    _begin(h)
    assert state_op_module.conformance_problems(write, (h, 0, *_rows()), 0.0) == [], (
        "a kernel that writes what its reference adds, where it adds it, conforms"
    )


def test_a_handle_with_no_pages_refuses_pool_and_slots():
    h = _NoPages(_latent(), {})
    for call in (h._pages, lambda: h._slots(0), lambda: h.fused_target(0)):
        _refused(r"no layout yet: \w+ on a _NoPages", call)
