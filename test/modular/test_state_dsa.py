"""DSA reads an MLA latent array sparsely: an indexer scores each row's index keys, keeps its top positions, and the
latent read attends only those.

The keys are an array of their own, read by `TopK`, which today's DSA resource keeps in its own store. The latent array
reads through `Selected(of=...)`, and the model passes the picks. A pick past its row would attend an entry the row
hasn't reached, so the read masks it, and a key array that doesn't pair with its latent would pick positions the latent
doesn't hold.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
from test_state_declaration_checks import _StubModel

from mstar.engine.resources.dsa.config import DSAConfig, DSAStep
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.engine.resources.kv.plan import SequenceView
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    Entry,
    KVEntry,
    LatentAttend,
    LatentEntry,
    Positions,
    Read,
    Selected,
    TensorType,
    TopK,
    Window,
    selects,
)
from mstar.engine.state.adapters import DSA_KEYS, LATENT, StateSetup, _resolve, build_handle, build_specs, storage_of
from mstar.engine.state.analysis import state_declaration_problems

LAYERS, PAGES, PAGE, HEADS, RANK, ROPE, DIM, COUNT = 2, 4, 4, 4, 16, 8, 8, 4
SLOTS = PAGES * PAGE


def _keys(**overrides) -> Array:
    return Array("keys", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)),
        "entry": Entry(tensors={"k": TensorType((DIM,), dtype=torch.bfloat16)}), "max_len": SLOTS,
        "reads": {"index": Read(mask=Causal(), op=TopK(COUNT, heads=HEADS))}, **overrides,
    })


def _latent(**overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)), "entry": LatentEntry(rank=RANK, rope_dim=ROPE),
        "max_len": SLOTS, "reads": {"attn": Read(mask=Selected("keys"), op=LatentAttend(HEADS, scale=0.125))},
        **overrides,
    })


class _StubPages:
    """MLA pages per layer, (pages, page_size, rank + rope_dim), that a test fills itself."""

    def __init__(self):
        self.pages = torch.randn(LAYERS, PAGES, PAGE, RANK + ROPE)

    def layer_view(self, i):
        return self.pages[i]

    def opens_label(self, rid, label):
        return True


class _StubMLA:
    """Counts the dense reads and returns zeros."""

    requires_kv_write = True

    def __init__(self):
        self.runs = 0

    def run(self, q, label=None, kv_cache_layer=None, q_pe=None):
        self.runs += 1
        return torch.zeros_like(q)


def _dsa(length: int = 6) -> DSAManager:
    """A DSA resource planned for one request decoding at position ``length - 1`` over pages 0 and 1."""
    dsa = DSAManager(DSAConfig(kv_cache="kv", topk=COUNT), page_size=PAGE, device=torch.device("cpu"))
    views = [SequenceView("r0", "main", [0, 1], length=length, to_compute=1)]
    dsa.plan(DSAStep(), SimpleNamespace(plan_results={"kv": {"main": SimpleNamespace(views=views)}}))
    return dsa


def _handles(dsa: DSAManager, mla=None):
    arrays = [_latent(), _keys()]
    resources = {"kv": _StubPages(), "attn": mla or _StubMLA(), "keys": dsa}
    resolved = _resolve(arrays, StateSetup())
    latent, keys = (build_handle(array, resources, resolved) for array in arrays)
    step = SubmoduleStep(segments=[Segment(1, "main", 1)], state={"kv": ArrayStep(), "keys": ArrayStep()})
    step.set_ctx(StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False))
    for h in (latent, keys):
        h.begin_step(step)
    return latent, keys


def _no_layout(pattern: str, arrays) -> None:
    with pytest.raises(ValueError, match=pattern):
        build_specs(arrays)


# The declarations


def test_topk_and_selected_refuse_their_own_bad_fields():
    for op, cause in ((TopK(0, heads=HEADS), "count must be a positive int"),
                      (TopK(COUNT, heads=0), "heads must be a positive int")):
        with pytest.raises(ValueError, match=cause):
            _keys(reads={"index": Read(mask=Causal(), op=op)})
    with pytest.raises(ValueError, match="TopK needs entry field k 1-D and unsharded"):
        _keys(entry=Entry(tensors={"k": TensorType((2, DIM), dtype=torch.bfloat16)}))
    with pytest.raises(ValueError, match="Selected.of must be a non-empty str"):
        Selected("")


@pytest.mark.parametrize("latent, keys, problem", [
    ({}, None, "'keys' names no declared array"),
    ({}, {"nodes": {"LLM", "vision"}}, "Array 'keys' has other nodes"),
    ({}, {"max_len": SLOTS // 2}, "Array 'keys' has other max_len"),
    ({"verify_span": 2}, {}, "Array 'keys' has other verify_span"),
    ({}, {"reads": {"index": Read(mask=Causal(), op=TopK(COUNT, heads=HEADS)), "reach": Read(mask=Causal())}},
     "Array 'keys' has a read other than TopK"),
])
def test_selected_names_the_array_that_picks_for_it(latent, keys, problem):
    arrays = [_latent(**latent)] + ([] if keys is None else [_keys(**keys)])
    problems = state_declaration_problems(_StubModel(*arrays))
    assert any(problem in each for each in problems), f"the picks would index a log the keys don't hold: {problems}"


def test_a_key_array_builds_the_dsa_resource_over_its_latent_pool():
    assert storage_of(_keys()) is DSA_KEYS and storage_of(_latent()) is LATENT
    (spec,) = [spec for spec in build_specs([_latent(), _keys()]) if spec.resource_key == "keys"]
    assert spec.config == DSAConfig(kv_cache="kv", topk=COUNT), "the DSA resource would plan another pool's rows"


@pytest.mark.parametrize("arrays, cause", [
    ([_keys()], r"latent arrays \[\] pick through it"),
    ([_latent(), _keys(entry=Entry(tensors={"k": TensorType((DIM,), dtype=torch.bfloat16),
                                            "x": TensorType((DIM,), dtype=torch.bfloat16)}))],
     "an entry other than k alone"),
    ([_latent(), _keys(positions=Positions())], "positions, which today's resources don't keep on DsaKeyStorage"),
    ([_latent(), _keys(verify_span=2)], "verify_span 2, which today's resources settle on no DsaKeyStorage"),
    ([_latent(), _keys(reads={"index": Read(mask=Window(8), op=TopK(COUNT, heads=HEADS))})],
     r"reads\['index'\].mask must be Causal"),
    ([Array("q", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=SLOTS,
            reads={"self": Read(mask=Selected("keys"), op=Attend(q_heads=HEADS))}), _keys()],
     r"reads\['self'\].mask must be"),
])
def test_a_key_array_is_one_topk_read_of_k_for_one_latent(arrays, cause):
    _no_layout(cause, arrays)


# The calls


def test_the_key_store_stores_and_picks_as_main_does():
    torch.manual_seed(0)
    history, k = torch.randn(5, DIM), torch.randn(1, DIM)
    q, weights = torch.randn(1, HEADS, DIM), torch.randn(1, HEADS)
    ours, main = _dsa(), _dsa()
    for dsa in (ours, main):
        dsa.keys.append("r0", 1, history, start_pos=0)
    _, keys = _handles(ours)
    keys.add(1, k=k)
    picked = keys.read(1, q, weights=weights)
    main.store_keys(1, k)
    assert torch.equal(picked, main.select(1, q, weights)), "the key array would pick what main's resource doesn't"


@pytest.mark.parametrize("length", [COUNT, COUNT + 1])
def test_selects_is_the_dsa_plan_s_answer(length):
    dsa = _dsa(length)
    _, keys = _handles(dsa)
    assert selects(keys) == dsa.needs_selection() == (length > COUNT)


def test_a_selected_read_with_none_runs_the_dense_plan():
    mla = _StubMLA()
    latent, _ = _handles(_dsa(), mla)
    latent.read(0, torch.randn(1, HEADS, RANK), q_pe=torch.randn(1, HEADS, ROPE), selected=None)
    assert mla.runs == 1, "a step that picks everything its rows reach attends through the dense plan"


def test_a_selected_read_masks_a_pick_past_its_row():
    torch.manual_seed(1)
    latent, _ = _handles(_dsa(length=6))
    q, q_pe = torch.randn(1, HEADS, RANK), torch.randn(1, HEADS, ROPE)
    past = latent.read(0, q, q_pe=q_pe, selected=torch.tensor([[0, 5, 7, -1]], dtype=torch.int32))
    reached = latent.read(0, q, q_pe=q_pe, selected=torch.tensor([[0, 5, -1, -1]], dtype=torch.int32))
    assert torch.equal(past, reached), "the row at position 5 would attend position 7, which it hasn't reached"


def test_the_dsa_resource_refuses_capture():
    assert not _dsa().can_capture(), "a capture would record one step's host loop over its requests"
