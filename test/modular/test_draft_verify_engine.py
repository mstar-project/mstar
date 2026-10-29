"""Draft-verify decoding through the engine's own step path, on CPU: a tiny model whose verify
steps draft k tokens with a chained head, run by ``Engine.exec_and_postprocess`` and
``Engine.check_stop_for_batch`` as the worker runs them.

Greedy verify must emit exactly what plain greedy decode emits, whatever the drafter gets
right; the KV and positions it rewinds must stay inside the layout's footprint; and a node
that can't verify, or a request a greedy verify would answer differently, is refused before
it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources import SamplingReqConfig  # noqa: E402
from mstar.engine.resources.base import Resource  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402
from mstar.engine.resources.spec import NodeResourceSpec  # noqa: E402

CPU = torch.device("cpu")
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25], [60, 2, 33]]
MAX_TOKENS = 16


@pytest.fixture(autouse=True)
def _threads():
    # the comparisons are exact; keep the CPU reductions' order fixed
    torch.set_num_threads(2)


def _run(k: int, corrupt: int = 0, batched: bool = True, max_tokens: int = MAX_TOKENS,
         prompts=PROMPTS, ignore_eos: bool = False, extra_step: bool = False, seed: int = toy.SEED):
    sub = toy.ToySubmodule(k=k, corrupt=corrupt, seed=seed)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, max_tokens, ignore_eos=ignore_eos)
    out = drv.run(list(range(len(prompts))), batched=batched)
    if extra_step:
        # the async worker's step after a stop: launched before the stop was seen
        before = {rid: list(tokens) for rid, tokens in out.items()}
        drv.decode(list(out))
        assert all(drv.emitted[rid] == before[rid] for rid in out), "a step after the stop emitted"
    return out, eng, drv


def _reference(prompts=PROMPTS, max_tokens=MAX_TOKENS, ignore_eos=False, seed=toy.SEED):
    return {
        rid: toy.reference(prompt, max_tokens, ignore_eos=ignore_eos, seed=seed)
        for rid, prompt in enumerate(prompts)
    }


def test_the_toy_streams_stop_mid_block_and_at_the_cap():
    """Precondition for the parity tests: some streams end on the stop token, some at
    max_tokens, and none is degenerate."""
    ref = _reference()
    assert any(tokens[-1] == toy.STOP for tokens in ref.values())
    assert any(len(tokens) == MAX_TOKENS for tokens in ref.values())
    assert all(len(set(tokens)) > 2 for tokens in ref.values())


def test_plain_decode_matches_the_reference():
    out, eng, _ = _run(k=0)
    assert out == _reference()
    # a node without drafts carries no draft-verify state anywhere
    assert eng._runner._rewinders == {}
    assert set(eng._submodules[toy.NODE].verify_walks.values()) == {None}


@pytest.mark.parametrize("batched", [True, False], ids=["batched", "one_per_step"])
@pytest.mark.parametrize(("k", "corrupt"), [(1, 0), (3, 0), (3, 2), (5, 3), (4, 1)])
def test_greedy_verify_emits_what_plain_decode_emits(k, corrupt, batched):
    out, eng, _ = _run(k, corrupt, batched=batched)
    assert out == _reference()
    dv = eng._resources[toy.DV]
    mean = dv.drafts_accepted / dv.rows_verified
    if corrupt == 1:
        assert mean == 0, "every draft is wrong, so every verify step rejects all of them"
    else:
        assert 0 < mean, dv.stats()
    if corrupt:
        assert mean < k, "some drafts must be rejected for the rewinds to be exercised"


def test_ignore_eos_runs_past_the_stop_token_to_the_cap():
    out, _, _ = _run(3, corrupt=2, ignore_eos=True)
    assert out == _reference(ignore_eos=True)
    assert all(len(tokens) == MAX_TOKENS for tokens in out.values())


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_max_tokens_cuts_inside_a_block(max_tokens):
    # k = 3 with a perfect-ish drafter: a block of 4 overshoots small caps
    out, _, _ = _run(3, max_tokens=max_tokens)
    assert out == _reference(max_tokens=max_tokens)
    assert all(len(tokens) <= max_tokens for tokens in out.values())


def test_a_step_that_emits_nothing_sends_tensors_the_transport_takes():
    """max_tokens 1: the prefill's token is the whole reply, so the first verify step emits
    no token for any row. The empty tensors must still serialize for the wire (a numpy mask
    that selects nothing has a stride of 0, which a byte view refuses)."""
    from mstar.communication.tensors import _serialize_tensor

    sub = toy.ToySubmodule(k=3)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, 1)
    out = drv._step("decode", {rid: drv.next[rid] for rid in range(len(PROMPTS))})
    for rid in range(len(PROMPTS)):
        (tokens,) = out[rid]["new_token"]
        assert tokens.numel() == 0 and _serialize_tensor(tokens) == b""
    assert set(drv.stops) == set(range(len(PROMPTS)))


def test_an_engine_loaded_without_its_model_binds_the_walk_before_its_first_forward():
    """Without the model at load, a walk's policy is asked at the walk's first step. That
    has to be before the step's forward: its verify runs the sampler the walk binds."""
    sub = toy.ToySubmodule(k=3, corrupt=2)
    eng = toy.engine(sub, CPU, bare=True)
    assert eng._submodules[toy.NODE].verify_walks == {}
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS)
    assert drv.run(list(range(len(PROMPTS)))) == _reference()


@pytest.mark.parametrize("rewound", ["kv", "positions"])
def test_without_a_rewind_the_stream_goes_wrong(rewound, monkeypatch):
    """The parity above is not vacuous: skip either rewind and the target reads the rejected
    drafts' rows, or reads its own at the wrong positions. On seed 0, whose streams follow
    the attention closely (the default seed's settle into short cycles)."""
    assert _run(3, corrupt=2, seed=0)[0] == _reference(seed=0)
    owner = kv_manager.KVManager if rewound == "kv" else RopeManager
    monkeypatch.setattr(owner, "rewind", lambda self, verdicts: None)
    out, _, _ = _run(3, corrupt=2, seed=0)
    assert out != _reference(seed=0)


@pytest.mark.parametrize(("k", "corrupt"), [(3, 0), (3, 1), (2, 3)], ids=["accept", "reject_all", "mixed"])
def test_the_streams_stay_inside_their_footprint(k, corrupt, monkeypatch):
    """Every request, to its cap or its stop, then the async worker's one more step: its KV
    holds no more pages than its layout's footprint (2k + 1 of lookahead), its committed
    slots stay under the layout's bound (the debug assert at every commit), and its
    position moves with its length."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    seen = []
    real = kv_manager.KVManager.commit

    def commit(self, step, ctx):
        real(self, step, ctx)
        for rid in ctx.request_ids:
            stream = self._streams[rid]["main"]
            seen.append((len(stream.page_indices), stream.footprint))

    monkeypatch.setattr(kv_manager.KVManager, "commit", commit)
    out, eng, drv = _run(k, corrupt, ignore_eos=True, extra_step=True)
    assert out == _reference(ignore_eos=True)
    assert seen and all(held <= footprint for held, footprint in seen), max(seen)
    kv, pos = eng._resources[toy.KV], eng._resources[toy.POS]
    for rid in out:
        assert kv._streams[rid]["main"].stored_len == pos.position(rid, "main")


def test_the_last_step_of_a_full_block_reaches_the_lookahead(monkeypatch):
    """The footprint is tight: a stream that accepts every draft, stopped by max_tokens a
    token into a block, then runs the async worker's extra step, writes 2k + 1 slots past the
    kept tokens. One slot less of lookahead and the debug bound fires."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    k = 3
    prompt = [3, 9, 12, 40]
    ref = toy.reference(prompt, 64, ignore_eos=True)
    sub = toy.ToySubmodule(k=k)
    sub.oracle = torch.tensor(prompt + ref)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    # a cap the accepted blocks pass by k: the stream keeps k drafts past it
    drv.add(0, prompt, 6, ignore_eos=True)
    drv.run([0])
    drv.decode([0])
    stream = eng._resources[toy.KV]._streams[0]["main"]
    assert drv.emitted[0] == ref[:6]
    assert eng._resources[toy.DV].drafts_accepted == 3 * k, "the oracle drafter missed"
    # the layout's bound is prompt + (cap - 1) + lookahead: reached, not passed
    assert stream.stored_len == len(prompt) + 6 - 1 + 2 * k + 1


def test_a_request_a_greedy_verify_cannot_serve_is_refused_at_ingest():
    sub = toy.ToySubmodule(k=3)
    eng = toy.engine(sub, CPU)
    for bad in ({"temperature": 0.7}, {"repetition_penalty": 1.2}, {"min_tokens": 2}):
        with pytest.raises(ValueError, match="greedy-only"):
            eng.add_request(9, {toy.SAMPLER: SamplingReqConfig(**{"temperature": 0.0, **bad})})
        # the conductor refuses it first, as a 400
        spec = next(s for s in toy.specs() if s.resource_key == toy.SAMPLER)
        with pytest.raises(ValueError, match="greedy-only"):
            SamplingReqConfig(**{"temperature": 0.0, **bad}).validate(spec)
    # nothing was left behind by the refused ingests
    assert 9 not in eng._resources[toy.KV]._streams
    assert 9 not in eng._resources[toy.DV]._emitted


def test_a_layout_without_the_lookahead_is_refused():
    sub = toy.ToySubmodule(k=3)
    eng = toy.engine(sub, CPU)
    with pytest.raises(ValueError, match="lookahead of 3, not 7"):
        eng.add_request(1, {
            toy.KV: toy.layout(4, 8, lookahead=3),
            toy.SAMPLER: SamplingReqConfig(temperature=0.0),
        })


class _Opaque(Resource):
    """Per-token state the draft-verify rewind knows nothing of."""

    @classmethod
    def build(cls, spec, info):
        return cls()


class _OpaqueSpec(NodeResourceSpec):
    @property
    def resource_class(self):
        return _Opaque


def _load(specs, model=None, k=3):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    eng = Engine(None, autocast_dtype=None)
    eng.load_model(
        submodules={toy.NODE: toy.ToySubmodule(k=k)}, specs=specs, parallel_groups=toy._Group(),
        device=CPU, transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        kv_cache_type=torch.float32, model=model or toy.ToyModel(),
    )
    return eng


def test_a_node_that_cannot_verify_fails_to_load():
    specs = toy.specs()
    without_dv = [s for s in specs if s.resource_key != toy.DV]
    with pytest.raises(ValueError, match="needs one DraftVerifySpec"):
        _load(without_dv)
    plain_sampler = [s for s in toy.specs(draft_verify=False)] + [s for s in specs if s.resource_key == toy.DV]
    with pytest.raises(ValueError, match="draft_verify=True"):
        _load(plain_sampler)
    with pytest.raises(ValueError, match=r"\['opaque'\] declare no rewind_kind"):
        _load([*specs, _OpaqueSpec("opaque", {toy.NODE})])

    class Combined(toy.ToyModel):
        def get_combined_graph_walks(self):
            return {toy.NODE: {"mixed": {"prefill", "decode"}}}

    with pytest.raises(ValueError, match="combined walk"):
        _load(specs, Combined())
    # the same specs with no drafts load: the checks are the verifying walk's
    _load(without_dv, k=0)
