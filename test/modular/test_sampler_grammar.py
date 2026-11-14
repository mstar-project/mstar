"""Grammar-constrained sampling (``structured_outputs``): a constrained row's
draws follow its grammar on the eager and graph sampler, a free row beside it
is untouched, ``after`` starts the grammar at a text, and the request field is
checked where the conductor builds the sampler config."""

import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

xgrammar = pytest.importorskip("xgrammar")

from test_byte_level_text_chunks import fast_tokenizer  # noqa: E402, F401

from mstar.api_server.openai.adapters import _refuse_unserved_chat  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.engine.accelerator_graph_backend import SplitGraph  # noqa: E402
from mstar.engine.resources import StepRunner  # noqa: E402
from mstar.engine.resources.sampler import grammar, utils  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.sampler.grammar import MAX_GRAMMAR_CHARS, GrammarRequest, grammar_request  # noqa: E402
from mstar.engine.resources.sampler.resource import GrammarSamplerResource  # noqa: E402
from mstar.engine.resources.step import ASKS, SlotLease, StepContext  # noqa: E402
from mstar.engine.state.contract import Sampler, SampleSettings  # noqa: E402
from mstar.model.submodule_base import TokenStop  # noqa: E402
from mstar.model.utils import TextChunkDetokenizer  # noqa: E402
from mstar.testing_utils.resource import StubResource  # noqa: E402

VOCAB = [b"{", b"}", b'"', b":", b",", b"a", b"b", b"1", b"2", b" ", b"true", b"x", b"<think>", b"</think>", b""]
EOS = len(VOCAB) - 1
V = 20  # wider logits than the vocabulary, as a padded lm_head is
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
          "additionalProperties": False}


@pytest.fixture(autouse=True)
def _argmax_draw(monkeypatch):
    monkeypatch.setattr(utils, "sample_tokens", lambda logits, **kw: logits.argmax(-1))
    monkeypatch.setattr(utils, "sample_accelerator_graphable_gpu", lambda logits, *a, **kw: logits.argmax(-1))


def _resource(graph: bool, comm_group=None, tokenizer_eos: bool = False) -> GrammarSamplerResource:
    """``tokenizer_eos``: the model declares no stop and its HF tokenizer's EOS is the only one."""
    detok = SimpleNamespace(token_bytes=lambda i: VOCAB[i] if i < len(VOCAB) else b"")
    if tokenizer_eos:
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        words = {(v.decode() or "<eos>"): i for i, v in enumerate(VOCAB)}
        detok.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(words, unk_token="{")), eos_token="<eos>")
    text_output = SimpleNamespace(detokenizer=detok, stop_token_ids=frozenset() if tokenizer_eos else frozenset({EOS}))
    res = GrammarSamplerResource(
        vocab_size=V, enable_repetition_penalty=False, device=torch.device("cpu"),
        text_output=text_output, grammar_cache_mb=16, comm_group=comm_group,
    )
    if graph:
        res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=4, max_seq_len=1)
    return res


def _decode(res, rids, graph, steps=40, seed=0):
    """Draw greedily from random logits until each row draws EOS; each row's tokens."""
    gen = torch.Generator().manual_seed(seed)
    out = {rid: [] for rid in rids}
    live = list(rids)
    for _ in range(steps):
        if not live:
            break
        lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4)) if graph else None
        ctx = StepContext(request_ids=tuple(live), graph_walk="w", slot=0, capture=False, slot_lease=lease)
        step = SamplerStep(apply_penalty=False)
        res.plan(step, ctx)
        logits = torch.randn(4 if graph else len(live), V, generator=gen)
        tokens = res.sample(list(live), logits)
        res.commit(step, ctx)
        for rid, tok in zip(list(live), tokens.tolist(), strict=False):
            out[rid].append(tok)
            if tok == EOS:
                live.remove(rid)
    return out


def _text(tokens):
    return b"".join(VOCAB[t] for t in tokens if t != EOS).decode()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_a_constrained_row_draws_its_grammar_beside_a_free_row(graph):
    res = _resource(graph)
    res.ingest_request("json", SamplingReqConfig(grammar=grammar_request({"json": SCHEMA})))
    res.ingest_request("choice", SamplingReqConfig(grammar=grammar_request({"choice": ["ab", "ba"]})))
    res.ingest_request("free", SamplingReqConfig())

    out = _decode(res, ["json", "choice", "free"], graph)

    assert out["json"][-1] == EOS and isinstance(json.loads(_text(out["json"]))["a"], int)
    assert out["choice"][-1] == EOS and _text(out["choice"]) in ("ab", "ba")
    assert any(t >= len(VOCAB) for t in out["free"]), "the free row was masked too"


def test_a_request_given_a_finished_one_s_handle_starts_its_grammar_afresh():
    """Worker handles are recycled: the last draw of a finished request is not
    the next request's."""
    res = _resource(graph=False)
    for seed in (0, 1):
        res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab", "ba"]})))
        out = _decode(res, [0], graph=False, seed=seed)[0]
        assert out[-1] == EOS and _text(out) in ("ab", "ba")
        res.remove_request(0)


def test_a_replay_commits_the_draw_of_the_graph_it_replays():
    """Two graphs captured on one slot at one batch size (a decode and a mixed
    bucket, say) each draw into their own tensor; a replay of the first must
    advance the grammar by the first's tokens, not the second's."""
    res = _resource(graph=True)
    lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4))
    step = SamplerStep(apply_penalty=False)
    pads = ("p0", "p1", "p2", "p3")
    for rid in pads:
        res.ingest_capture_row(rid)
    a, b, x = VOCAB.index(b"a"), VOCAB.index(b"b"), VOCAB.index(b"x")
    drawn = []
    for favoured in (a, x):  # capture the first bucket, then the second
        ctx = StepContext(request_ids=pads, graph_walk="w", slot=0, capture=True, slot_lease=lease)
        res.plan(step, ctx)
        logits = torch.zeros(4, V)
        logits[:, favoured] = 1.0
        drawn.append(res.sample(list(pads), logits))
        res.commit(step, ctx)

    res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab", "ba"]})))
    ctx = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    res.plan(step, ctx)
    # replay the first graph: it rewrites its own draw and the slot's last-token row
    replayed = torch.full((4,), a)
    drawn[0].copy_(replayed)
    res._cg_sampler.last_token_buf.copy_(replayed)
    res.commit(step, ctx)

    res.plan(step, ctx)
    logits = torch.zeros(4, V)
    logits[:, a] = 1.0
    assert res.sample([0], logits)[0].item() == b, "the grammar advanced by another graph's draw"


def test_a_smaller_bucket_on_a_slot_leaves_no_mask_for_a_larger_one():
    """The buckets of one slot share its mask buffer: a constrained row's mask in a
    row a smaller bucket doesn't cover is gone before a larger bucket applies it."""
    res = _resource(graph=True)
    res.ingest_request("c", SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
    step = SamplerStep(apply_penalty=False)
    for rids in (("f0", "f1", "f2", "c"), ("f0", "f1"), ("f0", "f1", "f2", "f3")):
        lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=len(rids)))
        res.plan(step, StepContext(request_ids=rids, graph_walk="w", slot=0, capture=False, slot_lease=lease))
        res._fill_grammar()
    assert (res._grammar_masks[0] == -1).all(), "a free row was masked by another request's grammar"


class _Pool(StubResource):
    """A KV pool with room for every request, which each new request asks."""

    def __init__(self):
        self.held = set()

    def ask(self, rid, node_name, graph_walk):
        return None if rid in self.held else ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        self.held.add(rid)

    def unreserve(self, rid):
        self.held.discard(rid)


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_request_whose_grammar_compiles_holds_back_no_other(ranks):
    """A compile is readiness, not capacity: the requests behind it start
    meanwhile, and it waits out of every step on any number of ranks (the one
    step that waited for it would stall every row on the worker)."""
    res = _resource(graph=False, comm_group=SimpleNamespace(world_size=ranks, rank=0))
    compiled = Future()
    res._grammar._compile_pool = SimpleNamespace(submit=lambda fn, *args: compiled)
    request = grammar_request({"choice": ["ab"]})
    res.ingest_request("a", SamplingReqConfig(grammar=request))
    res.ingest_request("b", SamplingReqConfig())
    runner = StepRunner({"kv": _Pool(), "sampler": res}, node_resources={"LLM": ["kv", "sampler"]})

    assert not runner.admit_retrieve("a", "LLM", "w").ready
    assert runner.admit_retrieve("b", "LLM", "w").ready, "a request waited behind another's compile"
    compiled.set_result(request.compile(res._grammar._compiler))
    assert runner.admit_retrieve("a", "LLM", "w").ready


@pytest.mark.parametrize("failure", ["compile", "start", "refused draw", "two rows"])
def test_a_grammar_that_can_no_longer_hold_fails_its_request(monkeypatch, failure):
    """A grammar that fails to compile is the request's error at its next admit;
    a matcher that fails to start, the same; a draw it refuses, at the admit after the step that accepts it, and that
    step's draw may be anything but a stop, so the reply can't end as a 200 before the error lands; two rows of one
    step (a draft's), at that step's admit.
    A pre-admit gives the same answer as a full admit, as a pre-admitted step
    skips its full admit on one TP rank and not another."""
    res = _resource(graph=True)
    lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4))
    step = SamplerStep(apply_penalty=False)
    if failure == "compile":
        # past the tokenizer-free check, as a grammar xgrammar compiles differently would be
        res.ingest_request(0, SamplingReqConfig(grammar=GrammarRequest("grammar", "root ::= undefined_rule")))
    elif failure == "start":
        res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
        monkeypatch.setattr(xgrammar, "GrammarMatcher", lambda compiled: 1 / 0)
    elif failure == "two rows":
        res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
    else:
        res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
        ctx = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=lease)
        res.plan(step, ctx)
        # a replay whose draw the mask did not cover
        res._cg_sampler.last_token_buf.fill_(VOCAB.index(b"x"))
        res.commit(step, ctx)
        pre = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=lease)
        pre.is_preplan = True
        assert res.admit(step, pre).ok and res.admit(step, ctx).ok, "ranks would disagree on this step"
        res.plan(step, ctx)
        logits = torch.zeros(4, V)
        logits[:, EOS] = 1.0  # the model's own pick is its stop
        assert res.sample([0], logits)[0].item() != EOS, "the reply ended before its error"
    rows = (0, 1, 0) if failure == "two rows" else (0, 1)
    ctx = StepContext(request_ids=rows, graph_walk="w", slot=0, capture=False, slot_lease=lease)
    outcome = res.admit(step, ctx)
    assert not outcome.ok and outcome.reason.terminal and outcome.reason.request_ids == {0}


@pytest.mark.parametrize("tokenizer_eos", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
def test_a_fill_or_accept_that_raises_fails_what_every_rank_agrees_on(monkeypatch, ranks, tokenizer_eos):
    """A replay fills the masks between its forward and its draw: a fill that raises
    still draws. On a lone rank the row may draw anything but a stop for that draw, so its request
    reaches the next admit and fails there with the error (a stop would end the reply
    first, as a 200); under several ranks the row may only stop, since a fill runs on
    rank 0 alone and an error set there would split the admits. One row's own fill
    raising fails only that row. An accept runs on every rank, so one that raises
    fails its row's request on each, and the other rows' draws are still accepted. The
    stops are the ones xgrammar resolves, so a model that declares none but whose tokenizer
    has an EOS is masked alike."""
    res = _resource(graph=True, comm_group=SimpleNamespace(world_size=ranks, rank=0), tokenizer_eos=tokenizer_eos)
    res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
    lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4))
    step = SamplerStep(apply_penalty=False)
    ctx = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    res.plan(step, ctx)
    replayed = []
    parts = [SimpleNamespace(replay=lambda part=part: replayed.append(part)) for part in ("forward", "draw")]
    graph = SplitGraph(parts, res._fill_grammar)
    with monkeypatch.context() as patch:
        patch.setattr(grammar._Row, "matcher", property(lambda row: 1 / 0))
        graph.replay()
    assert replayed == ["forward", "draw"]
    logits = torch.zeros(4, V)
    logits[:, EOS] = 1.0  # the model's own pick is its stop
    res._grammar.apply(logits, res._grammar_mask)
    forced_stop = logits[0].argmax().item() == EOS
    assert forced_stop == (ranks > 1), "a lone rank's failed fill must not end the reply before its error"
    assert res.admit(step, ctx).ok == (ranks > 1)

    res = _resource(graph=True, comm_group=SimpleNamespace(world_size=ranks, rank=0), tokenizer_eos=tokenizer_eos)
    for rid in (1, 2):
        res.ingest_request(rid, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
    ctx2 = StepContext(request_ids=(1, 2), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    res.plan(step, ctx2)
    real = res._grammar._rows[1].matcher
    res._grammar._rows[1]._matcher = SimpleNamespace(
        is_terminated=real.is_terminated, fill_next_token_bitmask=lambda mask, i: 1 / 0)
    res._fill_grammar()
    res._grammar.apply(logits := torch.zeros(4, V), res._grammar_mask)
    assert (logits[0].argmax().item() == EOS) == (ranks > 1), "the failed row's mask"
    assert logits[1, VOCAB.index(b"a")] == 0 and logits[1, VOCAB.index(b"b")] == float("-inf"), \
        "the healthy row lost its mask"
    assert [r for r in (1, 2) if res._grammar.failures([r])] == ([] if ranks > 1 else [1])

    if tokenizer_eos:
        return  # the accept half below does not depend on where the stops come from
    outcomes, accept = [], grammar._Row.accept
    for rank in range(ranks):
        res = _resource(graph=True, comm_group=SimpleNamespace(world_size=ranks, rank=rank))
        for rid in (1, 2):
            res.ingest_request(rid, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
        ctx = StepContext(request_ids=(1, 2), graph_walk="w", slot=0, capture=False, slot_lease=lease)
        res.plan(step, ctx)
        res._fill_grammar()
        res._cg_sampler.last_token_buf.fill_(VOCAB.index(b"a"))
        res.commit(step, ctx)
        res.plan(step, ctx)
        with monkeypatch.context() as patch:
            patch.setattr(grammar._Row, "accept", lambda row, token, broken=res._grammar._rows[1], accept=accept:
                          1 / 0 if row is broken else accept(row, token))
            res._fill_grammar()
        assert not res._grammar.failures([2])
        if rank == 0:
            res._grammar.apply(logits := torch.zeros(4, V), res._grammar_mask)
            assert logits[1, VOCAB.index(b"b")] == 0 and logits[1, VOCAB.index(b"a")] == float("-inf"), \
                "a row after the one whose accept raised lost its draw"
        outcome = res.admit(step, ctx)
        outcomes.append((outcome.ok, outcome.reason and outcome.reason.request_ids))
    assert outcomes == [(False, {1})] * ranks


@pytest.mark.parametrize("batched", [False, True])
def test_only_rank_0_fills_masks_and_every_rank_fails_a_refused_draw_alike(monkeypatch, batched):
    """Rank 0's draw is every rank's, so only it fills masks; every rank accepts
    the draws, so a refused one fails its request at the same admit on each. The
    fill gives the same masks serially and through xgrammar's batch matcher."""
    if batched:
        monkeypatch.setattr(grammar, "FILL_ROWS_PER_THREAD", 1)
    lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4))
    step = SamplerStep(apply_penalty=False)
    ctx = StepContext(request_ids=(9, 0, 1), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    filled, outcomes = [], []
    for rank in (0, 1):
        res = _resource(graph=True, comm_group=SimpleNamespace(world_size=2, rank=rank))
        res._grammar._fill_threads = 2 if batched else 1
        res.ingest_request(9, SamplingReqConfig())  # a free row first, so the rows' indices are not their ranks
        for rid in (0, 1):
            res.ingest_request(rid, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
        res.plan(step, ctx)
        res._fill_grammar()
        filled.append([bool((res._grammar_mask[i] != -1).any()) for i in (1, 2)])
        assert (res._grammar_mask[0] == -1).all()
        if rank == 0:
            assert bool(res._grammar._batches) == batched
            res._grammar.apply(logits := torch.zeros(4, V), res._grammar_mask)
            assert (logits[1:3] == 0).sum().item() == 2 and logits[1:3, VOCAB.index(b"a")].eq(0).all()
            assert (logits[0] == 0).all(), "the free row was masked"
        # a draw the mask did not cover
        res._cg_sampler.last_token_buf.fill_(VOCAB.index(b"x"))
        res.commit(step, ctx)
        res.plan(step, ctx)
        res._fill_grammar()
        outcome = res.admit(step, ctx)
        outcomes.append((outcome.ok, outcome.reason and outcome.reason.request_ids))
    assert filled == [[True, True], [False, False]]
    assert outcomes == [(False, {0, 1}), (False, {0, 1})]

    # one bad row fails only itself, however the rows are filled
    res = _resource(graph=True, comm_group=SimpleNamespace(world_size=1, rank=0))
    res._grammar._fill_threads = 2 if batched else 1
    for rid in (0, 1):
        res.ingest_request(rid, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"]})))
    ctx = StepContext(request_ids=(0, 1), graph_walk="w", slot=0, capture=False, slot_lease=lease)
    res.plan(step, ctx)
    res._grammar._rows[1]._matcher = SimpleNamespace(
        is_terminated=lambda: False, fill_next_token_bitmask=lambda mask, i: 1 / 0)
    res._fill_grammar()
    assert not res._grammar.failures([0]) and res._grammar.failures([1])
    res._grammar.apply(logits := torch.zeros(4, V), res._grammar_mask)
    assert logits[0, VOCAB.index(b"b")] == float("-inf"), "the healthy row lost its mask"


def test_after_starts_the_grammar_once_the_reply_holds_its_text():
    """``after`` may span tokens: the reply is free text until it holds it."""
    res = _resource(graph=False)
    res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"], "after": "</think>x"})))
    x, think = VOCAB.index(b"x"), VOCAB.index(b"</think>")
    favoured = [x, think, x, x, x, x]  # free text, the two tokens of the trigger, then the grammar's
    drawn = []
    for token in favoured:
        ctx = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=None)
        step = SamplerStep(apply_penalty=False)
        res.plan(step, ctx)
        logits = torch.zeros(1, V)
        logits[0, token] = 1.0
        drawn.append(res.sample([0], logits).item())
        res.commit(step, ctx)
    assert drawn[:3] == [x, think, x] and _text(drawn[3:5]) == "ab" and drawn[5] == EOS


def test_after_reasoning_ignores_trailing_special_tokens(fast_tokenizer):  # noqa: F811
    """MiniCPM-o's audio prompt closes the think block, then adds ``<|tts_bos|>``
    (kept in its detokenizer's bytes): the reasoning is closed, as for a text prompt.
    An end that itself contains a special token still reads as closed."""
    tok = fast_tokenizer
    detok = TextChunkDetokenizer(tok)
    ask = {"json_object": True}
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>", "<|im_end|>")}
    nl, nl2, hi = (tok.encode(t) for t in ("\n", "\n\n", "hi"))
    think, close, special = ids["<think>"], ids["</think>"], ids["<|im_end|>"]

    def after(end, *seq):
        out = SimpleNamespace(detokenizer=detok, reasoning_end=end)
        return grammar.after_reasoning(ask, out, torch.tensor([*hi, *seq])).get("after")

    assert after("</think>", think, *nl) == "</think>"
    assert after("</think>", close, *nl2, special) is None
    assert after("</think>", close, *hi, special) == "</think>"
    assert after("</think><|im_end|>", close, special) is None


@pytest.mark.parametrize("value, error", [
    ({"json": {"type": "object"}, "regex": "a"}, "exactly one of"),
    ({"regex": "(("}, "structured_outputs regex"),
    ({"choice": []}, "non-empty list"),
    ({"grammar": "root ::= nothing_defined"}, "structured_outputs grammar"),
    ({"json_object": True, "after": ""}, "after must be"),
    ({"regex": "a" * (MAX_GRAMMAR_CHARS + 1)}, "at most"),
    ({"regex": "a", "after": "x" * MAX_GRAMMAR_CHARS}, "at most"),
])
def test_a_bad_grammar_is_refused(value, error):
    with pytest.raises(ValueError, match=error):
        grammar_request(value)


@pytest.mark.parametrize("response_format, want", [
    ({"type": "json_object"}, {"json_object": True}),
    ({"type": "json_schema", "json_schema": {"name": "n", "schema": SCHEMA}}, {"json": SCHEMA}),
    ({"type": "text"}, None),
])
def test_a_chat_response_format_becomes_structured_outputs(response_format, want):
    mk = {"response_format": response_format}
    _refuse_unserved_chat(SimpleNamespace(n=None, stop=None), mk)
    assert mk.get("structured_outputs") == want and "response_format" not in mk


def _conductor(enable_grammar: bool, attaches: bool = True):
    """The conductor's intake: there a ValueError is the client's 400. ``attaches``: the model
    puts the request's grammar on its sampler settings, as a model declaring STRUCTURED_OUTPUTS does."""
    spec = SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=V, enable_grammar=enable_grammar)

    def configs(i, o, kw):
        grammar = grammar_request(kw.get("structured_outputs")) if attaches else None
        return {"sampler": SampleSettings(temperature=0.3, grammar=grammar)}

    conductor = Conductor.__new__(Conductor)
    conductor.model = SimpleNamespace(get_request_resource_configs=configs)
    conductor._kv_labels = {}
    conductor._node_resources_cache = [spec]
    conductor._declarations = lambda: [Sampler("sampler", nodes={"LLM"}, vocab_size=V)]
    return conductor


def test_the_conductor_serves_structured_outputs_only_where_a_sampler_enables_it():
    kwargs = {"structured_outputs": {"regex": "a+"}}
    with pytest.raises(ValueError, match="does not serve structured_outputs"):
        _conductor(False)._get_resource_configs(["text"], ["text"], kwargs)
    with pytest.raises(ValueError, match="does not serve structured_outputs"):
        _conductor(True, attaches=False)._get_resource_configs(["text"], ["text"], kwargs)
    config = _conductor(True)._get_resource_configs(["text"], ["text"], kwargs)["sampler"]
    assert config.temperature == 0.3 and config.grammar.kind == "regex", "the model's own settings were dropped"
    plain = _conductor(False)._get_resource_configs(["text"], ["text"], {})["sampler"]
    assert plain.grammar is None and plain.temperature == 0.3


@pytest.mark.parametrize("knob", [{"min_tokens": 4}, {"ignore_eos": True}], ids=str)
def test_a_grammar_with_a_knob_that_bars_or_ignores_its_end_is_refused(knob):
    """A finished grammar allows only a stop token: ``min_tokens`` would bar
    every token, and under ``ignore_eos`` the reply would go on unconstrained."""
    with pytest.raises(ValueError, match="structured_outputs"):
        SamplingReqConfig(grammar=grammar_request({"regex": "a+"}), **knob).validate()


def test_enable_grammar_on_a_sampler_narrower_than_the_text_vocabulary_is_refused():
    """A non-text sampler (an audio codec's, say) draws ids no grammar describes."""
    text_output = SimpleNamespace(detokenizer=SimpleNamespace(tokenizer=[b""] * (V + 1)), stop_token_ids=())
    with pytest.raises(ValueError, match="belongs on the text sampler"):
        GrammarSamplerResource(
            vocab_size=V, enable_repetition_penalty=False, device=torch.device("cpu"),
            text_output=text_output, grammar_cache_mb=16,
        )


def test_a_loop_stop_the_model_does_not_declare_is_refused_for_a_grammar_deployment():
    from mstar.worker.engine_manager import _refuse_unlisted_grammar_stops

    def deployment(text_stops, enable_grammar=True):
        stop = TokenStop(loop="loop", buffer="token", stop_ids=frozenset({EOS, EOS + 1}), sampler="s")
        model = SimpleNamespace(
            text_output=lambda: SimpleNamespace(stop_token_ids=text_stops, detokenizer=SimpleNamespace(tokenizer=None))
        )
        spec = SamplerSpec(resource_key="s", nodes={"Talker"}, vocab_size=V, enable_grammar=enable_grammar)
        other = TokenStop(loop="tts_loop", buffer="code", stop_ids=frozenset({999}), sampler="tts")
        submodules = {"Talker": SimpleNamespace(token_stop=stop), "Tts": SimpleNamespace(token_stop=other)}
        return submodules, [spec], model

    _refuse_unlisted_grammar_stops(*deployment(frozenset({EOS, EOS + 1})))
    _refuse_unlisted_grammar_stops(*deployment(frozenset(), enable_grammar=False))
    with pytest.raises(ValueError, match=rf"\[{EOS + 1}\].*does not list"):
        _refuse_unlisted_grammar_stops(*deployment(frozenset({EOS})))

    # grammar on a sampler narrower than the text vocabulary, or on a model without text_output(): the
    # sampler's own refusal speaks
    submodules, specs, model = deployment(frozenset({EOS}))
    model.text_output = lambda: SimpleNamespace(
        stop_token_ids=frozenset(), detokenizer=SimpleNamespace(tokenizer=[b""] * (V + 1))
    )
    _refuse_unlisted_grammar_stops(submodules, specs, model)
    model.text_output = lambda: None
    _refuse_unlisted_grammar_stops(submodules, specs, model)
