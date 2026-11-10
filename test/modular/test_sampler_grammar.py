"""Grammar-constrained sampling (``structured_outputs``): a constrained row's
draws follow its grammar on the eager and graph sampler, a free row beside it
is untouched, ``after`` starts the grammar at a text, and the request field is
checked where the conductor builds the sampler config."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("xgrammar")

from mstar.api_server.openai.adapters import _constrain_after_thinking, _refuse_unserved_chat  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.engine.resources.sampler import utils  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.sampler.grammar import grammar_request  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.engine.resources.step import SlotLease, StepContext  # noqa: E402

VOCAB = [b"{", b"}", b'"', b":", b",", b"a", b"b", b"1", b"2", b" ", b"true", b"x", b"<think>", b"</think>", b""]
EOS = len(VOCAB) - 1
V = 20  # wider logits than the vocabulary, as a padded lm_head is
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
          "additionalProperties": False}


@pytest.fixture(autouse=True)
def _argmax_draw(monkeypatch):
    monkeypatch.setattr(utils, "sample_tokens", lambda logits, **kw: logits.argmax(-1))
    monkeypatch.setattr(utils, "sample_accelerator_graphable_gpu", lambda logits, *a, **kw: logits.argmax(-1))


def _resource(graph: bool) -> SamplerResource:
    text_output = SimpleNamespace(
        detokenizer=SimpleNamespace(token_bytes=lambda i: VOCAB[i] if i < len(VOCAB) else b""),
        stop_token_ids=frozenset({EOS}),
    )
    res = SamplerResource(
        vocab_size=V, enable_repetition_penalty=False, device=torch.device("cpu"),
        enable_grammar=True, text_output=text_output,
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


@pytest.mark.parametrize("value, error", [
    ({"json": {"type": "object"}, "regex": "a"}, "exactly one of"),
    ({"regex": "(("}, "structured_outputs regex"),
    ({"choice": []}, "non-empty list"),
    ({"grammar": "root ::= nothing_defined"}, "structured_outputs grammar"),
    ({"json_object": True, "after": ""}, "after must be"),
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


@pytest.mark.parametrize("mk, after", [
    ({"structured_outputs": {"json_object": True}}, "</think>"),
    ({"structured_outputs": {"json_object": True}, "enable_thinking": False}, None),
    ({"structured_outputs": {"json_object": True, "after": "<answer>"}}, "<answer>"),
])
def test_a_qwen_thinking_turn_starts_its_grammar_after_the_think_block(mk, after):
    _constrain_after_thinking(mk)
    assert mk["structured_outputs"].get("after") == after


def _conductor(enable_grammar: bool, node_groups=()):
    """The conductor's intake: there a ValueError is the client's 400."""
    spec = SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=V, enable_grammar=enable_grammar)
    conductor = Conductor.__new__(Conductor)
    conductor.model = SimpleNamespace(
        get_request_resource_configs=lambda i, o, kw: {"sampler": SamplingReqConfig(temperature=0.3)})
    conductor.model_config = {"node_groups": list(node_groups)}
    conductor._kv_labels = {}
    conductor._node_resources_cache = [spec]
    return conductor


def test_the_conductor_serves_structured_outputs_only_where_a_whole_sampler_enables_it():
    kwargs = {"structured_outputs": {"regex": "a+"}}
    with pytest.raises(ValueError, match="enable_grammar: true"):
        _conductor(False)._get_resource_configs(["text"], ["text"], kwargs)
    config = _conductor(True)._get_resource_configs(["text"], ["text"], kwargs)["sampler"]
    assert config.temperature == 0.3 and config.grammar.kind == "regex", "the model's own config was dropped"
    pd = [{"node_names": ["LLM"], "ranks": [0], "graph_walks": ["prefill"]},
          {"node_names": ["LLM"], "ranks": [1], "graph_walks": ["decode"]}]
    with pytest.raises(ValueError, match="splits prefill and decode"):
        _conductor(True, pd)._get_resource_configs(["text"], ["text"], kwargs)
