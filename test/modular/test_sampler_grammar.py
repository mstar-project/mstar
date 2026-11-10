"""Grammar-constrained sampling (``structured_outputs``): a constrained row's
draws follow its grammar on the eager and graph sampler, a free row beside it
is untouched, ``after`` starts the grammar at a token, and the request field is
checked where the conductor builds the sampler config."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("xgrammar")

from mstar.api_server.openai.adapters import _refuse_unserved_chat  # noqa: E402
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
        for rid, tok in zip(list(live), tokens.tolist()):
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


def test_after_starts_the_grammar_at_its_token():
    res = _resource(graph=False)
    res.ingest_request(0, SamplingReqConfig(grammar=grammar_request({"choice": ["ab"], "after": "</think>"})))
    think = VOCAB.index(b"</think>")
    gen = torch.Generator().manual_seed(1)
    drawn = []
    for i in range(6):
        ctx = StepContext(request_ids=(0,), graph_walk="w", slot=0, capture=False, slot_lease=None)
        step = SamplerStep(apply_penalty=False)
        res.plan(step, ctx)
        logits = torch.randn(1, V, generator=gen)
        logits[0, think] = 100.0 if i == 2 else -100.0  # the model ends its reasoning at the third draw
        logits[0, len(VOCAB):] += 50.0  # and would draw past the vocabulary if it could
        drawn.append(res.sample([0], logits).item())
        res.commit(step, ctx)
    assert all(t >= len(VOCAB) for t in drawn[:2]), "the grammar applied before </think>"
    assert drawn[2] == think and _text(drawn[3:5]) == "ab" and drawn[5] == EOS


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


def test_the_text_sampler_takes_the_grammar_into_the_model_s_config():
    spec = SamplerSpec(resource_key="sampler", nodes={"lm"}, vocab_size=V)
    kwargs = {"structured_outputs": {"regex": "a+"}}
    assert spec.request_config(kwargs, SamplingReqConfig()) is None, "served without enable_grammar"
    spec.apply_yaml_overrides(enable_grammar=True)
    merged = spec.request_config(kwargs, SamplingReqConfig(temperature=0.2))
    assert merged.temperature == 0.2 and merged.grammar.kind == "regex"
    assert spec.request_config({}, SamplingReqConfig()) is None


@pytest.mark.parametrize("response_format, want", [
    ({"type": "json_object"}, {"json_object": True}),
    ({"type": "json_schema", "json_schema": {"name": "n", "schema": SCHEMA}}, {"json": SCHEMA}),
    ({"type": "text"}, None),
])
def test_a_chat_response_format_becomes_structured_outputs(response_format, want):
    mk = {"response_format": response_format}
    _refuse_unserved_chat(SimpleNamespace(n=None, stop=None), mk)
    assert mk.get("structured_outputs") == want and "response_format" not in mk
