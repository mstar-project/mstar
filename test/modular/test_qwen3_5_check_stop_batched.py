"""Qwen3.5 LLMSubmodule.check_stop_batched agrees with check_stop, and the
engine falls back to the per-request path when it can't run."""
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_5.config import SAMPLER
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import HostRows

EOS = 7


def _info(ignore_eos=False, done=0, max_tokens=100):
    return SimpleNamespace(
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": done},
        max_tokens=max_tokens,
    )


def _sub():
    return SimpleNamespace(config=SimpleNamespace(stop_token_ids=frozenset({EOS, 9})))


@pytest.mark.parametrize("shape", ["flat", "column"])
def test_batched_agrees_with_per_request(shape):
    tokens = torch.tensor([EOS, 3, EOS, 9, 4, 0, 0])  # 2 padding rows
    if shape == "column":
        tokens = tokens[:, None]
    # rows in forward order: d=EOS, a=3, c=EOS, b=9 (a stop id), e=4
    row_rids = ("d", "a", "c", "b", "e")
    infos = {
        "d": _info(),                        # EOS: stops
        "a": _info(done=99, max_tokens=100), # out of budget: stops
        "c": _info(ignore_eos=True),         # EOS ignored: keeps going
        "b": _info(),                        # other stop id: stops
        "e": _info(),                        # stopped a step ago, not checked
    }
    rows = {rid: tokens[i:i + 1] for i, rid in enumerate(row_rids)}
    check = ["a", "b", "c", "d"]
    sub = _sub()
    batched = LLMSubmodule.check_stop_batched(
        sub, check, infos, HostRows(row_rids, {"new_token": tokens}),
    )
    per_request = {}
    for rid in check:
        got = LLMSubmodule.check_stop(sub, rid, infos[rid], {"new_token": [rows[rid]]})
        if got:
            per_request[rid] = got
    assert batched == per_request
    assert batched == {"d": {"decode_loop"}, "a": {"decode_loop"}, "b": {"decode_loop"}}


def test_no_token_row_means_per_request_path():
    got = LLMSubmodule.check_stop_batched(
        _sub(), ["a"], {"a": _info()}, HostRows(("a",), {"other": torch.zeros(1)}),
    )
    assert got is None


def test_engine_falls_back_and_attributes_failures():
    """A batched check that raises (here: a request with no info) is redone
    per request, where the failure lands on that request alone."""
    from mstar.engine.engine import Engine

    failures = {}
    batch = SimpleNamespace(
        node_name="LLM",
        request_ids=["a", "x"],
        per_request_info={"a": _info(done=99, max_tokens=100)},  # "x" missing
        register_failure=failures.setdefault,
        step_context=SimpleNamespace(graph_walk="decode"),
    )
    sub = SimpleNamespace(
        check_stop_batched=lambda *a: LLMSubmodule.check_stop_batched(_sub(), *a),
        check_stop=lambda rid, info, out: LLMSubmodule.check_stop(_sub(), rid, info, out),
    )
    engine = SimpleNamespace(_submodules={"LLM": SimpleNamespace(submodule=sub)})
    tokens = torch.tensor([3, 3])
    outputs = {"a": {"new_token": [tokens[0:1]]}, "x": {"new_token": [tokens[1:2]]}}
    stops = Engine.check_stop_for_batch(
        engine, batch, outputs, host_rows=HostRows(("a", "x"), {"new_token": tokens}),
    )
    assert stops == {"a": {"decode_loop"}}
    assert list(failures) == ["x"]
