"""A submodule's declared stop policies agree with its own ``check_stop``.

The engine runs ``get_stop_policies`` in place of ``check_stop`` when a walk
declares them, so for every model that declares any, both must stop the same
rows, on every walk, over random tokens, iteration counts, limits and
``ignore_eos`` settings.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import HostRows, StopKind, StopPolicy

LOOP = "decode_loop"
EOS = (7, 9)

MODELS = {
    "qwen3_5": (LLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                ("decode", "prefill_text", "prefill_vision")),
    "orpheus": (OrpheusLLMSubmodule, SimpleNamespace(stop_token_id=EOS[0]), ("prefill", "decode")),
}


def _sampler_key(cls):
    return __import__(cls.__module__, fromlist=["SAMPLER"]).SAMPLER


def _info(rng, walk, sampler):
    max_tokens = int(rng.integers(1, 6))
    return SimpleNamespace(
        graph_walk=walk, step_metadata={}, max_tokens=max_tokens,
        resource_configs={sampler: SimpleNamespace(ignore_eos=bool(rng.integers(2)))},
        dynamic_loop_iter_counts={LOOP: int(rng.integers(0, max_tokens + 1))},
    )


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {"N": SimpleNamespace(submodule=sub)}
    eng._stop_policies = {}
    eng._stop_limits = {}
    return eng


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("seed", range(8))
def test_declared_policies_stop_the_rows_check_stop_does(model, seed):
    cls, config, walks = MODELS[model]
    sampler = _sampler_key(cls)
    sub = cls.__new__(cls)
    torch.nn.Module.__init__(sub)
    sub.config = config
    rng = __import__("numpy").random.default_rng(seed)
    for walk in walks:
        assert sub.get_stop_policies(walk) is not None, "every walk is declared"
        n = 16
        rids = list(range(n))
        tokens = torch.tensor(rng.choice([*EOS, 1, 2, 3], size=n))
        infos = {r: _info(rng, walk, sampler) for r in rids}
        expected = {}
        for r in rids:
            got = sub.check_stop(r, infos[r], {"new_token": [tokens[r:r + 1]]})
            if got:
                expected[r] = set(got)
        batch = SimpleNamespace(
            node_name="N", request_ids=rids, per_request_info_wrapped=infos,
            step_context=SimpleNamespace(graph_walk=walk, request_walks=None),
        )
        declared = Engine.check_stop_for_batch(
            _engine(sub), batch, {}, host_rows=HostRows(tuple(rids), {"new_token": tokens}),
        )
        assert declared == expected, walk


def test_limits_are_computed_once_per_request_and_walk_and_dropped_with_it():
    calls = []

    def max_iters(info):
        calls.append(info.graph_walk)
        return info.max_tokens

    policy = StopPolicy(StopKind.ITER, LOOP, max_iters=max_iters)
    sub = SimpleNamespace(get_stop_policies=lambda walk: (policy,))
    eng = _engine(sub)

    def check(walk, done):
        info = SimpleNamespace(
            graph_walk=walk, max_tokens=3, resource_configs={},
            dynamic_loop_iter_counts={LOOP: done},
        )
        batch = SimpleNamespace(
            node_name="N", request_ids=[5], per_request_info_wrapped={5: info},
            step_context=SimpleNamespace(graph_walk=walk, request_walks=None),
        )
        return Engine.check_stop_for_batch(eng, batch, {}, host_rows=None)

    assert check("a", 0) == {} and check("a", 1) == {} and check("a", 2) == {5: {LOOP}}
    assert calls == ["a"]
    check("b", 0)
    assert calls == ["a", "b"], "a new walk computes its own limits"
    Engine.remove_request(SimpleNamespace(
        _loopback_steps={}, _stop_limits=eng._stop_limits,
        _runner=SimpleNamespace(remove_request=lambda rid: None), _submodules={},
    ), 5)
    check("a", 0)
    assert calls == ["a", "b", "a"], "a recycled handle does not inherit the limits"


def test_a_step_that_mixes_walks_takes_check_stop():
    sub = SimpleNamespace(
        get_stop_policies=lambda walk: pytest.fail("a mixed step reads no policy"),
        check_stop_batched=lambda *a, **k: None,
        check_stop=lambda rid, info, outputs: {LOOP},
    )
    batch = SimpleNamespace(
        node_name="N", request_ids=[1], per_request_info_wrapped={1: None},
        per_request_info={1: None},
        step_context=SimpleNamespace(graph_walk="mixed", request_walks={1: "decode"}),
    )
    out = Engine.check_stop_for_batch(_engine(sub), batch, {1: {"new_token": [torch.tensor([1])]}})
    assert out == {1: {LOOP}}
