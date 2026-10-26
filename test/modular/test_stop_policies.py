"""A submodule's declared stop policies agree with its own ``check_stop``.

The graph runtime runs ``get_stop_policies`` in place of ``check_stop`` when
a walk declares them, so for every model that declares any, each row a
policy stops (``StopPolicy.stops``) must be one ``check_stop`` stops, on
every walk, over random tokens, iteration counts, limits and ``ignore_eos``
settings.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_5.submodules import LLMSubmodule

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


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("seed", range(8))
def test_declared_policies_stop_the_rows_check_stop_does(model, seed):
    cls, config, walks = MODELS[model]
    sampler = _sampler_key(cls)
    sub = cls.__new__(cls)
    torch.nn.Module.__init__(sub)
    sub.config = config
    rng = np.random.default_rng(seed)
    for walk in walks:
        policies = sub.get_stop_policies(walk)
        assert policies is not None, "every walk is declared"
        for rid in range(16):
            token = int(rng.choice([*EOS, 1, 2, 3]))
            info = _info(rng, walk, sampler)
            expected = sub.check_stop(rid, info, {"new_token": [torch.tensor([token])]}) or set()
            declared = {
                loop for loop, p in policies.items()
                if p.stops(info.dynamic_loop_iter_counts.get(loop, 0), token, p.limits(info))
            }
            assert declared == set(expected), (walk, rid)
