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

from mstar.model.multimodal import vision_encoder
from mstar.model.muse.submodules import LLMSubmodule as GlimmerLLMSubmodule
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_5.submodules import LLMSubmodule

LOOP = "decode_loop"
EOS = (7, 9)

MODELS = {
    # (submodule, its config, its walks, its TokenStop if it declares one)
    "qwen3_5": (LLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                ("decode", "prefill_text", "prefill_vision"),
                LLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                             encoders=(vision_encoder((5,)),)).token_stop),
    "orpheus": (OrpheusLLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)), ("prefill", "decode"),
                OrpheusLLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS))).token_stop),
    # the shared VLM decoder's stop, as Muse Glimmer serves it
    "muse_glimmer": (GlimmerLLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                     ("decode", "prefill_text", "prefill_vision"),
                     GlimmerLLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                                         (vision_encoder((5,)),)).token_stop),
}


def _sampler_key(cls):
    return getattr(cls, "SAMPLER", None) or __import__(cls.__module__, fromlist=["SAMPLER"]).SAMPLER


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
    cls, config, walks, stop = MODELS[model]
    sampler = _sampler_key(cls)
    sub = cls.__new__(cls)
    torch.nn.Module.__init__(sub)
    sub.config = config
    if stop is not None:
        sub.token_stop = stop
    rng = np.random.default_rng(seed)
    declared_walks = 0
    for walk in walks:
        policies = sub.get_stop_policies(walk)
        if policies is None:
            continue  # check_stop decides (a TokenStop's prompt walks)
        declared_walks += 1
        for rid in range(16):
            token = int(rng.choice([*EOS, 1, 2, 3]))
            info = _info(rng, walk, sampler)
            expected = sub.check_stop(rid, info, {"new_token": [torch.tensor([token])]}) or set()
            declared = {
                loop for loop, p in policies.items()
                if p.stops(info.dynamic_loop_iter_counts.get(loop, 0), token, p.limits(info))
            }
            assert declared == set(expected), (walk, rid)
    assert declared_walks
