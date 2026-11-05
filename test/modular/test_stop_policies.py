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
from mstar.model.orpheus.submodules import token_stop as orpheus_token_stop
from mstar.model.qwen3_5.submodules import LLMSubmodule, token_stop
from mstar.model.qwen3_asr.submodules import Qwen3ASRLLMSubmodule
from mstar.model.submodule_base import TokenStop

LOOP = "decode_loop"
EOS = (7, 9)

MODELS = {
    # (submodule, its config, its walks, its TokenStop if it declares one)
    "qwen3_5": (LLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                ("decode", "prefill_text", "prefill_vision"), token_stop(EOS)),
    "orpheus": (OrpheusLLMSubmodule, SimpleNamespace(stop_token_id=EOS[0]), ("prefill", "decode"),
                orpheus_token_stop(EOS[0])),
    # a `TokenStepNode`: the runtime checks its decode walk, which routes the token as text_inputs
    "qwen3_asr": (Qwen3ASRLLMSubmodule, None, ("prefill", "decode"), TokenStop(
        loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler=Qwen3ASRLLMSubmodule.SAMPLER,
        walk="decode", prefill_walks=frozenset({"prefill"}),
    )),
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


def test_a_token_stop_names_the_walks_that_end_its_loop():
    from mstar.model.submodule_base import TokenStop

    def stop(walk):
        return TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s", walk=walk)

    assert stop(None).stops_on("a")
    assert stop("a").stops_on("a") and not stop("a").stops_on("b")
    both = stop(frozenset({"a", "b"}))
    assert both.stops_on("a") and both.stops_on("b") and not both.stops_on("c")


def test_a_token_stop_keeps_check_stop_where_check_stop_does_more():
    """The runtime's check never calls ``check_stop``: not for a class whose
    own check does more (Whisper keeps each token for its align walk)."""
    from mstar.model.whisper.submodules import WhisperDecoderSubmodule

    sub = WhisperDecoderSubmodule.__new__(WhisperDecoderSubmodule)
    torch.nn.Module.__init__(sub)
    sub.token_stop = TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s")
    assert sub.get_stop_policies("decode") is None


def test_a_withholding_token_stop_is_the_runtimes_too():
    sub = Qwen3ASRLLMSubmodule.__new__(Qwen3ASRLLMSubmodule)
    torch.nn.Module.__init__(sub)
    sub.token_stop = TokenStop(
        loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s", withhold=("codes",),
    )
    assert sub.get_stop_policies("decode")[LOOP].withhold == ("codes",)


@pytest.mark.parametrize(("module", "name", "policies"), [
    # a count-only node: the loop's cap (or no loop) stops it
    ("mstar.model.orpheus.submodules", "SNACDecoderSubmodule", {}),
    # an AR node with no token stop
    ("mstar.model.qwen3_asr.submodules", "Qwen3ASRLLMSubmodule", {}),
    # a denoise loop: the request's step cap stops it
    ("mstar.model.z_image.submodules", "ZImageDenoiseSubmodule", {}),
    ("mstar.model.vjepa2.submodules", "VJepa2RolloutPredictorSubmodule", {}),
    # its own check_stop decides
    ("mstar.model.waypoint.submodules", "WaypointDitSubmodule", None),
])
def test_a_class_without_its_own_stop_check_declares_no_stops(module, name, policies):
    cls = getattr(__import__(module, fromlist=[name]), name)
    sub = cls.__new__(cls)
    torch.nn.Module.__init__(sub)
    assert sub.get_stop_policies("decode") == policies


def test_the_omni_talker_withholds_its_codec_eos_frame_through_its_token_stop():
    """The Talker's stop is a ``TokenStop`` like any other: the runtime checks
    it, and its codec_eos frame stays out of the Code2Wav stream."""
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule, talker_token_stop

    config = Qwen3OmniModelConfig()
    sub = TalkerSubmodule.__new__(TalkerSubmodule)
    torch.nn.Module.__init__(sub)
    sub.token_stop = talker_token_stop(config)
    (policy,) = sub.get_stop_policies("talker_decode").values()
    assert policy.eos_tokens == {config.talker.codec_eos_token_id}
    assert policy.withhold == ("codec_tokens",)
