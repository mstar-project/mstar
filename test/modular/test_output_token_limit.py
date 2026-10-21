"""A decode loop built with ``max_iters`` from ``get_max_output_tokens()`` ends
there whatever the request asked, so a request for more tokens would get fewer,
with no error. Each such model publishes the most tokens a request can get, and the
conductor refuses a request above it with a 400.
"""
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.submodules import LLMSubmodule as BagelLLMSubmodule
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import TalkerSubmodule as Qwen3TTSTalkerSubmodule


def _bare(cls, **attrs):
    model = object.__new__(cls)
    model.__dict__.update(attrs)
    return model


def _qwen3_omni(pick):
    # Qwen3-Omni's submodules import flashinfer at module level.
    pytest.importorskip("flashinfer")
    from mstar.model.qwen3_omni import submodules
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    return _bare(Qwen3OmniModel), pick(submodules)


def _case(model, submodule_cls, decode_walk: str, walks=None):
    """The model, the submodule whose stop check ends its decode loop, the loop's walk and the loop itself."""
    return model, submodule_cls, decode_walk, (walks or model.get_graph_walk_graphs())[decode_walk]


CASES = {
    "orpheus": lambda: _case(_bare(OrpheusModel), OrpheusLLMSubmodule, "decode"),
    "qwen3_omni_thinker": lambda: _case(*_qwen3_omni(lambda s: s.ThinkerSubmodule), "thinker_decode"),
    "qwen3_omni_talker": lambda: _case(*_qwen3_omni(lambda s: s.TalkerSubmodule), "talker_decode"),
    "bagel": lambda: _case(
        _bare(BagelModel, config=SimpleNamespace(num_timesteps=50), _image_gen_remote_handoff=False,
              _has_cfg_parallel=False),
        BagelLLMSubmodule, "decode",
    ),
    "cosmos3": lambda: _case(_bare(Cosmos3Model), Cosmos3ReasonerSubmodule, "reasoner_decode",
                             _bare(Cosmos3Model)._reasoner_walks()),
    "higgs_audio": lambda: _case(_bare(HiggsAudioModel), HiggsAudioLLMSubmodule, "decode"),
    "qwen3_tts": lambda: _case(_bare(Qwen3TTSModel, config=Qwen3TTSModelConfig()), Qwen3TTSTalkerSubmodule,
                               "talker_decode"),
}
_LIMIT = "max_output_tokens is {}; this model produces at most {}$"


def _served(model, submodule_cls, decode_walk: str, loop, max_tokens: int) -> int:
    """Count the token the prefill samples, then one per decode iteration until
    the stop check ends ``loop`` or ``loop`` reaches its ``max_iters``. No EOS id
    matches the sampled token."""
    eos = SimpleNamespace(stop_token_id=-1, im_end_token_id=-1, stop_token_ids=[], codec_eos_token_id=-1)
    submodule = _bare(submodule_cls, config=SimpleNamespace(**vars(eos), talker=eos), talker_config=eos,
                      eos_token_id=-1, request_states={})
    served = 0
    for walk, k in [("prefill", 0)] + [(decode_walk, k) for k in range(loop.max_iters)]:
        served += 1
        info = CurrentForwardPassInfo(
            request_id=0, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=max_tokens,
            step_metadata={"is_last_prefill": True, "sample_prefill_token": True, "audio_output": False,
                           "talker_max_tokens": max_tokens},
            dynamic_loop_iter_counts={loop.name: k},
            resource_configs=defaultdict(lambda: SimpleNamespace(ignore_eos=False)),
        )
        outputs = {"new_token": [torch.tensor([0])]}
        submodule.postprocess(0, info, outputs)
        if walk == decode_walk and loop.name in submodule.check_stop(0, info, outputs):
            break
    return served


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_the_limit_is_the_most_tokens_a_request_gets(case):
    model, *decode = case()
    limit = model.get_max_output_tokens_limit()
    assert _served(model, *decode, limit) == limit, "a request at the limit would get fewer tokens than it asked for"
    assert _served(model, *decode, limit + 1) == limit, "the limit would refuse a request the loop serves in full"


def test_the_conductor_serves_the_limit_and_refuses_past_it():
    conductor = _bare(Conductor, model=CASES["qwen3_tts"]()[0])
    limit = conductor.model.get_max_output_tokens_limit()
    assert conductor._max_output_tokens({"max_output_tokens": limit}) == limit
    for cap in ("max_output_tokens", "max_new_tokens"):
        with pytest.raises(ValueError, match=_LIMIT.format(limit + 1, limit)):
            conductor._max_output_tokens({cap: limit + 1})

