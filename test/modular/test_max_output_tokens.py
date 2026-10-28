"""A request gets max_output_tokens tokens, the prefill's first token included.

Each model's own walks, loop caps, postprocess and stop check run through the
Python graph runtime (``request_walks.run_walks``), with the conductor's count of
the prefill's token between walks. The budget is the decode loop's cap; the stop
check ends the loop only on a stop token.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from request_walks import run_walks

from mstar.engine.resources import SamplingReqConfig
from mstar.graph.base import Loop
from mstar.model.submodule_base import HostRows

TOKEN = 5
EOS = 7


class Case(NamedTuple):
    model: object
    # the decoder node's submodule: its postprocess and stop check run on each step
    sub: object
    node: str
    sampler: str
    signals: dict | None = None
    # the step metadata of the request's (one) prefill walk
    prefill_metadata: dict | None = None
    batched: bool = False


def _qwen3_5(batched: bool = False) -> Case:
    from mstar.model.qwen3_5.config import SAMPLER
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    model = object.__new__(Qwen3_5DenseModel)
    model.config = SimpleNamespace(max_position_embeddings=4096)
    sub = object.__new__(LLMSubmodule)
    sub.config = SimpleNamespace(stop_token_ids=frozenset({EOS}))
    return Case(model, sub, "LLM", SAMPLER, prefill_metadata={"last_prefill": True}, batched=batched)


def _higgs_audio() -> Case:
    from mstar.model.higgs_audio.config import SAMPLER, HiggsAudioModelConfig
    from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule

    model = object.__new__(HiggsAudioModel)
    model.config = HiggsAudioModelConfig(stop_token_ids=[EOS])
    return Case(
        model, HiggsAudioLLMSubmodule(None, model.config), "LLM", SAMPLER,
        prefill_metadata={"is_last_prefill": True},
    )


def _audio_signals(prompt):
    return {"audio": [torch.zeros(16)], "text_inputs": [prompt]}


def _canary() -> Case:
    from mstar.model.canary.canary_model import CanaryModel
    from mstar.model.canary.config import SAMPLER
    from mstar.model.canary.submodules import CanaryDecoderSubmodule

    model = object.__new__(CanaryModel)
    model.config = SimpleNamespace(max_position_embeddings=1024)
    sub = object.__new__(CanaryDecoderSubmodule)
    sub.stop_token_ids = frozenset({EOS})
    return Case(model, sub, "decoder", SAMPLER, signals=_audio_signals)


def _canary_qwen() -> Case:
    from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel
    from mstar.model.canary_qwen.config import SAMPLER
    from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule

    sub = object.__new__(CanaryQwenLLMSubmodule)
    sub.stop_token_ids = frozenset({EOS})
    return Case(object.__new__(CanaryQwenModel), sub, "llm", SAMPLER, signals=_audio_signals)


def _qwen3_asr() -> Case:
    from mstar.model.qwen3_asr.config import LLM_NODE, SAMPLER
    from mstar.model.qwen3_asr.qwen3_asr_model import Qwen3ASRModel
    from mstar.model.qwen3_asr.submodules import Qwen3ASRLLMSubmodule

    model = object.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(text=SimpleNamespace(max_position_embeddings=4096), max_new_tokens=4096)
    sub = object.__new__(Qwen3ASRLLMSubmodule)
    sub.stop_token_ids = frozenset({EOS})
    return Case(model, sub, LLM_NODE, SAMPLER, signals=_audio_signals)


CASES = {
    "qwen3_5": _qwen3_5,
    "qwen3_5 batched": lambda: _qwen3_5(batched=True),
    "higgs_audio": _higgs_audio,
    "canary": _canary,
    "canary_qwen": _canary_qwen,
    "qwen3_asr": _qwen3_asr,
}


def _client_tokens(case: Case, max_tokens: int, eos_at: int | None = None) -> list[int]:
    """The tokens a request with ``max_tokens`` sends; the ``eos_at``-th is a stop token."""
    walks = case.model.get_graph_walk_graphs()
    sampled = 0

    def step(walk, inputs, iteration):
        nonlocal sampled
        loop = walks[walk]
        info = SimpleNamespace(
            graph_walk=walk, rid_handle="r", max_tokens=max_tokens,
            step_metadata={} if isinstance(loop, Loop) else dict(case.prefill_metadata or {}),
            resource_configs={case.sampler: SamplingReqConfig(ignore_eos=False)},
            dynamic_loop_iter_counts={loop.name: iteration} if isinstance(loop, Loop) else {},
        )
        sampled += 1
        token = EOS if sampled == eos_at else TOKEN
        outputs = {"new_token": [torch.tensor([token])]}
        case.sub.postprocess("r", info, outputs)
        if case.batched:
            rows = HostRows(("r",), {"new_token": torch.tensor([token])})
            stops = case.sub.check_stop_batched(["r"], {"r": info}, rows, step_walk=walk).get("r", set())
        else:
            stops = case.sub.check_stop("r", info, outputs)
        return outputs, stops

    prompt = torch.tensor([1, 2, 3])
    signals = case.signals(prompt) if callable(case.signals) else case.signals
    return run_walks(case.model, step, prompt, max_tokens, max_steps=200, node=case.node, signals=signals)


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 64])
@pytest.mark.parametrize("model", list(CASES))
def test_a_request_gets_max_output_tokens(model, max_tokens):
    assert len(_client_tokens(CASES[model](), max_tokens)) == max_tokens


@pytest.mark.parametrize("model", list(CASES))
def test_a_stop_token_before_the_budget_ends_the_reply(model):
    tokens = _client_tokens(CASES[model](), 64, eos_at=10)
    assert len(tokens) == 10 and tokens[-1] == EOS
