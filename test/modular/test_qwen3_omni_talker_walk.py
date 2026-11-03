"""Qwen3-Omni's Thinker decides the Talker's walk.

Not in cpu-core: importing the model needs flashinfer. The mechanism itself is
tested in test_stream_walk_transitions.py.
"""
import pytest

pytest.importorskip("flashinfer")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.qwen3_omni.qwen3_omni_model import (  # noqa: E402
    _talker_walk,
    _thinker_to_talker_policy,
)
from mstar.streaming.topology import ProducerWalkCtx  # noqa: E402

_FWD = CurrentForwardPassInfo(
    request_id="r", graph_walk="", fwd_index=0, random_seed=0, max_tokens=0,
)


def test_the_first_decode_iteration_closes_the_talker_prefill():
    walks = [
        _talker_walk(ProducerWalkCtx(walk, "Thinker", iters, _FWD))
        for walk, iters in [
            ("prefill_text", {}), ("prefill_audio", {}), ("prefill_text", {}),
            ("thinker_decode", {"thinker_decode_loop": 0}),
            ("thinker_decode", {"thinker_decode_loop": 1}),
            ("thinker_decode", {"thinker_decode_loop": 2}),
        ]
    ]
    assert walks == ["talker_prefill"] * 3 + [
        "talker_last_prefill", "talker_decode", "talker_decode",
    ]


def test_only_the_decode_loop_runs_past_the_thinkers_end():
    policy = _thinker_to_talker_policy()
    assert policy.continues_in("talker_decode")
    assert not policy.continues_in("talker_prefill")
    assert not policy.continues_in("talker_last_prefill")


def test_the_talker_takes_the_thinkers_prefill_a_chunk_at_a_time():
    """Each chunk's states go out as it runs, into a consumer that takes them,
    so the Talker extends its KV per chunk rather than after the whole prompt."""
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    from mstar.validation.placement import _chunked_streams

    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.config = Qwen3OmniModelConfig()
    assert _thinker_to_talker_policy().allow_partial_input()
    for (_node, _walk), policy in model.chunking_policy_of().items():
        if policy.chunkable:
            assert policy.per_chunk_outputs == {"thinker_states", "thinker_mask"}
            assert not policy.accumulated_outputs
    assert list(_chunked_streams(model, model.get_graph_walk_graphs())) == []


def _postprocess(text_inclusion):
    import torch

    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    thinker = object.__new__(ThinkerSubmodule)
    thinker.config = Qwen3OmniModelConfig()
    hidden = thinker.config.thinker_hidden_size
    n = len(text_inclusion)
    outputs = {
        "thinker_states": [torch.zeros(n, 2 * hidden)],
        "thinker_mask": [torch.stack([torch.zeros(n, dtype=torch.bool), torch.tensor(text_inclusion)])],
    }
    info = CurrentForwardPassInfo(
        request_id="r", graph_walk="prefill_text", fwd_index=0, random_seed=0, max_tokens=0,
    )
    thinker.postprocess("r", info, outputs)
    return outputs


def test_a_chunk_of_only_system_prompt_sends_the_talker_nothing():
    """The Talker would otherwise run a prefill with no tokens in it."""
    assert "thinker_states" not in _postprocess([False, False, False])
    assert "thinker_mask" not in _postprocess([False, False, False])


def test_a_chunk_with_kept_tokens_sends_just_those():
    outputs = _postprocess([False, True, True])
    assert outputs["thinker_states"][0].shape[0] == 2
    assert outputs["thinker_mask"][0].shape == (2,)
