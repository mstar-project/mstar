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


def test_each_prefill_walk_runs_the_talker_prefill_and_decode_runs_its_decode():
    walks = [
        _talker_walk(ProducerWalkCtx(walk, "Thinker", iters, _FWD))
        for walk, iters in [
            ("prefill_text", {}), ("prefill_audio", {}), ("prefill_vision", {}), ("prefill_audio_vision", {}),
            # the first decode iteration streams nothing, so the hook's answer for it only names where the next run
            ("thinker_decode", {"thinker_decode_loop": 0}),
            ("thinker_decode", {"thinker_decode_loop": 1}),
        ]
    ]
    assert walks == ["talker_prefill"] * 4 + ["talker_decode"] * 2


def test_only_the_decode_loop_runs_past_the_thinkers_end():
    policy = _thinker_to_talker_policy()
    assert policy.continues_in("talker_decode")
    assert not policy.continues_in("talker_prefill")
