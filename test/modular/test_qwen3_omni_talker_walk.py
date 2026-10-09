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


def test_the_first_decode_pass_closes_the_talker_prefill():
    walks = [
        _talker_walk(ProducerWalkCtx(walk, n, None, _FWD))
        for walk, n in [
            ("prefill_text", 0), ("prefill_audio", 0), ("prefill_text", 0),
            ("thinker_decode", 0), ("thinker_decode", 1), ("thinker_decode", 2),
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
