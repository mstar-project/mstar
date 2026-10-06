"""The Thinker reports each row's KV span from its inputs alone, so admission
can leave out a row whose KV does not fit before prepare_inputs runs."""
import pytest
import torch

from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

HIDDEN = 4


@pytest.fixture
def thinker():
    return object.__new__(ThinkerSubmodule)


def _len(thinker, walk, inputs):
    info = thinker.get_input_sequence_len(walk, None, inputs)
    return None if info is None else info.seq_len


def test_text_is_its_token_count(thinker):
    assert _len(thinker, "prefill_text", {"text_inputs": [torch.zeros(7)]}) == 7


@pytest.mark.parametrize("walk,name", [
    ("prefill_audio", "audio_embeds"), ("prefill_vision", "vision_embeds"),
])
def test_embeds_count_their_two_sentinels(thinker, walk, name):
    assert _len(thinker, walk, {name: [torch.zeros(10, HIDDEN)]}) == 12


def test_decode_is_one_token(thinker):
    assert _len(thinker, "thinker_decode", {}) == 1


def test_a_missing_input_declares_nothing(thinker):
    assert _len(thinker, "prefill_audio", {}) is None
