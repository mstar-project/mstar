"""Qwen3-Omni's Thinker builds an audio row (a clip between its sentinels and
the text around it) from one embedding call, and the row is the one the spans
would give laid end to end."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.qwen3_omni.config import THINKER_POS, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.qwen3_omni_model import TEXT_BEFORE
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule, talker_text_mask

HIDDEN = 8
START = 5


class _CountingEmbedding(nn.Embedding):
    calls = 0

    def forward(self, ids):
        self.calls += 1
        return super().forward(ids)


@pytest.fixture
def config():
    return Qwen3OmniModelConfig()


@pytest.fixture
def thinker(config):
    thinker = ThinkerSubmodule.__new__(ThinkerSubmodule)
    nn.Module.__init__(thinker)
    thinker.config = config
    ids = (config.thinker.audio_start_token_id, config.thinker.audio_end_token_id,
           config.im_start_token_id, config.im_end_token_id, config.assistant_token_id)
    thinker.model = SimpleNamespace(model=SimpleNamespace(
        embed_tokens=_CountingEmbedding(max(config.thinker_text.vocab_size, *ids) + 1, HIDDEN),
    ))
    # the code before `_audio_row` cached the sentinel embeddings here; set as its
    # __init__ did, so on that code only the call count fails
    thinker._audio_bos_embed = thinker._audio_eos_embed = None
    thinker.__dict__["_cached_device"] = torch.device("cpu")
    thinker.node_resources = {THINKER_POS: SimpleNamespace(position=lambda rid, label: START)}
    return thinker


def _prepare(thinker, texts, text_before, audio):
    info = SimpleNamespace(rid_handle=0, step_metadata={TEXT_BEFORE: text_before})
    return thinker.prepare_inputs("prefill_audio", info, {"audio_embeds": [audio], "text_inputs": texts})


@pytest.mark.parametrize("text_before,texts", [
    (1, [[1, 9, 10], [11, 2, 1, 3]]),  # text before and after the clip
    (0, [[12, 13]]),  # after only
    (1, [[14]]),  # before only
    (0, []),  # the clip alone
])
def test_an_audio_row_is_its_spans_end_to_end(thinker, config, text_before, texts):
    texts = [torch.tensor(t) for t in texts]
    before = torch.cat([torch.zeros(0, dtype=torch.long), *texts[:text_before]])
    after = torch.cat([torch.zeros(0, dtype=torch.long), *texts[text_before:]])
    audio = torch.randn(7, HIDDEN)
    embed = thinker.model.model.embed_tokens

    row = _prepare(thinker, texts, text_before, audio)

    sentinels = embed(torch.tensor([config.thinker.audio_start_token_id, config.thinker.audio_end_token_id]))
    p, clip = len(before), len(audio) + 2
    n = p + clip + len(after)
    assert row.input_seq_len == n
    assert torch.equal(row.input_embeds, torch.cat([embed(before), sentinels[:1], audio, sentinels[1:], embed(after)]))
    # text and audio both advance all three MRoPE components by one a row
    assert torch.equal(row.custom_pos_ids, (torch.arange(n, dtype=torch.float) + START).expand(3, -1))
    media = torch.zeros(n, dtype=torch.bool)
    media[p + 1:p + clip - 1] = True
    masks = row.tensor_inputs["masks_for_talker"]
    assert torch.equal(masks[0], media)
    assert torch.equal(masks[1], torch.cat([
        talker_text_mask(before, config), torch.ones(clip, dtype=torch.bool), talker_text_mask(after, config),
    ]) & ~media)
    assert row.tensor_inputs["mrope_pos_advance"] == n
    assert torch.equal(row.tensor_inputs["text_token_ids"], torch.cat([before, after]))


def test_an_audio_row_embeds_its_ids_in_one_call(thinker):
    texts = [torch.tensor([1, 9, 10]), torch.tensor([11, 2])]
    _prepare(thinker, texts, 1, torch.randn(7, HIDDEN))
    embed = thinker.model.model.embed_tokens
    embed.calls = 0
    _prepare(thinker, texts, 1, torch.randn(7, HIDDEN))
    assert embed.calls == 1
