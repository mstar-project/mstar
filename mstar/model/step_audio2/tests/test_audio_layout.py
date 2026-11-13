"""Step-Audio2's audio arithmetic and the encoder's key mask, on the CPU.

The reference encodes a request's clips as one padded batch and masks keys by a rule on the
padded length (``modeling_step_audio_2.py:246-264``); the port runs each clip as a segment of
a key-prefix attention. These tests check the token counts, that the mask matters (masked
frames feed kept tokens in 1873 of 4998 cases, so truncating clips is not equivalent), and
that the port's per-clip layout reproduces the reference's padded batch exactly on a small
random tower in float64.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from mstar.model.step_audio2.components.audio import StepAudio2Audio, clip_layout, request_layout
from mstar.model.step_audio2.config import WINDOW_SAMPLES, AudioEncoderConfig, audio_tokens, mel_frames


def compute_token_num(max_feature_len: int) -> int:
    """``utils.py:56-69``, verbatim."""
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding, kernel_size, stride = 1, 3, 2
    return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1


def test_token_counts_follow_the_reference():
    for t in range(3, 2600):
        assert audio_tokens(t) == compute_token_num(t)
    # a full 25 s window: 2502 mel frames (the centred STFT over 400 000 + 479 samples, less one)
    assert mel_frames(WINDOW_SAMPLES) == 2502
    assert audio_tokens(2502) == 313


def masked_frames_feed_a_kept_token(t: int, off: int) -> bool:
    """Whether a key-masked post-conv frame of a clip of ``t`` mel frames (mask offset
    ``off``) is read by one of its kept adapter tokens."""
    x_len = t - 2
    post = (t + 1) // 2
    keys = min(post, math.ceil((x_len - off) / 2))
    tokens = ((x_len + 1) // 2 // 2 - 1) // 2 + 1
    pooled = post // 2
    need = 2 * min(2 * tokens - 1, pooled - 1) + 1  # the last post-conv frame a kept token reads
    return need >= keys


def test_masked_frames_feed_kept_tokens_so_truncating_clips_is_not_equivalent():
    cases = [(t, off) for t in range(4, 2503) for off in (0, 1)]
    assert len(cases) == 4998
    assert sum(masked_frames_feed_a_kept_token(t, off) for t, off in cases) == 1873


def test_the_layout_computes_masked_frames_exactly_when_kept_tokens_read_them():
    for t, padded in ((n, n + extra) for n in range(4, 2503) for extra in (0, 1, 7)):
        clip = clip_layout(t, padded)
        off = (padded + 1) % 2
        post = (padded - 1) // 2 + 1
        assert clip.key_len == min(post, math.ceil((t - 2 - off) / 2))
        assert clip.tokens == audio_tokens(t)
        # every valid key runs; past them, only frames a kept token reads
        reads = min(4 * clip.tokens, 2 * (post // 2))
        assert clip.frames == max(clip.key_len, reads) <= post
        if padded == t:
            assert (reads > clip.key_len) == masked_frames_feed_a_kept_token(t, off)


def test_the_oracle_clips_lay_out_as_recorded():
    # s2t: one 3.8 s clip; s2t_long: a 25 s window and the 2.2 s remainder, padded to it
    assert request_layout([380]) == [clip_layout(380, 380)]
    assert [tuple(c) for c in request_layout([380])] == [(380, 380, 189, 189, 47)]
    assert [tuple(c) for c in request_layout([2502, 225])] == [
        (2502, 2502, 1250, 1250, 313), (225, 225, 112, 111, 28),
    ]


def test_a_clip_with_no_valid_key_is_refused():
    with pytest.raises(ValueError):
        clip_layout(2, 4)


# --- the per-clip layout against the reference's padded batch, on a tiny random tower ---

TINY = AudioEncoderConfig(n_mels=6, n_audio_ctx=64, n_audio_state=16, n_audio_head=2, n_audio_layer=2,
                          llm_dim=12, adapter_hidden=10)


class KeyPrefix:
    """A key-prefix attention resource stand-in: segment ``i`` attends its first ``key_lens[i]``."""

    def __init__(self, layout):
        self.layout = layout

    def run(self, q, k, v):
        out, off = [], 0
        for clip in self.layout:
            n = clip.frames
            qs, ks, vs = (x[off:off + n].transpose(0, 1) for x in (q, k, v))
            o = F.scaled_dot_product_attention(qs, ks[:, : clip.key_len], vs[:, : clip.key_len])
            out.append(o.transpose(0, 1))
            off += n
        return torch.cat(out)


def reference_tower(model: StepAudio2Audio, mels: list[torch.Tensor]) -> list[torch.Tensor]:
    """``AudioEncoder`` + ``Adaptor`` as the reference runs them (``modeling_step_audio_2.py``),
    on the port's weights: a padded batch, ``wav_lens = T - 2``, the additive key mask."""
    enc, ada = model.encoder, model.adapter
    lens = torch.tensor([m.shape[1] - 2 for m in mels])
    x = torch.nn.utils.rnn.pad_sequence([m.t() for m in mels], batch_first=True).transpose(1, 2)
    t = x.shape[-1]
    x = F.gelu(enc.conv2(F.gelu(enc.conv1(x)))).permute(0, 2, 1)
    valid = (torch.arange(t)[None] < lens[:, None])[:, None, (t + 1) % 2::2]
    bias = (1.0 - valid.to(x.dtype)) * -1.0e10  # (B, 1, T2)
    x = x + enc.positional_embedding.weight[: x.shape[1]]
    for block in enc.blocks:
        at = block.self_attn
        h = block.self_attn_layer_norm(x)
        q, k, v = at.qkv_proj(h).split(h.shape[-1], dim=-1)
        b, n, d = q.shape
        heads = at.num_heads
        scale = (d // heads) ** -0.25
        q = q.view(b, n, heads, -1).permute(0, 2, 1, 3) * scale
        k = k.view(b, n, heads, -1).permute(0, 2, 3, 1) * scale
        v = v.view(b, n, heads, -1).permute(0, 2, 1, 3)
        w = torch.softmax(q @ k + bias[:, None], dim=-1)
        x = x + at.out_proj((w @ v).permute(0, 2, 1, 3).flatten(2))
        x = x + block.fc2(block.act(block.fc1(block.final_layer_norm(x))))
    x = enc.after_norm(F.avg_pool1d(x.permute(0, 2, 1), 2, 2).permute(0, 2, 1))
    x = F.gelu(ada.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
    out = ada.linear2(F.relu(ada.linear1(x)))
    return [out[i, : audio_tokens(m.shape[1])] for i, m in enumerate(mels)]


@pytest.fixture(scope="module")
def tiny_tower():
    torch.manual_seed(0)
    model = StepAudio2Audio(TINY, attn_key="attn").double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)
        model.zero_missing_biases()
    return model


@pytest.mark.parametrize("lengths", [[40], [41], [57, 23], [24, 57], [50, 49, 13], [9, 10]])
def test_the_per_clip_layout_reproduces_the_padded_batch(tiny_tower, lengths):
    gen = torch.Generator().manual_seed(sum(lengths))
    mels = [torch.randn(TINY.n_mels, n, generator=gen, dtype=torch.float64) for n in lengths]
    layout = request_layout(lengths)
    for block in tiny_tower.encoder.blocks:
        block.self_attn.bind_resources({"attn": KeyPrefix(layout)})
    with torch.inference_mode():
        got = tiny_tower(mels, layout)
        want = torch.cat(reference_tower(tiny_tower, mels))
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_dropping_the_mask_moves_the_kept_tokens(tiny_tower):
    """The negative control: every post-conv frame a key."""
    mels = [torch.randn(TINY.n_mels, 41, generator=torch.Generator().manual_seed(1), dtype=torch.float64)]
    layout = request_layout([41])
    unmasked = [c._replace(conv_frames=41, frames=21, key_len=21) for c in layout]
    with torch.inference_mode():
        want = torch.cat(reference_tower(tiny_tower, mels))
        for block in tiny_tower.encoder.blocks:
            block.self_attn.bind_resources({"attn": KeyPrefix(unmasked)})
        got = tiny_tower(mels, unmasked)
    assert (got - want).abs().max() > 1e-3


def test_the_encoder_refuses_to_run_without_its_attention_resource():
    model = StepAudio2Audio(TINY, attn_key="attn")
    with pytest.raises(RuntimeError, match="RaggedKeyPrefixAttentionSpec"):
        model([torch.zeros(TINY.n_mels, 20)], request_layout([20]))
