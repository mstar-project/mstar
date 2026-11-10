"""Step-Audio2's audio tower: a Whisper-large-v3 encoder, a 2x average pool, and an adapter
(a stride-2 conv, then two linears into the LLM's width), ~12.5 LLM tokens a second.

Ported from ``AudioEncoder`` and ``Adaptor`` in the checkpoint's
``modeling_step_audio_2.py`` (Apache-2.0). The layers are Whisper's, which is what
``AuTEncoderLayer`` is; the loader renames the checkpoint's ``attn_ln`` / ``attn.query`` /
``mlp.0`` ... onto it.

**The key padding mask.** The reference encodes a request's clips as one padded batch and
masks keys by a rule on the batch's padded length ``T_pad`` (its longest clip):
post-conv frame ``j`` of a clip of ``M`` mel frames is a valid key iff
``2 j + (T_pad + 1) % 2 < M - 2`` (``modeling_step_audio_2.py:251-252``, fed
``wav_lens = M - 2`` by ``utils.py:84``). Masked frames still run as queries and feed the
kept tokens through the pool and the adapter's conv, so they cannot be dropped: here every
clip is one segment of the ``RaggedKeyPrefixAttentionSpec`` resource, each of its frames a
query, attending its first ``key_len`` frames. ``clip_layout`` computes both on the host.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.aut_encoder import AuTEncoderConfig, AuTEncoderLayer
from mstar.model.step_audio2.config import AudioEncoderConfig, audio_tokens


class ClipLayout(NamedTuple):
    """How one clip (a 25 s window) runs through the tower."""

    # the clip's mel frames, M
    mel_frames: int
    # mel frames the convolutions see: the clip zero-padded as far as the reference's batch
    # padding reaches the frames computed here
    conv_frames: int
    # post-conv frames computed (the clip's attention segment): those the kept tokens read
    frames: int
    # the segment's valid keys, a prefix of it
    key_len: int
    # LLM tokens kept
    tokens: int


def clip_layout(mel_frames: int, padded_frames: int) -> ClipLayout:
    """The layout of a clip of ``mel_frames`` in a request whose longest clip has
    ``padded_frames`` (the reference's ``T_pad``).

    Two kinds of post-conv frame matter. Every valid key feeds every frame of its clip, so
    those run in full. And kept token ``k`` reads pooled frames ``2k - 1 .. 2k + 1`` (the
    adapter's conv), so post-conv frames up to ``4k + 3``: the first ``4 * tokens``, or all
    the reference pools (``2 * (T2 // 2)``) if fewer; those past the valid keys are queries
    only. Any further frame is a masked key that nothing kept reads, so it is not computed.
    The frames read mel frames up to ``2 * frames``; past the clip's own the reference's
    batch has zeros, up to ``T_pad``, which the convolutions here see too.
    """
    if not 3 <= mel_frames <= padded_frames:
        raise ValueError(f"a clip of {mel_frames} mel frames in a batch padded to {padded_frames}")
    tokens = audio_tokens(mel_frames)
    padded_post_conv = (padded_frames - 1) // 2 + 1
    # frame j's mask entry is mel position 2j + offset (``mask[:, :, (T + 1) % 2::2]``)
    offset = (padded_frames + 1) % 2
    key_len = min(padded_post_conv, max(0, (mel_frames - 2 - offset + 1) // 2))
    if key_len == 0:
        raise ValueError(f"a clip of {mel_frames} mel frames has no valid key")
    frames = max(key_len, min(4 * tokens, 2 * (padded_post_conv // 2)))
    conv_frames = min(padded_frames, max(mel_frames, 2 * frames))
    return ClipLayout(mel_frames, conv_frames, frames, key_len, tokens)


def request_layout(mel_frames: list[int]) -> list[ClipLayout]:
    """One request's clips, padded to its longest as the reference pads them
    (``padding_mels``, ``utils.py:71-91``)."""
    padded = max(mel_frames)
    return [clip_layout(n, padded) for n in mel_frames]


def _padded(n: int, quantum: int) -> int:
    """``n`` rounded up to a multiple of ``quantum``.

    A convolution's input zero-padded on its right gives the same outputs as the unpadded
    one where those read only the input and its own zero padding, so a clip's convolutions
    can run at one of a few widths. cuDNN builds a plan per input width, which costs 5-20 ms
    the first time a width is seen (H200), so with clips of every length it would land on
    almost every request."""
    return -(-n // quantum) * quantum


class WhisperEncoder(nn.Module):
    """``AudioEncoder``; parameter paths mirror the checkpoint's ``encoder.*`` but for each
    block's internals, which are ``AuTEncoderLayer``'s."""

    def __init__(self, config: AudioEncoderConfig, attn_key: str | None):
        super().__init__()
        self.config = config
        d = config.n_audio_state
        self.conv1 = nn.Conv1d(config.n_mels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(config.n_audio_ctx, d)
        layer_config = AuTEncoderConfig(
            d_model=d,
            encoder_layers=config.n_audio_layer,
            encoder_attention_heads=config.n_audio_head,
            encoder_ffn_dim=config.ffn_dim,
            activation_function="gelu",
        )
        self.blocks = nn.ModuleList(
            AuTEncoderLayer(layer_config, attn_key=attn_key) for _ in range(config.n_audio_layer)
        )
        self.after_norm = nn.LayerNorm(d)

    def frontend(self, mels: list[torch.Tensor], layout: list[ClipLayout], quantum: int = 1) -> torch.Tensor:
        """Per clip ``[n_mels, M]`` -> packed ``[sum frames, d]``, positions from 0 per clip.

        ``quantum``: run the convolutions over the clip zero-padded on to a multiple of this
        many mel frames, which gives the same frames (``_padded``) on one of a few widths."""
        dtype = self.conv1.weight.dtype
        out = []
        for mel, clip in zip(mels, layout, strict=True):
            width = _padded(clip.conv_frames, quantum)
            x = F.pad(mel.to(dtype), (0, width - mel.shape[-1]))[None]
            h = F.gelu(self.conv1(x))
            if width > clip.conv_frames:
                # conv2 reads past the clip's conv frames as its own zero padding
                h[..., clip.conv_frames:] = 0
            x = F.gelu(self.conv2(h))[0, :, : clip.frames].transpose(0, 1)
            out.append(x + self.positional_embedding.weight[: clip.frames])
        return torch.cat(out)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The block stack over packed frames, through the bound key-prefix resource.
        Shape-only in the frame count."""
        if self.blocks[0].self_attn.ragged is None:
            raise RuntimeError(
                "Step-Audio2's audio encoder has no attention resource bound; it needs a "
                "RaggedKeyPrefixAttentionSpec (a key padding mask), which unbound SDPA would drop"
            )
        for block in self.blocks:
            # cu_seqlens is only read without a bound resource
            hidden_states = block(hidden_states, cu_seqlens=None)
        return hidden_states

    def pool(self, hidden_states: torch.Tensor, layout: list[ClipLayout]) -> torch.Tensor:
        """``AvgPool1d(2)`` per clip (dropping an odd last frame, as the reference's pool
        drops its batch's), then ``after_norm``."""
        pooled = [
            F.avg_pool1d(clip.transpose(0, 1)[None], 2, stride=2)[0].transpose(0, 1)
            for clip in hidden_states.split([c.frames for c in layout])
        ]
        return self.after_norm(torch.cat(pooled))


class Adapter(nn.Module):
    """``Adaptor``: conv (k3, s2, p1) + GELU, linear, ReLU, linear. Paths mirror ``adapter.*``."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        d = config.n_audio_state
        self.conv = nn.Conv1d(d, d, config.kernel_size, config.adapter_stride, padding=1)
        self.linear1 = nn.Linear(d, config.adapter_hidden)
        self.linear2 = nn.Linear(config.adapter_hidden, config.llm_dim)

    def forward(self, pooled: torch.Tensor, layout: list[ClipLayout], quantum: int = 1) -> torch.Tensor:
        """Packed pooled frames -> each clip's kept tokens, end to end ``[sum tokens, llm_dim]``.
        The conv pads each clip on its own, so it runs per clip. The linears run on all its
        outputs (at most one past the kept tokens) and, for one clip, on the conv's
        channel-major view with a batch axis, as the reference's do: the same GEMM shape and
        operand route (a 2D input of that layout takes another) keeps a one-clip request
        bit-identical to it. ``quantum`` pads each clip's conv input as ``frontend``'s does;
        at a padded width the conv's outputs are the same values, though cuDNN may then take
        another algorithm, so the served path is equivalent rather than bit-identical."""
        convolved = []
        for clip_frames in pooled.split([c.frames // 2 for c in layout]):
            n = clip_frames.shape[0]
            x = F.pad(clip_frames.transpose(0, 1)[None], (0, _padded(n, quantum) - n))
            # the outputs of the unpadded input (stride 2, padding 1)
            convolved.append(F.gelu(self.conv(x))[0, :, : (n - 1) // 2 + 1].transpose(0, 1))
        rows = convolved[0] if len(convolved) == 1 else torch.cat(convolved)
        out = self.linear2(F.relu(self.linear1(rows[None])))[0]
        if all(c.tokens == x.shape[0] for c, x in zip(layout, convolved, strict=True)):
            return out
        return torch.cat([
            rows[: c.tokens] for rows, c in zip(out.split([x.shape[0] for x in convolved]), layout, strict=True)
        ])


class StepAudio2Audio(nn.Module):
    """Mel clips -> LLM embeddings. ``attn_key`` names the encoder's key-prefix attention."""

    def __init__(self, config: AudioEncoderConfig, attn_key: str | None):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config, attn_key)
        self.adapter = Adapter(config)

    def zero_missing_biases(self) -> None:
        """The reference's ``key`` projection has no bias (``modeling_step_audio_2.py:175``),
        so the checkpoint fills only the q and v slices of each fused ``qkv_proj`` bias."""
        for block in self.encoder.blocks:
            block.self_attn.qkv_proj.bias.data.zero_()

    def forward(self, mels: list[torch.Tensor], layout: list[ClipLayout], quantum: int = 1) -> torch.Tensor:
        """A step's clips (each ``[n_mels, M]``, any number of requests, with ``layout`` from
        ``request_layout`` per request) -> ``[sum tokens, llm_dim]``, clips end to end."""
        hidden = self.encoder.encode(self.encoder.frontend(mels, layout, quantum))
        return self.adapter(self.encoder.pool(hidden, layout), layout, quantum)
