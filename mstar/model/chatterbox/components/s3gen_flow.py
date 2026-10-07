"""S3Gen's token encoder: speech tokens (25 Hz) -> flow-matching condition ``mu`` (50 Hz).

Reference: ``chatterbox/models/s3gen/flow.py`` (``CausalMaskedDiffWithXvec``)
and ``chatterbox/models/s3gen/transformer/{upsample_encoder,attention,
encoder_layer,embedding,subsampling,positionwise_feed_forward}.py`` (WeNet /
ESPnet conformer pieces as configured by CosyVoice 2: relative-position
self-attention, no convolution module, no macaron FFN).

Attention is full within the padded sequence (the reference runs with a
static chunk size of 0), so a chunked streaming caller must feed the encoder
the context it wants attended to.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.config import S3GenConfig, S3GenEncoderConfig


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``[B, max_len]`` bool, True where ``t < lengths[b]``."""
    positions = torch.arange(max_len, device=lengths.device)
    return positions[None, :] < lengths[:, None].long()


class RelPositionalEncoding(nn.Module):
    """ESPnet relative positional encoding: scales the input by ``sqrt(d)`` and
    returns the ``2T-1`` relative-position embeddings for offsets ``T-1 .. -(T-1)``."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe_positive = torch.zeros(max_len, d_model)
        pe_negative = torch.zeros(max_len, d_model)
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe = torch.cat([torch.flip(pe_positive, [0]), pe_negative[1:]], dim=0).unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = x.size(1)
        if size > self.max_len:
            raise ValueError(f"sequence of {size} frames exceeds the positional table ({self.max_len})")
        centre = self.pe.size(1) // 2
        pos_emb = self.pe[:, centre - size + 1 : centre + size]
        return x * self.xscale, pos_emb


class LinearEmbed(nn.Module):
    """Linear + LayerNorm input projection paired with a positional encoding."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, output_size), nn.LayerNorm(output_size, eps=1e-5), nn.Dropout(0.0),
        )
        self.pos_enc = RelPositionalEncoding(output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos_enc(self.out(x))


class RelPositionAttention(nn.Module):
    """Transformer-XL style multi-head attention with learned position biases."""

    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_head, self.d_k))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, t, n = x.shape
        zero_pad = x.new_zeros(b, h, t, 1)
        x_padded = torch.cat([zero_pad, x], dim=-1).view(b, h, n + 1, t)
        return x_padded[:, :, 1:].view(b, h, t, n)[:, :, :, : n // 2 + 1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, D]``; ``mask``: ``[B, 1, T]`` bool (True = attend);
        ``pos_emb``: ``[1, 2T-1, D]``."""
        b = x.size(0)
        q = self.linear_q(x).view(b, -1, self.h, self.d_k)
        k = self.linear_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self._rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        pad = mask.unsqueeze(1).eq(0)[:, :, :, : scores.size(-1)]
        scores = scores.masked_fill(pad, -float("inf"))
        attn = torch.softmax(scores, dim=-1).masked_fill(pad, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(out)


class FeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_1(x)))


class EncoderLayer(nn.Module):
    """Pre-norm block: relative-position attention, then the FFN."""

    def __init__(self, size: int, n_head: int, linear_units: int, eps: float):
        super().__init__()
        self.self_attn = RelPositionAttention(n_head, size)
        self.feed_forward = FeedForward(size, linear_units)
        self.norm_ff = nn.LayerNorm(size, eps=eps)
        self.norm_mha = nn.LayerNorm(size, eps=eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm_mha(x), mask, pos_emb)
        return x + self.feed_forward(self.norm_ff(x))


class PreLookaheadLayer(nn.Module):
    """Two convolutions that let each frame peek ``pre_lookahead_len`` frames ahead."""

    def __init__(self, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        return outputs.transpose(1, 2).contiguous() + inputs


class CausalUpsample(nn.Module):
    """Nearest x``stride`` upsample followed by a left-padded convolution."""

    def __init__(self, channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        return self.conv(outputs)


class UpsampleConformerEncoder(nn.Module):
    """``num_blocks`` layers at the token rate, x2 upsample, ``num_up_blocks`` layers at the mel rate."""

    def __init__(self, config: S3GenEncoderConfig):
        super().__init__()
        self.config = config
        size = config.output_size
        self.embed = LinearEmbed(config.input_size, size)
        self.pre_lookahead_layer = PreLookaheadLayer(size, config.pre_lookahead_len)
        self.encoders = nn.ModuleList([
            EncoderLayer(size, config.attention_heads, config.linear_units, config.layer_norm_eps)
            for _ in range(config.num_blocks)
        ])
        self.up_layer = CausalUpsample(size, size, config.up_stride)
        self.up_embed = LinearEmbed(config.input_size, size)
        self.up_encoders = nn.ModuleList([
            EncoderLayer(size, config.attention_heads, config.linear_units, config.layer_norm_eps)
            for _ in range(config.num_up_blocks)
        ])
        self.after_norm = nn.LayerNorm(size, eps=1e-5)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, T, D]`` -> (``[B, stride*T, D]``, mask ``[B, 1, stride*T]`` bool)."""
        masks = lengths_to_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb = self.embed(xs)
        # The look-ahead convolution reads three frames to the right. Zero the
        # padding first so a request batched with longer ones sees the same
        # zeros beyond its end that it sees when run alone.
        xs = xs * masks.transpose(1, 2).to(xs.dtype)
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            xs = layer(xs, masks, pos_emb)

        xs = self.up_layer(xs.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        xs_lens = xs_lens * self.up_layer.stride
        masks = lengths_to_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb = self.up_embed(xs)
        for layer in self.up_encoders:
            xs = layer(xs, masks, pos_emb)
        return self.after_norm(xs), masks


class FlowTokenEncoder(nn.Module):
    """Speech tokens + x-vector -> (``mu`` ``[B, 80, 2T]``, ``mask`` ``[B, 1, 2T]``, ``spk`` ``[B, 80]``)."""

    def __init__(self, config: S3GenConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.encoder.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.encoder = UpsampleConformerEncoder(config.encoder)
        self.encoder_proj = nn.Linear(config.encoder.output_size, config.output_size)

    def project_speaker(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.spk_embed_affine_layer(F.normalize(embedding, dim=1))

    def forward(self, tokens: torch.Tensor, token_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = lengths_to_mask(token_lens, tokens.size(1)).unsqueeze(-1).to(self.input_embedding.weight.dtype)
        h = self.input_embedding(tokens.long()) * mask
        h, h_masks = self.encoder(h, token_lens)
        mu = self.encoder_proj(h).transpose(1, 2).contiguous()
        return mu, h_masks
