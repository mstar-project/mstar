"""Token2wav's flow: 25 Hz s3 speech tokens -> 50 Hz 80-bin mel, chunk by chunk.

Ported from Step-Audio2's ``CausalMaskedDiffWithXvec`` (``cosyvoice2/flow/flow.py``),
``UpsampleConformerEncoderV2`` (``cosyvoice2/transformer/``), ``CausalConditionalCFM``
(``flow_matching.py``) and ``DiT`` (``decoder_dit.py``), shipped in ``minicpmo-utils``
(Apache-2.0); only the streaming ``*_chunk`` paths exist here.

The reference keeps its streaming caches in buffers shared by every caller. Here each call
takes the request's cache tensors and a valid length, reads ``[:length]`` and writes the new
entries in place, so a request's whole state can live in one fixed-size slot. Ops, operand
layouts and concatenation orders follow the reference so the result is bit-identical: the
conformer appends new keys after its cache, the DiT puts them in front of it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

TOKEN_VOCAB = 6561
TOKEN_DIM = 512
MEL_BINS = 80
SPK_DIM = 192
PRE_LOOKAHEAD = 3
UP_RATE = 2
ENC_HEADS = 8
ENC_HEAD_DIM = 64
ENC_BLOCKS = 6
ENC_UP_BLOCKS = 4
DIT_DEPTH = 16
DIT_HEADS = 8
DIT_HEAD_DIM = 64
DIT_HIDDEN = 512
CFG_RATE = 0.7
N_TIMESTEPS = 10
NOISE_FRAMES = 50 * 600


def rel_position_table(d_model: int, max_len: int = 5000) -> torch.Tensor:
    """ESPnet's relative sinusoid table ``[1, 2 * max_len - 1, d_model]``, positive offsets
    reversed then negative ones; built on the host in float32 as the reference does."""
    pe_pos = torch.zeros(max_len, d_model)
    pe_neg = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe_pos[:, 0::2] = torch.sin(position * div_term)
    pe_pos[:, 1::2] = torch.cos(position * div_term)
    pe_neg[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_neg[:, 1::2] = torch.cos(-1 * position * div_term)
    return torch.cat([torch.flip(pe_pos, [0]).unsqueeze(0), pe_neg[1:].unsqueeze(0)], dim=1)


# ---------------------------------------------------------------------------
# Token encoder
# ---------------------------------------------------------------------------


class LinearEmbed(nn.Module):
    """Linear + LayerNorm, scaled by ``sqrt(d)`` (``LinearNoSubsampling`` + its positional
    module, whose own table the chunked path never reads)."""

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(idim, odim), nn.LayerNorm(odim, eps=1e-5))
        self.xscale = math.sqrt(odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x) * self.xscale


class RelPositionAttention(nn.Module):
    """Transformer-XL relative-position attention with a key/value cache appended to."""

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
    def rel_shift(x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1).view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        return x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, kv_cache: torch.Tensor, n_cached: int,
    ) -> torch.Tensor:
        """``x [B, T, D]``; ``kv_cache [B, H, cap, 2 * d_k]`` holds ``n_cached`` entries and
        receives this chunk's keys/values at ``[n_cached, n_cached + T)``."""
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, -1, self.h, self.d_k)
        k = self.linear_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        kv_cache[:, :, n_cached:n_cached + t] = torch.cat((k, v), dim=-1)
        if n_cached > 0:
            key_cache, value_cache = torch.split(kv_cache[:, :, :n_cached], self.d_k, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(out)


class FeedForward(nn.Module):
    def __init__(self, idim: int, hidden: int):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden)
        self.activation = nn.SiLU()
        self.w_2 = nn.Linear(hidden, idim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class ConformerLayer(nn.Module):
    """Pre-LN attention + feed-forward (the conformer's macaron and conv branches are off)."""

    def __init__(self, size: int, n_head: int, linear_units: int):
        super().__init__()
        self.self_attn = RelPositionAttention(n_head, size)
        self.feed_forward = FeedForward(size, linear_units)
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, kv_cache: torch.Tensor, n_cached: int) -> torch.Tensor:
        x = x + self.self_attn(self.norm_mha(x), pos_emb, kv_cache, n_cached)
        return x + self.feed_forward(self.norm_ff(x))


class PreLookaheadLayer(nn.Module):
    """Sees ``PRE_LOOKAHEAD`` future tokens and emits that many fewer frames; its second
    conv carries 2 frames of left context in the cache."""

    def __init__(self, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3)

    def forward(self, inputs: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        """``inputs [B, T, C]``, ``cache [B, C, 2]`` (updated in place) -> ``[B, T - 3, C]``."""
        outputs = F.leaky_relu(self.conv1(inputs.transpose(1, 2).contiguous()))
        outputs = torch.cat([cache, outputs], dim=2)
        cache.copy_(outputs[..., -2:])
        outputs = self.conv2(outputs).transpose(1, 2).contiguous()
        return outputs + inputs[:, : -self.pre_lookahead_len]


class CausalUpsample(nn.Module):
    """Nearest x2 then a causal width-5 conv whose 4 frames of left context are cached."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, channels, stride * 2 + 1)

    def forward(self, inputs: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        """``inputs [B, C, T]``, ``cache [B, C, 4]`` (updated in place) -> ``[B, C, 2T]``."""
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = torch.cat([cache, outputs], dim=2)
        cache.copy_(outputs[..., -self.stride * 2:])
        return self.conv(outputs)


class StreamingTokenEncoder(nn.Module):
    """``UpsampleConformerEncoderV2.forward_chunk``: 6 conformer layers at 25 Hz, x2
    upsample, 4 more at 50 Hz. Parameter paths mirror ``flow.pt``'s ``encoder.*``."""

    def __init__(self):
        super().__init__()
        d = TOKEN_DIM
        self.embed = LinearEmbed(d, d)
        self.pre_lookahead_layer = PreLookaheadLayer(d, PRE_LOOKAHEAD)
        self.encoders = nn.ModuleList(ConformerLayer(d, ENC_HEADS, 2048) for _ in range(ENC_BLOCKS))
        self.up_layer = CausalUpsample(d, UP_RATE)
        self.up_embed = LinearEmbed(d, d)
        self.up_encoders = nn.ModuleList(ConformerLayer(d, ENC_HEADS, 2048) for _ in range(ENC_UP_BLOCKS))
        self.after_norm = nn.LayerNorm(d, eps=1e-5)
        self.register_buffer("pe", rel_position_table(d), persistent=False)

    def position_encoding(self, size: int) -> torch.Tensor:
        center = self.pe.size(1) // 2
        return self.pe[:, center - size + 1: center + size]

    def forward(
        self,
        xs: torch.Tensor,
        last_chunk: bool,
        cnn_cache: torch.Tensor,
        kv1: torch.Tensor,
        len1: int,
        kv2: torch.Tensor,
        len2: int,
    ) -> torch.Tensor:
        """Token embeddings ``[B, T, 512]`` -> ``[B, 2 (T - 3), 512]`` (``2T`` on the last
        chunk, whose lookahead is zero padding). ``cnn_cache [B, 512, 6]``; ``kv1 [6, B, H,
        cap1, 128]`` / ``kv2 [4, B, H, cap2, 128]`` hold ``len1`` / ``len2`` entries and
        receive this chunk's after them. The reference positions the 25 Hz stage at
        ``len2 // 2``, which equals ``len1`` because ``len2 == 2 * len1`` on entry."""
        assert len2 == UP_RATE * len1, (len1, len2)
        xs = self.embed(xs)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, PRE_LOOKAHEAD))
        xs = self.pre_lookahead_layer(xs, cnn_cache[:, :, :2])
        pos_emb = self.position_encoding(len1 + xs.shape[1])
        for idx, layer in enumerate(self.encoders):
            xs = layer(xs, pos_emb, kv1[idx], len1)

        xs = self.up_layer(xs.transpose(1, 2).contiguous(), cnn_cache[:, :, 2:]).transpose(1, 2).contiguous()
        xs = self.up_embed(xs)
        pos_emb = self.position_encoding(len1 * UP_RATE + xs.shape[1])
        for idx, layer in enumerate(self.up_encoders):
            xs = layer(xs, pos_emb, kv2[idx], len2)
        return self.after_norm(xs)


# ---------------------------------------------------------------------------
# DiT estimator
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
        )
        half = frequency_embedding_size // 2
        # built on the host like the reference, then moved with the module
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = (t * 1000)[:, None] * self.freqs.to(t)[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class DiTAttention(nn.Module):
    """QK-normed attention that puts this chunk's keys/values in front of the cached ones."""

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(inner, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.reshape(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv_cache: torch.Tensor, n_cached: int) -> torch.Tensor:
        """``kv_cache [B, H, cap, 2 * head_dim]`` holds ``n_cached`` entries; afterwards it
        holds ``[this chunk, old entries]`` (``n_cached + T``)."""
        b, t, _ = x.shape
        q = self.q_norm(self._heads(self.to_q(x)))
        k = self.k_norm(self._heads(self.to_k(x)))
        v = self._heads(self.to_v(x))
        if n_cached > 0:
            k_cache, v_cache = kv_cache[:, :, :n_cached].chunk(2, dim=3)
            k = torch.cat([k, k_cache], dim=2)
            v = torch.cat([v, v_cache], dim=2)
        kv_cache[:, :, : n_cached + t] = torch.cat([k, v], dim=3)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, t, -1))


class CausalConv(nn.Conv1d):
    """Width-3 causal conv whose 2 frames of left context are cached."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__(channels, channels, kernel_size)

    def step(self, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cache, x], dim=2)
        out = super().forward(x)
        cache.copy_(x[..., -cache.shape[-1]:])
        return out


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = CausalConv(channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.Mish()
        self.conv2 = CausalConv(channels)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        """``x [B, T, C]``, ``cache [B, 2C, 2]`` (both convs', updated in place)."""
        x = self.conv1.step(x.transpose(1, 2), cache[:, : self.channels])
        x = self.act(self.norm(x.transpose(1, 2))).transpose(1, 2)
        return self.conv2.step(x, cache[:, self.channels:]).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """adaLN-zero block: attention, causal conv, MLP, each gated by the time embedding."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden, int(hidden * mlp_ratio))
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.conv = CausalConvBlock(hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, cnn_cache: torch.Tensor, kv_cache: torch.Tensor, n_cached: int,
    ) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), kv_cache, n_cached)
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Velocity estimator over ``[x, mu, spk, cond]`` (4 x 80 channels)."""

    def __init__(self):
        super().__init__()
        self.t_embedder = TimestepEmbedder(DIT_HIDDEN)
        self.in_proj = nn.Linear(4 * MEL_BINS, DIT_HIDDEN)
        self.blocks = nn.ModuleList(DiTBlock(DIT_HIDDEN, DIT_HEADS, DIT_HEAD_DIM) for _ in range(DIT_DEPTH))
        self.final_layer = FinalLayer(DIT_HIDDEN, MEL_BINS)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        n_cached: int,
    ) -> torch.Tensor:
        """``x, mu, cond [B, 80, T]``, ``t [B]``, ``spks [B, 80]``; ``cnn_cache [depth, B,
        1024, 2]`` and ``kv_cache [depth, B, H, cap, 128]`` are this Euler step's."""
        c = self.t_embedder(t).unsqueeze(1)
        x = torch.cat([x, mu, spks.unsqueeze(-1).expand(-1, -1, x.shape[-1]), cond], dim=1)
        x = self.in_proj(x.transpose(1, 2))
        for i, block in enumerate(self.blocks):
            x = block(x, c, cnn_cache[i], kv_cache[i], n_cached)
        return self.final_layer(x, c).transpose(1, 2)


class ChunkCFM(nn.Module):
    """Cosine-scheduled Euler solver with classifier-free guidance; each Euler step has its
    own estimator caches. The initial noise for frames ``[offset, offset + T)`` is a slice of
    a fixed ``[1, 80, 30000]`` buffer (the reference draws it once at construction), offset
    by the request's cache length."""

    def __init__(self):
        super().__init__()
        self.estimator = DiT()
        self.register_buffer("rand_noise", torch.zeros(1, MEL_BINS, NOISE_FRAMES), persistent=False)

    def t_span(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0, 1, N_TIMESTEPS + 1, device=device, dtype=dtype)
        return 1 - torch.cos(t * 0.5 * torch.pi)

    def forward(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        n_cached: int,
    ) -> torch.Tensor:
        """``mu, cond [1, 80, T]``, ``spks [1, 80]``; ``cnn_cache [steps, depth, 2, 1024, 2]``
        and ``kv_cache [steps, depth, 2, H, cap, 128]`` (batch 2 = conditional, unconditional)."""
        x = self.rand_noise[:, :, n_cached:n_cached + mu.size(2)] * 1.0
        t_span = self.t_span(mu.device, mu.dtype)
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(
                x.repeat(2, 1, 1), mu_in, t.repeat(2), spks_in, cond_in,
                cnn_cache[step - 1], kv_cache[step - 1], n_cached,
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + CFG_RATE) * dphi_dt - CFG_RATE * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x


class Token2WavFlow(nn.Module):
    """Tokens + speaker embedding -> mel chunk. Parameter paths mirror ``flow.pt``."""

    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(TOKEN_VOCAB, TOKEN_DIM)
        self.spk_embed_affine_layer = nn.Linear(SPK_DIM, MEL_BINS)
        self.encoder = StreamingTokenEncoder()
        self.encoder_proj = nn.Linear(TOKEN_DIM, MEL_BINS)
        self.decoder = ChunkCFM()

    def project_speaker(self, spk_emb: torch.Tensor) -> torch.Tensor:
        """Raw CAMPPlus x-vector ``[1, 192]`` -> the estimator's ``[1, 80]`` condition."""
        return self.spk_embed_affine_layer(F.normalize(spk_emb, dim=1))

    def forward(
        self,
        tokens: torch.Tensor,
        spk: torch.Tensor,
        cond: torch.Tensor | None,
        last_chunk: bool,
        enc_cnn: torch.Tensor,
        enc_kv1: torch.Tensor,
        enc_len1: int,
        enc_kv2: torch.Tensor,
        enc_len2: int,
        dit_cnn: torch.Tensor,
        dit_kv: torch.Tensor,
        dit_len: int,
    ) -> torch.Tensor:
        """``tokens [1, T]`` (int) -> mel ``[1, 80, 2 (T - 3)]`` (``2T`` when ``last_chunk``);
        ``spk`` is ``project_speaker``'s output, ``cond`` the prompt mel ``[1, 80, frames]``
        when building a voice's cache, else zeros. Writes every cache in place; the caller
        advances the lengths."""
        h = self.encoder(self.input_embedding(tokens), last_chunk, enc_cnn, enc_kv1, enc_len1, enc_kv2, enc_len2)
        h = self.encoder_proj(h)
        if cond is None:
            cond = torch.zeros_like(h).transpose(1, 2).contiguous()
        return self.decoder(h.transpose(1, 2).contiguous(), spk, cond, dit_cnn, dit_kv, dit_len)
