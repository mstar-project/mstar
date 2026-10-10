"""Token2wav's flow: 25 Hz s3 speech tokens -> 50 Hz 80-bin mel, chunk by chunk.

Ported from Step-Audio2's ``CausalMaskedDiffWithXvec`` (``cosyvoice2/flow/flow.py``),
``UpsampleConformerEncoderV2`` (``cosyvoice2/transformer/``), ``CausalConditionalCFM``
(``flow_matching.py``) and ``DiT`` (``decoder_dit.py``), shipped in ``minicpmo-utils``
(Apache-2.0); only the streaming ``*_chunk`` paths exist here.

The reference keeps its streaming caches in buffers shared by every caller. Here every
attention layer is handed a ``KVCache`` saying where its keys|values live: ``DenseKV`` over a
request's own cache tensor (any window, one request; ops, operand layouts and concatenation
orders follow the reference so the result is bit-identical), ``PoolKV`` straight over the rows
of a slot pool (the conformer's caches for a batch of windows, gathered and scattered by slot
index so it can be captured), or ``RingKV`` (the DiT's caches on the pool, updated in place).

Classifier-free guidance runs the conditional and unconditional rows of a request side by side
(rows ``2b`` and ``2b + 1``), which is also how the DiT caches store them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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


class KVCache:
    """One attention layer's keys|values (``[..., 2 * head_dim]``) between windows."""

    length: int  # valid entries before this window

    def read(self) -> torch.Tensor | None:
        """The valid entries ``[N, H, length, 2d]``, or None when there are none."""
        raise NotImplementedError

    def write(self, k: torch.Tensor, v: torch.Tensor, new_first: bool) -> None:
        """Persist after attention; ``k``/``v`` ``[N, H, length + T, d]`` are what the layer
        attended to, this window's entries first (``new_first``) or last."""
        raise NotImplementedError


class DenseKV(KVCache):
    """A request's own cache tensor ``[N, H, capacity, 2d]`` holding ``length`` entries;
    afterwards it holds ``length + T`` in attention order (the caller truncates)."""

    def __init__(self, buf: torch.Tensor, length: int):
        self.buf = buf
        self.length = length

    def read(self) -> torch.Tensor | None:
        return self.buf[:, :, : self.length] if self.length > 0 else None

    def write(self, k: torch.Tensor, v: torch.Tensor, new_first: bool) -> None:
        if new_first:
            self.buf[:, :, : k.shape[2]] = torch.cat([k, v], dim=3)
        else:
            n = self.length
            self.buf[:, :, n:k.shape[2]] = torch.cat((k[:, :, n:], v[:, :, n:]), dim=-1)


class PoolKV(KVCache):
    """Rows of slot-major pool views ``[slots, *rows, H, capacity, 2d]``. Reads gather
    ``[:length]`` of ``src`` at ``src_slots`` (a voice's initial state as a one-slot pool, say),
    flattening ``rows`` per slot into the batch (``[B * prod(rows), H, length, 2d]``). After
    attention each ``(keep, dst)`` in ``writes`` scatters the attended entries ``keep`` to
    ``[dst, dst + len(keep))`` of ``view`` at ``slots``: that is how a fixed-length window
    lands its cache, truncation included, without a separate pass."""

    def __init__(
        self,
        view: torch.Tensor,
        slots: torch.Tensor,
        length: int,
        writes: list[tuple[slice, int]],
        src: torch.Tensor | None = None,
        src_slots: torch.Tensor | None = None,
    ):
        self.view = view
        self.slots = slots
        self.length = length
        self.writes = writes
        self.src = view if src is None else src
        self.src_slots = slots if src_slots is None else src_slots

    def read(self) -> torch.Tensor | None:
        if self.length == 0:
            return None
        rows = self.src[..., : self.length, :].index_select(0, self.src_slots)
        return rows.flatten(0, rows.dim() - 4)

    def write(self, k: torch.Tensor, v: torch.Tensor, new_first: bool) -> None:
        del new_first
        rows = (self.slots.shape[0], *self.view.shape[1:-3])
        d = k.shape[-1]
        for keep, dst in self.writes:
            # keys and values straight into their halves of the slot, without a joined copy
            for half, part in ((slice(0, d), k), (slice(d, 2 * d), v)):
                part = part[:, :, keep].unflatten(0, rows)
                self.view[..., dst:dst + part.shape[-2], half].index_copy_(0, self.slots, part)


class RingKV(KVCache):
    """A DiT layer's cache for a batch of windows on the slot pool, in place.

    The reference attends ``[new, old]`` and then keeps the first ``2P`` entries and a fixed
    ``2P..2P+100`` tail. Followed through the windows, the head is the newest generated
    frames then a shrinking prefix of the voice prompt's, dropping from its end, and the rest
    is always a slice of the voice's own cache: ``voice[2P - n : 2P]`` with ``n = length -
    2P`` (0, 50, then 100). DiT attention has no mask and no positions, so only that multiset
    matters, not its order: the head lives in a ring of ``2P`` frames per slot (``view [S, 2,
    H, >= 2P, 2d]``) whose newest chunk starts at the slot's ``head`` index, and the tail is
    read from the voice (``voice [2, H, >= 2P, 2d]``). A window writes only its own frames, at
    ``[head - T, head)`` mod ``2P``, which is exactly what the reference drops; the caller
    moves ``head`` once per window (``advance_heads``). A request's first window reads the
    voice as its ring (``fresh``) and copies it into the slot before writing."""

    def __init__(
        self,
        view: torch.Tensor,
        heads: torch.Tensor,
        slots: torch.Tensor,
        voice: torch.Tensor,
        prompt_frames: int,
        length: int,
        fresh: bool,
        write: bool,
    ):
        self.view = view
        self.heads = heads
        self.slots = slots
        self.voice = voice
        self.p2 = prompt_frames
        self.length = length
        self.fresh = fresh
        self.write_back = write

    def keys_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The cached entries ``[2B, H, 2P + n, 2d]``: the ring (or the voice, fresh) then the
        voice's tail."""
        b, p2 = self.slots.shape[0], self.p2
        if self.fresh:
            ring = self.voice[None, :, :, :p2].expand(b, -1, -1, -1, -1)
        else:
            ring = self.view[:, :, :, :p2].index_select(0, self.slots)
        tail = self.voice[None, :, :, p2 - (self.length - p2):p2].expand(b, -1, -1, -1, -1)
        return torch.cat([ring, tail], dim=3).flatten(0, 1).chunk(2, dim=-1)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The CUDA path: attention straight over the ring and the voice (Triton), then this
        window's ``k, v`` into the ring. Returns ``[2B, T, H, d]``."""
        from mstar.model.minicpm_o.components.token2wav_kernels import ring_attention, ring_store

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = ring_attention(q, k, v, self.view, self.slots, self.voice, self.p2,
                             self.length - self.p2, self.fresh)
        if self.write_back:
            if self.fresh:
                b = self.slots.shape[0]
                self.view[:, :, :, :self.p2].index_copy_(
                    0, self.slots, self.voice[None, :, :, :self.p2].expand(b, -1, -1, -1, -1))
            ring_store(k, v, self.view, self.slots, self.heads, self.p2, self.fresh)
        return out

    def ring_positions(self, frames: int) -> torch.Tensor:
        """``[B, frames]`` ring indices this window's frames go to."""
        head = torch.zeros_like(self.slots) if self.fresh else self.heads.index_select(0, self.slots).long()
        offsets = torch.arange(frames, device=self.slots.device) - frames
        return (head[:, None] + offsets[None, :]) % self.p2

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """This window's ``k, v [2B, H, T, d]`` into the ring."""
        if not self.write_back:
            return
        b = self.slots.shape[0]
        if self.fresh:
            self.view[:, :, :, :self.p2].index_copy_(
                0, self.slots, self.voice[None, :, :, :self.p2].expand(b, -1, -1, -1, -1))
        new = torch.cat([k, v], dim=-1).unflatten(0, (b, 2))  # [B, 2, H, T, 2d]
        pos = self.ring_positions(new.shape[3])
        self.view[self.slots[:, None], :, :, pos] = new.permute(0, 3, 1, 2, 4)

    @staticmethod
    def advance_heads(heads: torch.Tensor, slots: torch.Tensor, frames: int, prompt_frames: int,
                      fresh: bool) -> None:
        """After a window's last layer: the newest chunk now starts ``frames`` earlier."""
        head = torch.zeros_like(slots) if fresh else heads.index_select(0, slots).long()
        heads.index_copy_(0, slots, ((head - frames) % prompt_frames).to(heads.dtype))


class BoundedLayerKV(KVCache):
    """A DiT layer's cache on the engine's bounded KV resource: ``plan`` is this
    batch's rows of the step's layout, ``source [2, H, 2P, 2d]`` the layer's voice in
    stream order (``token2wav.dit_source``). ``length`` is how many entries the rows
    retain, as the flow's noise offset."""

    def __init__(self, resource, layer_idx: int, source: torch.Tensor, plan, length: int):
        self.length = length
        self.resource = resource
        self.layer_idx = layer_idx
        self.source = source
        self.plan = plan

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.resource.attend(self.layer_idx, q, k, v, self.source, self.plan)


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

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, kv: KVCache) -> torch.Tensor:
        """``x [B, T, D]``; this chunk's keys/values go after the cached ones."""
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, -1, self.h, self.d_k)
        k = self.linear_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        cached = kv.read()
        if cached is not None:
            key_cache, value_cache = torch.split(cached, self.d_k, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        kv.write(k, v, new_first=False)

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

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, kv: KVCache) -> torch.Tensor:
        x = x + self.self_attn(self.norm_mha(x), pos_emb, kv)
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
        kv1: list[KVCache],
        kv2: list[KVCache],
    ) -> torch.Tensor:
        """Token embeddings ``[B, T, 512]`` -> ``[B, 2 (T - 3), 512]`` (``2T`` on the last
        chunk, whose lookahead is zero padding). ``cnn_cache [B, 512, 6]`` (updated in place);
        ``kv1`` / ``kv2``: one cache per 25 Hz / 50 Hz layer. The reference positions the 25 Hz
        stage at ``len2 // 2``, which equals ``len1`` because ``len2 == 2 * len1`` on entry."""
        len1, len2 = kv1[0].length, kv2[0].length
        assert len2 == UP_RATE * len1, (len1, len2)
        xs = self.embed(xs)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, PRE_LOOKAHEAD))
        xs = self.pre_lookahead_layer(xs, cnn_cache[:, :, :2])
        pos_emb = self.position_encoding(len1 + xs.shape[1])
        for idx, layer in enumerate(self.encoders):
            xs = layer(xs, pos_emb, kv1[idx])

        xs = self.up_layer(xs.transpose(1, 2).contiguous(), cnn_cache[:, :, 2:]).transpose(1, 2).contiguous()
        xs = self.up_embed(xs)
        pos_emb = self.position_encoding(len1 * UP_RATE + xs.shape[1])
        for idx, layer in enumerate(self.up_encoders):
            xs = layer(xs, pos_emb, kv2[idx])
        return self.after_norm(xs)


# ---------------------------------------------------------------------------
# DiT estimator
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale_plus_1: torch.Tensor) -> torch.Tensor:
    """adaLN's ``x * (1 + scale) + shift``, with ``1 + scale`` precomputed (same arithmetic)."""
    return x * scale_plus_1 + shift


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

    def forward(self, x: torch.Tensor, kv: KVCache) -> torch.Tensor:
        """This chunk's keys/values go in front of the cached ones."""
        b, t, _ = x.shape
        q = self.q_norm(self._heads(self.to_q(x)))
        k = self.k_norm(self._heads(self.to_k(x)))
        v = self._heads(self.to_v(x))
        if isinstance(kv, BoundedLayerKV):
            return self.proj(kv.attend(q, k, v).reshape(b, t, -1))
        if isinstance(kv, RingKV):
            if q.is_cuda:
                x = kv.attend(q, k, v)  # [b, t, H, d]
                return self.proj(x.reshape(b, t, -1))
            k_cache, v_cache = kv.keys_values()
            x = F.scaled_dot_product_attention(q, torch.cat([k, k_cache], dim=2), torch.cat([v, v_cache], dim=2))
            kv.store(k, v)
            return self.proj(x.transpose(1, 2).reshape(b, t, -1))
        cached = kv.read()
        if cached is not None:
            k_cache, v_cache = cached.chunk(2, dim=3)
            k = torch.cat([k, k_cache], dim=2)
            v = torch.cat([v, v_cache], dim=2)
        kv.write(k, v, new_first=True)
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

    def modulation(self, c: torch.Tensor) -> torch.Tensor:
        """The time condition's 9 modulation vectors ``[9, *c.shape]``, scales as ``1 + scale``."""
        mod = list(self.adaLN_modulation(c).chunk(9, dim=-1))
        for i in (1, 4, 7):
            mod[i] = 1 + mod[i]
        return torch.stack(mod)

    def forward(self, x: torch.Tensor, mod: torch.Tensor, cnn_cache: torch.Tensor, kv: KVCache) -> torch.Tensor:
        """``mod`` is ``modulation``'s output for this Euler step (rows broadcast over ``x``)."""
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = mod
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), kv)
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_channels)

    def modulation(self, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return torch.stack([shift, 1 + scale])

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = mod
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
        mods: torch.Tensor,
        final_mod: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        kv: list[KVCache],
    ) -> torch.Tensor:
        """``x, mu, cond [N, 80, T]``, ``spks [N, 80]``; ``mods [depth, 9, r, 1, 512]`` /
        ``final_mod [2, r, 1, 512]`` the step's time modulations (``ChunkCFM.constants``),
        ``cnn_cache [depth, N, 1024, 2]`` and ``kv`` (one per block) its caches."""
        if x.is_cuda and kv and isinstance(kv[0], (RingKV, BoundedLayerKV)):
            from mstar.model.minicpm_o.components.token2wav_kernels import dit_forward

            return dit_forward(self, x, mu, mods, final_mod, spks, cond, cnn_cache, kv)
        x = torch.cat([x, mu, spks.unsqueeze(-1).expand(-1, -1, x.shape[-1]), cond], dim=1)
        x = self.in_proj(x.transpose(1, 2))
        for i, block in enumerate(self.blocks):
            x = block(x, mods[i], cnn_cache[i], kv[i])
        return self.final_layer(x, final_mod).transpose(1, 2)


@dataclass(frozen=True)
class SolverConstants:
    mods: torch.Tensor  # [steps, depth, 9, r, 1, 512], r = 1 (rows equal) or 2
    finals: torch.Tensor  # [steps, 2, r, 1, 512]
    dts: torch.Tensor  # [steps]


class ChunkCFM(nn.Module):
    """Cosine-scheduled Euler solver with classifier-free guidance; each Euler step has its
    own estimator caches. The initial noise for frames ``[offset, offset + T)`` is a slice of
    a fixed ``[1, 80, 30000]`` buffer (the reference draws it once at construction), offset
    by the request's cache length."""

    def __init__(self):
        super().__init__()
        self.estimator = DiT()
        self.register_buffer("rand_noise", torch.zeros(1, MEL_BINS, NOISE_FRAMES), persistent=False)
        self._constants: SolverConstants | None = None

    def t_span(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0, 1, N_TIMESTEPS + 1, device=device, dtype=dtype)
        return 1 - torch.cos(t * 0.5 * torch.pi)

    def reset_constants(self) -> None:
        """Drop the cached ``constants``; call after changing weights, device or precision."""
        self._constants = None

    @torch.inference_mode()
    def constants(self, device: torch.device, dtype: torch.dtype) -> SolverConstants:
        """Everything the solver computes from the time schedule alone, built once with the
        reference's own ops (the time embedding on the guidance pair, each block's adaLN
        projection, the step sizes as it accumulates them), so using them is bit-identical
        to recomputing them every window. The pair's two rows are equal, so one is kept and
        broadcast over any batch."""
        if self._constants is not None and self._constants.dts.device == device:
            return self._constants
        dit = self.estimator
        t_span = self.t_span(device, dtype)
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        mods, finals, dts = [], [], []
        for step in range(1, len(t_span)):
            c = dit.t_embedder(t.repeat(2)).unsqueeze(1)
            mods.append(torch.stack([block.modulation(c) for block in dit.blocks]))
            finals.append(dit.final_layer.modulation(c))
            dts.append(dt.reshape(()))
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        mods, finals = torch.stack(mods), torch.stack(finals)
        first = (..., slice(0, 1), slice(None), slice(None))
        second = (..., slice(1, 2), slice(None), slice(None))
        if torch.equal(mods[first], mods[second]) and torch.equal(finals[first], finals[second]):
            mods, finals = mods[first].contiguous(), finals[first].contiguous()
        self._constants = SolverConstants(mods=mods, finals=finals, dts=torch.stack(dts))
        return self._constants

    def forward(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        kv: list[list[KVCache]],
    ) -> torch.Tensor:
        """``mu, cond [B, 80, T]``, ``spks [B, 80]``; ``cnn_cache [steps, depth, 2B, 1024, 2]``
        and ``kv`` (per step, per block) over the ``2B`` guidance rows. Every row has the same
        cache length, which is also the noise offset."""
        b, n_cached = mu.shape[0], kv[0][0].length
        x = self.rand_noise[:, :, n_cached:n_cached + mu.size(2)].expand(b, -1, -1) * 1.0
        consts = self.constants(mu.device, mu.dtype)
        mods, finals = consts.mods, consts.finals
        if mods.shape[-3] != 1 and b != 1:
            # guidance rows are interleaved per request
            mods = mods.repeat(*([1] * (mods.dim() - 3)), b, 1, 1)
            finals = finals.repeat(*([1] * (finals.dim() - 3)), b, 1, 1)
        mu_in = torch.stack([mu, torch.zeros_like(mu)], dim=1).flatten(0, 1)
        spks_in = torch.stack([spks, torch.zeros_like(spks)], dim=1).flatten(0, 1)
        cond_in = torch.stack([cond, torch.zeros_like(cond)], dim=1).flatten(0, 1)
        for step in range(N_TIMESTEPS):
            dphi_dt = self.estimator(
                x.repeat_interleave(2, dim=0), mu_in, mods[step], finals[step], spks_in, cond_in,
                cnn_cache[step], kv[step],
            ).unflatten(0, (b, 2))
            dphi_dt, cfg_dphi_dt = dphi_dt[:, 0], dphi_dt[:, 1]
            dphi_dt = (1.0 + CFG_RATE) * dphi_dt - CFG_RATE * cfg_dphi_dt
            x = x + consts.dts[step] * dphi_dt
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
        enc_kv1: list[KVCache],
        enc_kv2: list[KVCache],
        dit_cnn: torch.Tensor,
        dit_kv: list[list[KVCache]],
    ) -> torch.Tensor:
        """``tokens [B, T]`` (int) -> mel ``[B, 80, 2 (T - 3)]`` (``2T`` when ``last_chunk``);
        ``spk [B, 80]`` is ``project_speaker``'s output, ``cond`` the prompt mel ``[B, 80,
        frames]`` when building a voice's cache, else zeros. ``enc_cnn [B, 512, 6]`` and
        ``dit_cnn [steps, depth, 2B, 1024, 2]`` are updated in place, the attention caches
        through their ``KVCache``; the caller advances the lengths."""
        h = self.encoder(self.input_embedding(tokens), last_chunk, enc_cnn, enc_kv1, enc_kv2)
        h = self.encoder_proj(h)
        if cond is None:
            cond = torch.zeros_like(h).transpose(1, 2).contiguous()
        return self.decoder(h.transpose(1, 2).contiguous(), spk, cond, dit_cnn, dit_kv)
