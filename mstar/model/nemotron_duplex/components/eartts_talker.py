"""EarTTS talker (``tts_model.tts_model.*``): Gemma3-text transformer that
autoregressively emits RVQ codec codes, plus a mixture-of-gaussians (MoG) head,
a subword-conditioning encoder, and audio/text gated fusion.

Ported to exact checkpoint parity (28-layer Gemma3 backbone, head_dim 72 with
QK-norm; a 1-layer subword encoder; the MoG head; and the bespoke embedding
tables). The three top-level ``tts_model.*`` buffers (control codes, codec
silence tokens, per-voice audio-prompt latents) load here too.

Every architecture dimension / hyperparameter is read from ``EarTTSConfig``
(populated from the checkpoint's ``config.json``) and threaded down into the
submodules — there are no hardcoded model hyperparameters here. The only bare
literals are fixed table cardinalities that are not part of ``EarTTSConfig``
(the char-embedding vocab, the flag-embedding categories, and the
tokenizer-sized flag buffers).

The teacher-forced forward (backbone + gated fusion + MoG head) and the
char-aware subword encoder are numerically verified in fp32 against the NeMo
``RVQEARTTSModel`` reference (transformers Gemma3TextModel / T5GemmaEncoderModel
backbones): cosine > 0.9999 on the backbone hidden state, the MoG head params
and the pooled subword embeddings. The stochastic code-sampling loop
(``generate_step``: Gumbel mixture pick + iterative RVQ unmasking) is not
implemented here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import RMSNorm
from mstar.model.components.mlp import GatedMLP
from mstar.model.nemotron_duplex.components._util import RawWeight, param
from mstar.model.nemotron_duplex.config import EarTTSConfig

# Fixed table cardinalities that are NOT model hyperparameters in EarTTSConfig
# (tokenizer / vocabulary sizes baked into the checkpoint's flag tables).
_CHAR_VOCAB = 257            # embed_tokens rows: 256 chars + 1 padding slot
_SPECIAL_FLAG_CATS = 3       # BOS/EOS flag categories (0=regular, 1=BOS, 2=EOS)
_CONT_FLAG_CATS = 2          # continuation flag categories (0=word-start, 1=cont)
_SPECIAL_FLAGS_LEN = 131072  # bos/eos flag buffer length (text vocab size)
_CONT_FLAGS_LEN = 131073     # continuation flag buffer length (text vocab + pad)
_AUDIO_PROMPT_FRAMES = 37     # per-voice cached audio-prompt latent length


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma3 RMSNorm computed in fp32: ``(x * rsqrt(mean(x^2)+eps)) * (1 + w)``.

    Done manually (not via the shared fused ``RMSNorm`` kernel, which runs in
    bf16) so an fp32 numeric comparison is exact.
    """
    dtype = x.dtype
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(dtype)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filter over the last dim: mask out the low-probability tail
    beyond cumulative ``top_p`` with ``-inf`` (keeps at least the top token).
    Mirrors HF ``TopPLogitsWarper`` used by the reference MoG sampler."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    remove = cum - sorted_logits.softmax(-1) > top_p          # keep tokens whose left-cum <= top_p
    remove[..., 0] = False                                     # always keep the top token
    mask = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, theta: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard rotary cos/sin over the full ``dim`` head, computed in fp32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]   # [T, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                  # [T, dim]
    return emb.cos(), emb.sin()


def _masking_rate(rate: torch.Tensor, exponent: float) -> torch.Tensor:
    """MaskGIT keep-rate -> masking-rate schedule: ``(1 - rate**e) ** (1/e)``
    (its own inverse). Matches NeMo ``get_masking_rate``."""
    return (1.0 - rate.pow(exponent)).pow(1.0 / exponent)


def build_char_vocab(tokenizer) -> dict[str, int]:
    """Character vocabulary derived from the subword tokenizer (port of NeMo
    ``build_vocabs._build_char_vocab``): every single-character token in
    ``tokenizer.tokenizer.vocab``, densely re-indexed in ascending original-id
    order. ``char_vocab[char] -> dense index in [0, num_chars)``."""
    vocab = tokenizer.tokenizer.vocab                       # {token_str: id}
    single = {s: i for s, i in vocab.items() if len(s) == 1}
    ordered = sorted(single.keys(), key=lambda s: single[s])
    return {c: i for i, c in enumerate(ordered)}


def subword_to_char_ids(tokenizer, char_vocab: dict[str, int]) -> tuple[dict[int, tuple[int, ...]], int]:
    """Map each subword id to the tuple of its in-vocab character ids (port of
    NeMo ``build_vocabs`` steps 2-3). Subwords with no representable characters
    are dropped; a padding subword id ``len(tokenizer.vocab)`` maps to the char
    padding id ``len(char_vocab)``. Returns ``(map, subword_padding_idx)``."""
    vocab = tokenizer.tokenizer.vocab
    s2c = {sid: tuple(char_vocab[c] for c in s if c in char_vocab) for s, sid in vocab.items()}
    s2c = {k: v for k, v in s2c.items() if v}
    pad_idx = len(tokenizer.vocab)
    s2c[pad_idx] = (len(char_vocab),)
    return s2c, pad_idx


class _Attn(nn.Module):
    """q/k/v/o projections (no bias); optional Gemma3 per-head QK-norm.

    All dims (hidden size, head count/dim, query scaling, norm eps) come from
    ``EarTTSConfig``.
    """

    def __init__(self, config: EarTTSConfig, qk_norm: bool):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.eps = config.rms_norm_eps
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        causal: bool = True,
        attn_bias: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ):
        """Multi-head self-attention. Gemma3-style: optional per-head QK-norm
        (applied before rope), rope on the full head dim, and query scaling by
        ``query_pre_attn_scalar ** -0.5``. ``causal`` adds a triangular mask
        (talker); ``attn_bias`` is an additive mask broadcastable to
        ``[b, nh, tq, tk]`` (encoder key padding).

        For autoregressive decoding, ``past_kv`` (rope-applied ``(k, v)`` from
        earlier positions) is concatenated with the new keys/values and, when
        ``return_kv`` is set, the updated cache is returned. rope on the new
        query/key uses their absolute positions via ``cos``/``sin``; the causal
        mask offsets so the new queries attend all cached keys plus themselves."""
        b, t, _ = x.shape
        nh, hd = self.num_heads, self.head_dim
        q = F.linear(x, self.q_proj.weight).view(b, t, nh, hd)
        k = F.linear(x, self.k_proj.weight).view(b, t, nh, hd)
        v = F.linear(x, self.v_proj.weight).view(b, t, nh, hd)
        if hasattr(self, "q_norm"):
            q = _gemma_rmsnorm(q, self.q_norm.weight, self.eps)
            k = _gemma_rmsnorm(k, self.k_norm.weight, self.eps)
        q = q.transpose(1, 2)   # [b, nh, t, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if cos is not None:
            cos_b = cos[None, None, :, :]
            sin_b = sin[None, None, :, :]
            q = q * cos_b + _rotate_half(q) * sin_b
            k = k * cos_b + _rotate_half(k) * sin_b
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        new_kv = (k, v) if return_kv else None
        t_tot = k.shape[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if causal:
            offset = t_tot - t   # absolute index of the first query position
            neg = torch.full((t, t_tot), float("-inf"), device=x.device, dtype=scores.dtype)
            scores = scores + torch.triu(neg, diagonal=offset + 1)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, nh * hd)
        out = F.linear(out, self.o_proj.weight)
        if return_kv:
            return out, new_kv
        return out


class TalkerLayer(nn.Module):
    """Gemma3 decoder layer: input/post-attn/pre-ff/post-ff norms + QK-norm attn + gated MLP."""

    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(h)
        self.self_attn = _Attn(config, qk_norm=True)
        self.post_attention_layernorm = RMSNorm(h)
        self.pre_feedforward_layernorm = RMSNorm(h)
        self.mlp = GatedMLP(h, config.intermediate_size, activation="gelu_tanh", bias=False)
        self.post_feedforward_layernorm = RMSNorm(h)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ):
        """Gemma3 decoder layer with its 4 RMSNorms (input / post-attn /
        pre-ff / post-ff), each ``(1 + w)`` in fp32. Optionally reads/updates a
        per-layer ``(k, v)`` cache for autoregressive decoding."""
        residual = x
        h = _gemma_rmsnorm(x, self.input_layernorm.weight, self.eps)
        attn_out = self.self_attn(h, cos, sin, causal=True, past_kv=past_kv, return_kv=return_kv)
        if return_kv:
            h, new_kv = attn_out
        else:
            h, new_kv = attn_out, None
        h = _gemma_rmsnorm(h, self.post_attention_layernorm.weight, self.eps)
        x = residual + h

        residual = x
        h = _gemma_rmsnorm(x, self.pre_feedforward_layernorm.weight, self.eps)
        h = self.mlp(h)
        h = _gemma_rmsnorm(h, self.post_feedforward_layernorm.weight, self.eps)
        x = residual + h
        if return_kv:
            return x, new_kv
        return x


class _EncoderLayer(nn.Module):
    """Subword-encoder layer: pre/post self-attn + pre/post ff norms (no QK-norm)."""

    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.eps = config.rms_norm_eps
        self.self_attn = _Attn(config, qk_norm=False)
        self.pre_self_attn_layernorm = RMSNorm(h)
        self.post_self_attn_layernorm = RMSNorm(h)
        self.pre_feedforward_layernorm = RMSNorm(h)
        self.post_feedforward_layernorm = RMSNorm(h)
        self.mlp = GatedMLP(h, config.intermediate_size, activation="gelu_tanh", bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """T5Gemma encoder layer: bidirectional attention (no QK-norm) with
        pre/post self-attn and pre/post ff RMSNorms."""
        residual = x
        h = _gemma_rmsnorm(x, self.pre_self_attn_layernorm.weight, self.eps)
        h = self.self_attn(h, cos, sin, causal=False, attn_bias=attn_bias)
        h = _gemma_rmsnorm(h, self.post_self_attn_layernorm.weight, self.eps)
        x = residual + h

        residual = x
        h = _gemma_rmsnorm(x, self.pre_feedforward_layernorm.weight, self.eps)
        h = self.mlp(h)
        h = _gemma_rmsnorm(h, self.post_feedforward_layernorm.weight, self.eps)
        x = residual + h
        return x


class _EncoderBackbone(nn.Module):
    def __init__(self, config: EarTTSConfig, n_layers: int = 1):
        super().__init__()
        self.encoder = _Encoder(config, n_layers)


class _Encoder(nn.Module):
    def __init__(self, config: EarTTSConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer(config) for _ in range(n_layers)])
        self.norm = RMSNorm(config.hidden_size)


class _BosEosEmb(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.special_emb = nn.Embedding(_SPECIAL_FLAG_CATS, hidden_size)
        self.special_flags = param(_SPECIAL_FLAGS_LEN, dtype=torch.int64)
        self.pad_tensor = param(dtype=torch.int64)


class _SubwordFlagEmb(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.cont_emb = nn.Embedding(_CONT_FLAG_CATS, hidden_size)
        self.is_continuation = param(_CONT_FLAGS_LEN, dtype=torch.int64)
        self.pad_tensor = param(dtype=torch.int64)


class EmbedSubword(nn.Module):
    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.hidden_size = h
        self.head_dim = config.head_dim
        self.rope_theta_local = config.rope_theta_local
        self.eps = config.rms_norm_eps
        self.backbone = _EncoderBackbone(config, n_layers=1)
        self.embed_tokens = nn.Embedding(_CHAR_VOCAB, h)
        self.proj_embedding = nn.Linear(h, h, bias=False)
        self.bos_eos_emb = _BosEosEmb(h)
        self.subword_flag_emb = _SubwordFlagEmb(h)

    def encode_chars(self, char_ids: torch.Tensor, char_lengths: torch.Tensor) -> torch.Tensor:
        """Char-aware subword pooling: embed char ids, run the T5Gemma encoder
        (bidirectional, rope ``rope_theta_local``) with a key-padding mask,
        mean-pool over valid chars, and project. ``char_ids`` [N, Lc],
        ``char_lengths`` [N] -> ``[N, H]``."""
        n, lc = char_ids.shape
        char_mask = torch.arange(lc, device=char_ids.device)[None, :] < char_lengths[:, None]  # [N, Lc]
        char_embeds = self.embed_tokens(char_ids)  # [N, Lc, H]

        cos, sin = _rope_cos_sin(
            torch.arange(lc, device=char_ids.device), self.rope_theta_local, self.head_dim
        )
        neg = torch.finfo(char_embeds.dtype).min
        attn_bias = torch.where(char_mask[:, None, None, :], 0.0, neg).to(char_embeds.dtype)  # [N,1,1,Lc]

        # T5Gemma encoder scales the input embeddings by sqrt(hidden_size).
        x = char_embeds * (self.hidden_size ** 0.5)
        for layer in self.backbone.encoder.layers:
            x = layer(x, cos, sin, attn_bias)
        x = _gemma_rmsnorm(x, self.backbone.encoder.norm.weight, self.eps)

        masked_sum = (x * char_mask.unsqueeze(-1)).sum(dim=1)
        mean_emb = masked_sum / char_lengths.unsqueeze(-1).clamp(min=1)
        return self.proj_embedding(mean_emb)

    def forward(
        self,
        char_ids: torch.Tensor,
        char_lengths: torch.Tensor,
        subword_ids: torch.Tensor,
        subword_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full subword conditioning: pool char embeddings into per-subword
        vectors, scatter to [B, T, H], then add the continuation-flag and
        BOS/EOS embeddings looked up from the checkpoint flag tables.

        ``char_ids``/``char_lengths`` cover the valid (masked) subwords in row
        order (the caller supplies the subword->char id mapping from the
        tokenizer). ``subword_ids``/``subword_mask`` are [B, T]."""
        out_emb = self.encode_chars(char_ids, char_lengths)  # [N, H]
        subword_embeds = torch.zeros(
            subword_ids.shape + (self.hidden_size,), device=subword_ids.device, dtype=out_emb.dtype
        )
        subword_embeds[subword_mask] = out_emb

        # continuation-flag embedding (index 0 forced to zero in the checkpoint)
        vocab = self.subword_flag_emb.cont_emb.num_embeddings
        pad = self.subword_flag_emb.is_continuation.numel() - 1
        safe = torch.where(subword_ids >= pad, torch.full_like(subword_ids, pad), subword_ids)
        cont_flags = self.subword_flag_emb.is_continuation[safe]
        subword_embeds = subword_embeds + self.subword_flag_emb.cont_emb(cont_flags.clamp_max(vocab - 1))

        # BOS/EOS embedding (index 0 = regular, forced to zero)
        pad2 = self.bos_eos_emb.special_flags.numel() - 1
        safe2 = torch.where(subword_ids >= pad2, torch.full_like(subword_ids, pad2), subword_ids)
        flags = self.bos_eos_emb.special_flags[safe2]
        subword_embeds = subword_embeds + self.bos_eos_emb.special_emb(flags)
        return subword_embeds


class GatedFusion(nn.Module):
    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.num_quantizers = config.num_quantizers
        self.eps = config.rms_norm_eps
        self.audio_proj = nn.Linear(h, h, bias=True)
        self.text_proj = nn.Linear(h, h, bias=True)
        self.final_norm = RMSNorm(h)
        self.gate = param(h)
        self.residual_scale = param()  # scalar

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Gated projected sum of audio + text, then RMSNorm.

        ``audio_emb`` is divided by the number of codebooks; the per-channel
        ``gate`` and scalar ``residual_scale`` are sigmoided in fp32.
        """
        dtype = audio_emb.dtype
        audio_emb = audio_emb / self.num_quantizers
        audio_h = F.linear(audio_emb, self.audio_proj.weight, self.audio_proj.bias)
        text_h = F.linear(text_emb, self.text_proj.weight, self.text_proj.bias)
        gate = torch.sigmoid(self.gate.float())
        res = torch.sigmoid(self.residual_scale.float())
        h = gate.to(dtype) * audio_h + (1.0 - gate).to(dtype) * text_h
        h = res.to(dtype) * h
        h = _gemma_rmsnorm(h.float(), self.final_norm.weight, self.eps).to(dtype)
        return h


class _MogMLP(nn.Module):
    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.mlp = GatedMLP(h, config.mog_intermediate_size, activation="gelu_tanh", bias=False)
        self.pre_norm = RMSNorm(h)
        self.post_norm = RMSNorm(h)


class MogHead(nn.Module):
    """Mixture-of-gaussians output head over the RVQ latent space.

    ``mlp_stack`` is ``mog_num_layers`` gated-MLP blocks followed by a bare
    scale vector (final index, ``mlp_stack.<n>.weight`` [H]).
    """

    def __init__(self, config: EarTTSConfig):
        super().__init__()
        h = config.hidden_size
        self.num_predictions = config.mog_num_predictions
        self.low_rank = config.mog_low_rank
        self.out_size = config.code_dim
        self.min_log_std = config.mog_min_log_std
        self.eps = config.mog_eps
        self.num_layers = config.mog_num_layers
        self.low_mat = param(self.num_predictions, self.out_size, self.low_rank)
        self.mlp_stack = nn.ModuleList(
            [_MogMLP(config) for _ in range(self.num_layers)] + [RawWeight(h)]
        )
        self.proj_else = nn.Linear(h, self.out_size, bias=False)
        self.proj_logits = nn.Linear(h, self.num_predictions, bias=False)
        self.proj_logs = nn.Linear(h, 1, bias=False)
        self.proj_mus = nn.Linear(h, self.num_predictions * self.low_rank, bias=False)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """``mlp_stack``: ``num_layers`` residual MLPLayers (pre_norm -> mlp ->
        post_norm) followed by a final RMSNorm."""
        for block in self.mlp_stack[: self.num_layers]:
            y = _gemma_rmsnorm(x, block.pre_norm.weight, self.eps)
            y = block.mlp(y)
            y = _gemma_rmsnorm(y, block.post_norm.weight, self.eps)
            x = x + y
        return _gemma_rmsnorm(x, self.mlp_stack[self.num_layers].weight, self.eps)

    def forward(self, x: torch.Tensor):
        """Deterministic (training-style) MoG head forward.

        Returns ``(logits, mus, mu_res, logs)``:
          * ``logits`` [b, t, num_predictions] mixture weights
          * ``mus``    [b, t, num_predictions, low_rank] low-rank means
          * ``mu_res`` [b, t, out_size] residual mean
          * ``logs``   [b, t, 1] log std, clamped to ``mog_min_log_std``
        """
        b, t, _ = x.shape
        x = self._trunk(x)
        logits = F.linear(x, self.proj_logits.weight)
        mus = F.linear(x, self.proj_mus.weight).view(b, t, self.num_predictions, self.low_rank)
        logs = F.linear(x, self.proj_logs.weight).clamp_min(self.min_log_std)
        mu_res = F.linear(x, self.proj_else.weight)
        return logits, mus, mu_res, logs

    def infer(
        self,
        x: torch.Tensor,
        temperature: float = 0.0,
        guidance_scale: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a mixture component and return its target latent mean + log-std.

        Port of the reference ``MogHead.infer``. When ``guidance_scale > 0`` the
        input ``x`` is the stacked ``[cond; uncond]`` batch; after the trunk it is
        combined as ``x_cond + guidance_scale * (x_cond - x_uncond)`` (classifier-
        free guidance). ``top_p`` applies nucleus filtering to the mixture logits.
        The component is drawn by Gumbel-max (stochastic) when ``top_p`` is set or
        ``temperature > 0``; otherwise greedy ``argmax``. The chosen component's
        low-rank mean is projected up through ``low_mat``, scaled by ``exp(logs)``
        and offset by ``mu_res``: returns ``(mu * exp(logs) + mu_res, logs)``."""
        x = self._trunk(x)
        if guidance_scale and guidance_scale > 0.0:
            x_cond, x_uncond = x.chunk(2, dim=0)
            x = x_cond + guidance_scale * (x_cond - x_uncond)
        b, t, _ = x.shape
        logits = F.linear(x, self.proj_logits.weight)                       # [b, t, n]
        if top_p is not None and 0.0 < top_p < 1.0:
            logits = _top_p_filter(logits, top_p)
        if (top_p is not None and top_p < 1.0) or (temperature and temperature > 0.0):
            temp = temperature if (temperature and temperature > 0.0) else 1.0
            logp = F.log_softmax(logits, dim=-1) / temp
            u = torch.rand(logp.shape, device=logp.device, dtype=logp.dtype, generator=generator)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            idx = (logp + gumbel).argmax(-1)                                # [b, t]
        else:
            idx = logits.argmax(-1)                                        # [b, t]

        mus = F.linear(x, self.proj_mus.weight).view(b, t, self.num_predictions, self.low_rank)
        sel = idx[..., None, None].expand(b, t, 1, self.low_rank)
        mu = mus.gather(2, sel).squeeze(2)                                  # [b, t, low_rank]

        low_mat_sel = self.low_mat[idx]                                    # [b, t, out_size, low_rank]
        mu = torch.matmul(low_mat_sel, mu.unsqueeze(-1)).squeeze(-1)        # [b, t, out_size]

        mu_res = F.linear(x, self.proj_else.weight)                        # [b, t, out_size]
        logs = F.linear(x, self.proj_logs.weight).clamp_min(self.min_log_std)  # [b, t, 1]
        return mu * torch.exp(logs) + mu_res, logs


class _Backbone(nn.Module):
    def __init__(self, config: EarTTSConfig):
        super().__init__()
        self.layers = nn.ModuleList([TalkerLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)


class EarTTSTalker(nn.Module):
    def __init__(self, config: EarTTSConfig, voices=("Aria",)):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.embed_code = nn.Linear(config.code_dim, h, bias=False)
        self.backbone = _Backbone(config)
        self.embed_subword = EmbedSubword(config)
        self.gated_fusion_audio_text = GatedFusion(config)
        self.mog_head = MogHead(config)
        # bespoke embeddings / tables
        self.bos_emb = param(h)
        self.null_emb = param(h)
        self.audio_prompt_projection_W = param(h, h)
        self.rvq_embs = param(config.num_quantizers, config.codebook_size, config.code_dim)
        # top-level tts_model.* buffers (loaded via the talker node)
        self._control_codes = param(3, dtype=torch.int64)
        self.codec_silence_tokens = param(config.num_quantizers, dtype=torch.int64)
        self.audio_prompt_latents = nn.ParameterDict(
            {v: param(1, _AUDIO_PROMPT_FRAMES, h) for v in voices}
        )

    def depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
        """Sum of per-codebook RVQ embeddings: ``code`` [b, t, num_quantizers]
        (long) -> [b, t, code_dim]. Index ``codebook_size`` maps to a zero row
        (a padded extra slot), so masked-out codebooks contribute nothing."""
        b, t, d = code.shape
        _, _, hdim = self.rvq_embs.shape
        embs = F.pad(self.rvq_embs, [0, 0, 0, 1])   # [d, codebook_size+1, code_dim]
        ret = code.new_zeros((b, t, hdim), dtype=self.rvq_embs.dtype)
        for i in range(d):
            ret = ret + F.embedding(code[..., i], embs[i])
        return ret

    def backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache: list | None = None,
        start_pos: int = 0,
        return_cache: bool = False,
    ):
        """Gemma3-text decoder stack over ``inputs_embeds`` [b, t, H].

        Full-attention layers (``(i+1) % sliding_window_pattern == 0``) use
        ``rope_theta_full``; the rest (sliding-attention) use
        ``rope_theta_local``. For sequences shorter than the sliding window the
        two mask identically (plain causal), so only the rope base differs.

        For autoregressive decoding, pass the per-layer ``kv_cache`` list and the
        absolute ``start_pos`` of the new tokens; with ``return_cache`` the
        updated per-layer ``(k, v)`` list is returned alongside the hidden state.
        """
        cfg = self.config
        b, t, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + t, device=inputs_embeds.device)
        cos_g, sin_g = _rope_cos_sin(position_ids, cfg.rope_theta_full, cfg.head_dim)
        cos_l, sin_l = _rope_cos_sin(position_ids, cfg.rope_theta_local, cfg.head_dim)
        new_cache = [] if return_cache else None
        x = inputs_embeds
        for i, layer in enumerate(self.backbone.layers):
            full = (i + 1) % cfg.sliding_window_pattern == 0
            cos, sin = (cos_g, sin_g) if full else (cos_l, sin_l)
            past = kv_cache[i] if kv_cache is not None else None
            out = layer(x, cos, sin, past_kv=past, return_kv=return_cache)
            if return_cache:
                x, kv = out
                new_cache.append(kv)
            else:
                x = out
        x = _gemma_rmsnorm(x, self.backbone.norm.weight, cfg.rms_norm_eps)
        if return_cache:
            return x, new_cache
        return x

    def _rvq_quantize(self, z: torch.Tensor, code: torch.Tensor, depth_str: int, k: int) -> torch.Tensor:
        """Residual vector quantization of latent ``z`` [b, t, code_dim] into
        codebooks ``[depth_str, depth_str + k)``. Faithful port of NeMo
        ``depthsum_encoding_step``: nearest RVQ entry per codebook (minimising
        ``||e||^2 - 2<r, e>``), subtracting the chosen entry from the running
        residual ``r`` (which starts at ``z``)."""
        code = code.clone()
        r = z
        for i in range(depth_str, depth_str + k):
            ei = self.rvq_embs[i]                                            # [codebook_size, code_dim]
            dist = ei.pow(2).sum(-1) - 2.0 * torch.matmul(r, ei.transpose(-1, -2))  # [b, t, codebook_size]
            idx = dist.argmin(-1)                                            # [b, t]
            r = r - F.embedding(idx, ei)
            code[..., i] = idx
        return code

    @torch.no_grad()
    def generate_step(
        self,
        hidden_states: torch.Tensor,
        hidden_uncond: torch.Tensor | None = None,
        num_iter: int = 8,
        exponent: float = 3.0,
        temperature: float = 0.0,
        guidance_scale: float = 0.0,
        noise_scale: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Iterative MoG-conditioned RVQ decode of one (or more) frame(s) — port
        of the reference ``generate_step``.

        Given the conditional backbone hidden state ``[b, t, H]`` (and, for
        classifier-free guidance, the unconditional ``hidden_uncond``), fills the
        ``num_quantizers`` RVQ codebooks over ``num_iter`` MaskGIT iterations.
        Each iteration re-embeds the current partial code, adds the hidden state,
        samples the MoG latent ``z = mu + exp(logs)*eps*noise_scale`` (mixture
        component drawn by Gumbel-top-p; guidance applied inside ``mog_head.infer``
        on the stacked ``[cond; uncond]`` batch), then residual-quantizes the next
        block of codebooks. Returns ``[b, t, num_quantizers]`` in ``[0, codebook_size)``.
        """
        cfg = self.config
        b, t, _ = hidden_states.shape
        d = cfg.num_quantizers
        device = hidden_states.device
        code = torch.full((b, t, d), cfg.codebook_size, dtype=torch.long, device=device)
        guided = guidance_scale and guidance_scale > 0.0 and hidden_uncond is not None

        rates = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(-1)
        num_maskings = torch.ceil(_masking_rate(rates, exponent) * d).long()
        ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])           # per-iter codebook counts, sum == d
        cnt = 0
        for i in range(num_iter):
            k = int(ks[i, 0].item())
            if k == 0:
                continue
            code_embed = self.embed_code(self.depthsum_embedding(code))
            if guided:
                mog_input = torch.cat([code_embed + hidden_states, code_embed + hidden_uncond], dim=0)
                mu, logs = self.mog_head.infer(
                    mog_input, temperature=temperature, guidance_scale=guidance_scale,
                    top_p=top_p, generator=generator,
                )
            else:
                mog_input = code_embed + hidden_states
                mu, logs = self.mog_head.infer(
                    mog_input, temperature=temperature, top_p=top_p, generator=generator,
                )
            z = mu
            if noise_scale:
                eps = torch.randn(mu.shape, device=device, dtype=mu.dtype, generator=generator)
                z = z + torch.exp(logs) * eps * noise_scale
            code = self._rvq_quantize(z, code, cnt, k)
            cnt += k
        return code

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        speaker: str = "Aria",
        device=None,
        dtype=None,
        subword_id_to_char_ids: dict | None = None,
        char_pad_idx: int | None = None,
        text_pad_id: int | None = None,
        text_eos_id: int | None = None,
        speech_pad_id: int | None = None,
    ) -> dict:
        """Warm up the autoregressive KV cache from a pre-baked speaker prompt,
        replicating the NeMo ``set_init_inputs(speaker_name=...)`` -> warmup path.

        The prompt is ``P`` frames (``P == len(audio_prompt_latents[speaker])``,
        37 for Aria). The reference builds a silent-carrier audio prompt whose
        codec codes are all the codec-silence frame; the pre-baked latent then
        *replaces* the code embedding at every pre-BOS position, so no codec
        encode is needed. The layout of the ``P`` warmup positions is:

          * positions ``0 .. P-2``: the audio-prompt latent frames (pre-BOS);
          * position ``P-1`` (the BOS frame): ``embed_code(depthsum(silence))``
            (the shifted last silence code) ``+ bos_emb``.

        Text conditioning during warmup is the shifted prompt text — pad tokens
        with the trailing EOS — active only on the last two positions
        (``subword_mask`` = ``[.., P-2, P-1]``), matching the reference.

        When the token-id args (``text_pad_id`` / ``text_eos_id`` /
        ``speech_pad_id``) and char maps are supplied, the warmup is exact and
        the returned ``prev_codes`` is the ``speech_pad`` frame the reference
        carries in. Otherwise a simplified latent-only warmup is used (BOS added
        to the last latent frame, zero text conditioning) and ``prev_codes``
        falls back to the codec-silence frame.
        """
        latent = self.audio_prompt_latents[speaker]
        if device is not None:
            latent = latent.to(device)
        if dtype is not None:
            latent = latent.to(dtype)
        b = batch_size
        h = self.config.hidden_size
        p = latent.shape[1]
        latent = latent.expand(b, -1, -1)
        dev = latent.device

        faithful = (
            subword_id_to_char_ids is not None
            and char_pad_idx is not None
            and text_pad_id is not None
            and text_eos_id is not None
            and speech_pad_id is not None
        )

        if faithful:
            # position P-1 code embed: embed_code(depthsum(last silence code)).
            silence = self.codec_silence_tokens.to(dev).view(1, 1, -1).expand(b, 1, -1)
            bos_frame = self.embed_code(self.depthsum_embedding(silence)) + self.bos_emb.to(latent)
            code_embeds = torch.cat([latent[:, : p - 1], bos_frame], dim=1)     # [B, P, H]

            # warmup text: [pad ... pad, eos] with mask on the last two positions.
            subword_ids = torch.full((b, p), int(text_pad_id), dtype=torch.long, device=dev)
            subword_ids[:, p - 1] = int(text_eos_id)
            subword_mask = torch.zeros((b, p), dtype=torch.bool, device=dev)
            subword_mask[:, p - 2:] = True
            cond = self.text_conditioning(subword_ids, subword_mask, subword_id_to_char_ids, char_pad_idx)
            prev_codes = torch.full((b, self.config.num_quantizers), int(speech_pad_id), dtype=torch.long, device=dev)
        else:
            code_embeds = latent.clone()
            code_embeds[:, -1] = code_embeds[:, -1] + self.bos_emb.to(latent)
            cond = latent.new_zeros((1, 1, h))
            prev_codes = self.initial_prev_codes(b, device=dev)

        inputs_embeds = self.gated_fusion_audio_text(code_embeds, cond)
        _, cache = self.backbone_forward(inputs_embeds, start_pos=0, kv_cache=None, return_cache=True)
        # Unconditional stream for classifier-free guidance: same code (speaker)
        # embeddings, but the text conditioning replaced by the learned null embed
        # at every warmup position (reference ``uncond_dec_flag`` -> ``null_emb``).
        null_cond = self.null_emb.to(code_embeds).view(1, 1, h).expand(b, code_embeds.shape[1], h)
        inputs_uncond = self.gated_fusion_audio_text(code_embeds, null_cond)
        _, cache_uncond = self.backbone_forward(inputs_uncond, start_pos=0, kv_cache=None, return_cache=True)
        return {"kv": cache, "kv_uncond": cache_uncond, "pos": p, "prev_codes": prev_codes}

    def initial_prev_codes(self, batch_size: int, device=None) -> torch.Tensor:
        """The frame-0 ``prev_codes`` [B, num_quantizers]: the codec silence
        frame (the natural carry-in after the audio prompt)."""
        silence = self.codec_silence_tokens
        if device is not None:
            silence = silence.to(device)
        return silence.view(1, -1).expand(batch_size, -1).contiguous()

    @torch.no_grad()
    def infer_codes_one_step(
        self,
        state: dict,
        current_subword_id: torch.Tensor,
        prev_subword_id: torch.Tensor | None,
        prev_codes: torch.Tensor,
        cond: torch.Tensor | None = None,
        current_subword_mask: torch.Tensor | None = None,
        text_eos_id: int | None = None,
        num_iter: int = 8,
        temperature: float = 0.0,
        guidance_scale: float = 0.0,
        noise_scale: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """One autoregressive frame step.

        Args:
            state: cache dict from ``init_state`` / a previous step
                (``{"kv", "pos"}``).
            current_subword_id: [B, 1] nano text token id conditioning this frame
                (used only to force a codec-silence frame on text-EOS).
            prev_subword_id: [B, 1] previous text id — unused in this
                configuration (``context_hidden_size`` is ``None``); accepted for
                signature parity.
            prev_codes: [B, num_quantizers] codes emitted for the previous frame.
            cond: optional [B, 1, H] (or [1, 1, H]) precomputed text/subword
                conditioning (the caller runs ``embed_subword`` to build it, as
                that needs the tokenizer's subword->char map). ``None`` -> zero
                conditioning.
            text_eos_id: if given and ``current_subword_id == text_eos_id``,
                ``prev_codes`` is replaced by the codec silence frame before
                embedding (``inference_force_speech_silence_on_eos``).

        Returns:
            ``(codes [B, num_quantizers], new_state)`` — greedy + deterministic
            by default (``temperature == 0``, ``noise_scale == 0``).
        """
        if text_eos_id is not None:
            silence = self.codec_silence_tokens.view(1, -1).to(prev_codes.device).expand_as(prev_codes)
            prev_codes = torch.where(current_subword_id == text_eos_id, silence, prev_codes)

        code_embeds = self.embed_code(self.depthsum_embedding(prev_codes.unsqueeze(1)))  # [B, 1, H]
        if cond is None:
            cond = code_embeds.new_zeros((1, 1, self.config.hidden_size))
        inputs_embeds = self.gated_fusion_audio_text(code_embeds, cond)

        hidden, new_cache = self.backbone_forward(
            inputs_embeds, kv_cache=state["kv"], start_pos=state["pos"], return_cache=True
        )
        # Unconditional stream (classifier-free guidance): same code embeds, text
        # conditioning replaced by the learned null embedding, own KV cache.
        hidden_uncond, new_uncond = None, None
        guided = guidance_scale and guidance_scale > 0.0 and state.get("kv_uncond") is not None
        if guided:
            h = self.config.hidden_size
            null_cond = self.null_emb.to(code_embeds).view(1, 1, h).expand_as(code_embeds)
            inputs_uncond = self.gated_fusion_audio_text(code_embeds, null_cond)
            hidden_uncond, new_uncond = self.backbone_forward(
                inputs_uncond, kv_cache=state["kv_uncond"], start_pos=state["pos"], return_cache=True
            )
        codes = self.generate_step(
            hidden, hidden_uncond=hidden_uncond, num_iter=num_iter, temperature=temperature,
            guidance_scale=guidance_scale if guided else 0.0, noise_scale=noise_scale,
            top_p=top_p, generator=generator,
        )
        new_state = {"kv": new_cache, "kv_uncond": new_uncond, "pos": state["pos"] + 1}
        return codes.squeeze(1), new_state

    def forward(
        self,
        code: torch.Tensor,
        cond: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """Teacher-forced single pass producing the MoG head parameters.

        Args:
            code: [b, t, num_quantizers] long RVQ code ids for each frame (the
                inference-mode input: already the frame whose latent is embedded).
            cond: [b, t, H] (or broadcastable [1, 1, H]) text/subword conditioning;
                ``None`` means zero conditioning.
            position_ids: optional [t] positions (defaults to ``arange(t)``).

        Returns:
            ``(hidden_states, mog_logits, mog_mus, mog_mu_res, mog_logs)`` where
            ``hidden_states`` is the backbone output and the MoG params come from
            ``mog_head(embed_code(depthsum(code)) + hidden_states)``.
        """
        code_embeds = self.embed_code(self.depthsum_embedding(code))
        if cond is None:
            cond = code_embeds.new_zeros((1, 1, self.config.hidden_size))
        inputs_embeds = self.gated_fusion_audio_text(code_embeds, cond)
        hidden_states = self.backbone_forward(inputs_embeds, position_ids)

        mog_input = self.embed_code(self.depthsum_embedding(code)) + hidden_states
        mog_logits, mog_mus, mog_mu_res, mog_logs = self.mog_head(mog_input)
        return hidden_states, mog_logits, mog_mus, mog_mu_res, mog_logs

    def _prepare_char_inputs(
        self,
        subword_ids: torch.Tensor,
        subword_mask: torch.Tensor,
        subword_id_to_char_ids: dict,
        char_pad_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the valid (masked) subword ids into a padded char-id batch, in
        the row-major order ``embed_subword`` expects (port of NeMo
        ``CharAwareSubwordEncoder.prepare_inputs``)."""
        device = subword_ids.device
        sel = torch.masked_select(subword_ids, subword_mask).cpu().tolist()
        char_lists = [subword_id_to_char_ids.get(int(x), ()) for x in sel]
        lengths = torch.tensor([len(c) for c in char_lists], dtype=torch.long, device=device)
        n = lengths.numel()
        max_len = int(lengths.max().item()) if n > 0 else 0
        char_ids = torch.full((n, max_len), char_pad_idx, dtype=torch.long, device=device)
        for i, c in enumerate(char_lists):
            if c:
                char_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)
        return char_ids, lengths

    def text_conditioning(
        self,
        subword_ids: torch.Tensor,
        subword_mask: torch.Tensor | None,
        subword_id_to_char_ids: dict,
        char_pad_idx: int,
    ) -> torch.Tensor:
        """Per-frame text conditioning ``cond`` [B, T, H] (port of the NeMo
        ``_prepare_conditioning`` path for ``context_hidden_size is None``):
        run ``embed_subword`` on the subwords' char ids and add the subword-flag
        and BOS/EOS embeddings. Positions outside ``subword_mask`` stay zero."""
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids, dtype=torch.bool)
        cond = self.embed_code.weight.new_zeros((*subword_ids.shape, self.config.hidden_size))
        if not bool(subword_mask.any()):
            return cond
        char_ids, char_lengths = self._prepare_char_inputs(
            subword_ids, subword_mask, subword_id_to_char_ids, char_pad_idx
        )
        return self.embed_subword(char_ids, char_lengths, subword_ids, subword_mask)

    @torch.no_grad()
    def generate_codes(
        self,
        subword_stream: torch.Tensor,
        tokenizer,
        speaker: str = "Aria",
        temperature: float = 0.0,
        num_iter: int | None = None,
        guidance_scale: float | None = None,
        noise_scale: float | None = None,
        top_p: float | None = None,
        text_eos_id: int | None = None,
        text_pad_id: int | None = None,
        speech_pad_id: int | None = None,
        prev_codes: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Full autoregressive TTS: turn a subword/text-token stream ``[B, T]``
        into RVQ codes ``[B, T, num_quantizers]``.

        Builds the char-vocab maps from ``tokenizer`` once, warms the KV cache
        from the speaker prompt (``init_state``), then per frame computes the
        text conditioning from the current subword id and runs one
        ``infer_codes_one_step`` (greedy + deterministic by default). If
        ``text_eos_id`` is given, an EOS text token forces a codec-silence frame.

        Passing ``text_pad_id`` / ``text_eos_id`` / ``speech_pad_id`` enables the
        exact NeMo warmup (see ``init_state``); otherwise the simplified
        latent-only warmup is used.
        """
        # Inference sampling defaults from config (real speech needs CFG + noise +
        # top-p; without them the MoG head collapses to near-silence).
        if num_iter is None:
            num_iter = self.config.inference_num_iter
        if guidance_scale is None:
            guidance_scale = self.config.inference_guidance_scale
        if noise_scale is None:
            noise_scale = self.config.inference_noise_scale
        if top_p is None:
            top_p = self.config.inference_top_p

        b, t = subword_stream.shape
        device = subword_stream.device
        dtype = self.embed_code.weight.dtype
        char_vocab = build_char_vocab(tokenizer)
        s2c, _ = subword_to_char_ids(tokenizer, char_vocab)
        char_pad_idx = len(char_vocab)

        state = self.init_state(
            b, speaker=speaker, device=device, dtype=dtype,
            subword_id_to_char_ids=s2c, char_pad_idx=char_pad_idx,
            text_pad_id=text_pad_id, text_eos_id=text_eos_id, speech_pad_id=speech_pad_id,
        )
        if prev_codes is None:
            prev_codes = state.get("prev_codes")
            if prev_codes is None:
                prev_codes = self.initial_prev_codes(b, device=device)

        all_codes = []
        for i in range(t):
            cur = subword_stream[:, i: i + 1]
            prev = subword_stream[:, i - 1: i] if i > 0 else cur
            mask = torch.ones_like(cur, dtype=torch.bool)
            cond = self.text_conditioning(cur, mask, s2c, char_pad_idx)
            codes, state = self.infer_codes_one_step(
                state, cur, prev, prev_codes, cond=cond, text_eos_id=text_eos_id,
                num_iter=num_iter, temperature=temperature, guidance_scale=guidance_scale,
                noise_scale=noise_scale, top_p=top_p, generator=generator,
            )
            all_codes.append(codes)
            prev_codes = codes
        return torch.stack(all_codes, dim=1)

    @staticmethod
    def remap(name: str) -> str | None:
        top = {
            "tts_model._control_codes": "_control_codes",
            "tts_model.codec_silence_tokens": "codec_silence_tokens",
        }
        if name in top:
            return top[name]
        if name.startswith("tts_model.audio_prompt_latents."):
            voice = name[len("tts_model.audio_prompt_latents."):]
            return f"audio_prompt_latents.{voice}"
        prefix = "tts_model.tts_model."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        """Load ``tts_model.tts_model.*`` plus the top-level TTS buffers."""
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)
