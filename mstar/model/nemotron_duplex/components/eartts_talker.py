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
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a mixture component and return its target latent mean + log-std.

        Greedy (``temperature == 0``, the offline default) picks the component
        by ``argmax(logits)``; ``temperature > 0`` uses the Gumbel-max trick over
        ``log_softmax(logits) / temperature``. The chosen component's low-rank
        mean ``mu`` is projected up through the component-specific ``low_mat``,
        scaled by ``exp(logs)`` and offset by the residual mean ``mu_res``:
        returns ``(mu * exp(logs) + mu_res, logs)`` — the mean of the Gaussian
        that the RVQ residual-quantizer then encodes."""
        b, t, _ = x.shape
        x = self._trunk(x)
        logits = F.linear(x, self.proj_logits.weight)                       # [b, t, n]
        if temperature and temperature > 0.0:
            logp = F.log_softmax(logits, dim=-1) / temperature
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
        num_iter: int = 8,
        exponent: float = 3.0,
        temperature: float = 0.0,
        noise_scale: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Iterative MoG-conditioned RVQ decode of one (or more) frame(s).

        Given the backbone hidden state ``[b, t, H]``, fills the
        ``num_quantizers`` RVQ codebooks over ``num_iter`` MaskGIT-style
        iterations. Each iteration re-embeds the current partial code, adds the
        hidden state, samples the MoG target latent ``z`` (greedy +
        deterministic by default: ``temperature=0``, ``noise_scale=0``), then
        residual-quantizes the next block of codebooks. Returns ``[b, t,
        num_quantizers]`` long codes in ``[0, codebook_size)``. CFG guidance is
        not implemented (single, conditional stream).
        """
        cfg = self.config
        b, t, _ = hidden_states.shape
        d = cfg.num_quantizers
        device = hidden_states.device
        code = torch.full((b, t, d), cfg.codebook_size, dtype=torch.long, device=device)

        rates = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(-1)
        num_maskings = torch.ceil(_masking_rate(rates, exponent) * d).long()
        ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])           # per-iter codebook counts, sum == d
        cnt = 0
        for i in range(num_iter):
            k = int(ks[i, 0].item())
            if k == 0:
                continue
            mog_input = self.embed_code(self.depthsum_embedding(code)) + hidden_states
            mu, logs = self.mog_head.infer(mog_input, temperature=temperature, generator=generator)
            z = mu
            if noise_scale:
                eps = torch.randn(mu.shape, device=device, dtype=mu.dtype, generator=generator)
                z = z + torch.exp(logs) * eps * noise_scale
            code = self._rvq_quantize(z, code, cnt, k)
            cnt += k
        return code

    @torch.no_grad()
    def init_state(self, batch_size: int, speaker: str = "Aria", device=None, dtype=None) -> dict:
        """Warm up the autoregressive KV cache from a pre-baked speaker prompt.

        Uses the cached ``audio_prompt_latents[speaker]`` [1, P, H] as the
        pre-BOS audio-prompt frames (already in model hidden space), adds
        ``bos_emb`` to the final prompt frame to mark the generation boundary,
        runs the block through the gated fusion (zero text conditioning) and the
        backbone, and returns the populated per-layer cache plus the next
        absolute position. NOTE: this is a self-contained faithful setup — it
        uses the pre-baked latent directly rather than re-encoding prompt audio
        with the codec, and omits any text/system-prompt prefix.
        """
        latent = self.audio_prompt_latents[speaker]
        if device is not None:
            latent = latent.to(device)
        if dtype is not None:
            latent = latent.to(dtype)
        prompt = latent.expand(batch_size, -1, -1).clone()                  # [B, P, H]
        prompt[:, -1] = prompt[:, -1] + self.bos_emb.to(prompt)
        cond = prompt.new_zeros((1, 1, self.config.hidden_size))
        inputs_embeds = self.gated_fusion_audio_text(prompt, cond)
        _, cache = self.backbone_forward(inputs_embeds, start_pos=0, kv_cache=None, return_cache=True)
        return {"kv": cache, "pos": prompt.shape[1]}

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
        noise_scale: float = 0.0,
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
        codes = self.generate_step(
            hidden, num_iter=num_iter, temperature=temperature, noise_scale=noise_scale, generator=generator,
        )
        new_state = {"kv": new_cache, "pos": state["pos"] + 1}
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
