"""EarTTS talker (``tts_model.tts_model.*``): Gemma3-text transformer that
autoregressively emits RVQ codec codes, plus a mixture-of-gaussians (MoG) head,
a subword-conditioning encoder, and audio/text gated fusion.

Ported to exact checkpoint parity (28-layer Gemma3 backbone, head_dim 72 with
QK-norm; a 1-layer subword encoder; the MoG head; and the bespoke embedding
tables). The three top-level ``tts_model.*`` buffers (control codes, codec
silence tokens, per-voice audio-prompt latents) load here too.

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

H = 1152          # talker hidden
FF = 4608         # talker mlp intermediate
HEAD_DIM = 72
NUM_HEADS = 16
NUM_LAYERS = 28
CODE_DIM = 512    # rvq latent dim
NUM_QUANTIZERS = 31
CODEBOOK = 1024
NUM_PREDICTIONS = 1024   # MoG mixture components
LOW_RANK = 64            # MoG low-rank mean dim
MIN_LOG_STD = -4.0
RMS_EPS = 1e-6

# Gemma3-text backbone specifics (from HF Gemma3TextConfig defaults + checkpoint).
QUERY_PRE_ATTN_SCALAR = 256.0     # attention scaling = QUERY_PRE_ATTN_SCALAR ** -0.5
ROPE_THETA_GLOBAL = 1_000_000.0   # full-attention layers
ROPE_THETA_LOCAL = 10_000.0       # sliding-attention layers
SLIDING_WINDOW_PATTERN = 6        # layer i is full attention iff (i + 1) % 6 == 0


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
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


class _Attn(nn.Module):
    """q/k/v/o projections (no bias); optional Gemma3 per-head QK-norm."""

    def __init__(self, qk_norm: bool):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(HEAD_DIM)
            self.k_norm = RMSNorm(HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        causal: bool = True,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head self-attention. Gemma3-style: optional per-head QK-norm
        (applied before rope), rope on the full head dim, and query scaling by
        ``QUERY_PRE_ATTN_SCALAR ** -0.5``. ``causal`` adds a triangular mask
        (talker); ``attn_bias`` is an additive mask broadcastable to
        ``[b, nh, tq, tk]`` (encoder key padding)."""
        b, t, _ = x.shape
        q = F.linear(x, self.q_proj.weight).view(b, t, NUM_HEADS, HEAD_DIM)
        k = F.linear(x, self.k_proj.weight).view(b, t, NUM_HEADS, HEAD_DIM)
        v = F.linear(x, self.v_proj.weight).view(b, t, NUM_HEADS, HEAD_DIM)
        if hasattr(self, "q_norm"):
            q = _gemma_rmsnorm(q, self.q_norm.weight)
            k = _gemma_rmsnorm(k, self.k_norm.weight)
        q = q.transpose(1, 2)   # [b, nh, t, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if cos is not None:
            cos_b = cos[None, None, :, :]
            sin_b = sin[None, None, :, :]
            q = q * cos_b + _rotate_half(q) * sin_b
            k = k * cos_b + _rotate_half(k) * sin_b
        scaling = QUERY_PRE_ATTN_SCALAR ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
        if causal:
            neg = torch.full((t, t), float("-inf"), device=x.device, dtype=scores.dtype)
            scores = scores + torch.triu(neg, diagonal=1)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, NUM_HEADS * HEAD_DIM)
        return F.linear(out, self.o_proj.weight)


class TalkerLayer(nn.Module):
    """Gemma3 decoder layer: input/post-attn/pre-ff/post-ff norms + QK-norm attn + gated MLP."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.self_attn = _Attn(qk_norm=True)
        self.post_attention_layernorm = RMSNorm(H)
        self.pre_feedforward_layernorm = RMSNorm(H)
        self.mlp = GatedMLP(H, FF, activation="gelu_tanh", bias=False)
        self.post_feedforward_layernorm = RMSNorm(H)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Gemma3 decoder layer with its 4 RMSNorms (input / post-attn /
        pre-ff / post-ff), each ``(1 + w)`` in fp32."""
        residual = x
        h = _gemma_rmsnorm(x, self.input_layernorm.weight)
        h = self.self_attn(h, cos, sin, causal=True)
        h = _gemma_rmsnorm(h, self.post_attention_layernorm.weight)
        x = residual + h

        residual = x
        h = _gemma_rmsnorm(x, self.pre_feedforward_layernorm.weight)
        h = self.mlp(h)
        h = _gemma_rmsnorm(h, self.post_feedforward_layernorm.weight)
        x = residual + h
        return x


class _EncoderLayer(nn.Module):
    """Subword-encoder layer: pre/post self-attn + pre/post ff norms (no QK-norm)."""

    def __init__(self):
        super().__init__()
        self.self_attn = _Attn(qk_norm=False)
        self.pre_self_attn_layernorm = RMSNorm(H)
        self.post_self_attn_layernorm = RMSNorm(H)
        self.pre_feedforward_layernorm = RMSNorm(H)
        self.post_feedforward_layernorm = RMSNorm(H)
        self.mlp = GatedMLP(H, FF, activation="gelu_tanh", bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """T5Gemma encoder layer: bidirectional attention (no QK-norm) with
        pre/post self-attn and pre/post ff RMSNorms."""
        residual = x
        h = _gemma_rmsnorm(x, self.pre_self_attn_layernorm.weight)
        h = self.self_attn(h, cos, sin, causal=False, attn_bias=attn_bias)
        h = _gemma_rmsnorm(h, self.post_self_attn_layernorm.weight)
        x = residual + h

        residual = x
        h = _gemma_rmsnorm(x, self.pre_feedforward_layernorm.weight)
        h = self.mlp(h)
        h = _gemma_rmsnorm(h, self.post_feedforward_layernorm.weight)
        x = residual + h
        return x


class _EncoderBackbone(nn.Module):
    def __init__(self, n_layers: int = 1):
        super().__init__()
        self.encoder = _Encoder(n_layers)


class _Encoder(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer() for _ in range(n_layers)])
        self.norm = RMSNorm(H)


class _BosEosEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.special_emb = nn.Embedding(3, H)
        self.special_flags = param(131072, dtype=torch.int64)
        self.pad_tensor = param(dtype=torch.int64)


class _SubwordFlagEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.cont_emb = nn.Embedding(2, H)
        self.is_continuation = param(131073, dtype=torch.int64)
        self.pad_tensor = param(dtype=torch.int64)


class EmbedSubword(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _EncoderBackbone(n_layers=1)
        self.embed_tokens = nn.Embedding(257, H)
        self.proj_embedding = nn.Linear(H, H, bias=False)
        self.bos_eos_emb = _BosEosEmb()
        self.subword_flag_emb = _SubwordFlagEmb()

    def encode_chars(self, char_ids: torch.Tensor, char_lengths: torch.Tensor) -> torch.Tensor:
        """Char-aware subword pooling: embed char ids, run the T5Gemma encoder
        (bidirectional, rope theta 10000) with a key-padding mask, mean-pool over
        valid chars, and project. ``char_ids`` [N, Lc], ``char_lengths`` [N] ->
        ``[N, H]``."""
        n, lc = char_ids.shape
        char_mask = torch.arange(lc, device=char_ids.device)[None, :] < char_lengths[:, None]  # [N, Lc]
        char_embeds = self.embed_tokens(char_ids)  # [N, Lc, H]

        cos, sin = _rope_cos_sin(torch.arange(lc, device=char_ids.device), ROPE_THETA_LOCAL, HEAD_DIM)
        neg = torch.finfo(char_embeds.dtype).min
        attn_bias = torch.where(char_mask[:, None, None, :], 0.0, neg).to(char_embeds.dtype)  # [N,1,1,Lc]

        # T5Gemma encoder scales the input embeddings by sqrt(hidden_size).
        x = char_embeds * (H ** 0.5)
        for layer in self.backbone.encoder.layers:
            x = layer(x, cos, sin, attn_bias)
        x = _gemma_rmsnorm(x, self.backbone.encoder.norm.weight)

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
            subword_ids.shape + (H,), device=subword_ids.device, dtype=out_emb.dtype
        )
        subword_embeds[subword_mask] = out_emb

        # continuation-flag embedding (index 0 forced to zero in the checkpoint)
        vocab = self.subword_flag_emb.cont_emb.num_embeddings  # 2
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
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(H, H, bias=True)
        self.text_proj = nn.Linear(H, H, bias=True)
        self.final_norm = RMSNorm(H)
        self.gate = param(H)
        self.residual_scale = param()  # scalar

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Gated projected sum of audio + text, then RMSNorm.

        ``audio_emb`` is divided by the number of codebooks; the per-channel
        ``gate`` and scalar ``residual_scale`` are sigmoided in fp32.
        """
        dtype = audio_emb.dtype
        audio_emb = audio_emb / NUM_QUANTIZERS
        audio_h = F.linear(audio_emb, self.audio_proj.weight, self.audio_proj.bias)
        text_h = F.linear(text_emb, self.text_proj.weight, self.text_proj.bias)
        gate = torch.sigmoid(self.gate.float())
        res = torch.sigmoid(self.residual_scale.float())
        h = gate.to(dtype) * audio_h + (1.0 - gate).to(dtype) * text_h
        h = res.to(dtype) * h
        h = _gemma_rmsnorm(h.float(), self.final_norm.weight).to(dtype)
        return h


class _MogMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = GatedMLP(H, FF, activation="gelu_tanh", bias=False)
        self.pre_norm = RMSNorm(H)
        self.post_norm = RMSNorm(H)


class MogHead(nn.Module):
    """Mixture-of-gaussians output head over the RVQ latent space.

    ``mlp_stack`` is 3 gated-MLP blocks followed by a bare scale vector
    (index 3, ``mlp_stack.3.weight`` [H]).
    """

    def __init__(self, n_mlp: int = 3):
        super().__init__()
        self.low_mat = param(CODEBOOK, CODE_DIM, 64)
        self.mlp_stack = nn.ModuleList([_MogMLP() for _ in range(n_mlp)] + [RawWeight(H)])
        self.proj_else = nn.Linear(H, CODE_DIM, bias=False)
        self.proj_logits = nn.Linear(H, CODEBOOK, bias=False)
        self.proj_logs = nn.Linear(H, 1, bias=False)
        self.proj_mus = nn.Linear(H, 65536, bias=False)

    def forward(self, x: torch.Tensor):
        """Deterministic (training-style) MoG head forward.

        Returns ``(logits, mus, mu_res, logs)``:
          * ``logits`` [b, t, num_predictions] mixture weights
          * ``mus``    [b, t, num_predictions, low_rank] low-rank means
          * ``mu_res`` [b, t, out_size] residual mean
          * ``logs``   [b, t, 1] log std, clamped to ``MIN_LOG_STD``
        """
        b, t, _ = x.shape
        # mlp_stack: 3 residual MLPLayers (pre_norm -> mlp -> post_norm) then a final RMSNorm.
        for block in self.mlp_stack[:3]:
            y = _gemma_rmsnorm(x, block.pre_norm.weight)
            y = block.mlp(y)
            y = _gemma_rmsnorm(y, block.post_norm.weight)
            x = x + y
        x = _gemma_rmsnorm(x, self.mlp_stack[3].weight)

        logits = F.linear(x, self.proj_logits.weight)
        mus = F.linear(x, self.proj_mus.weight).view(b, t, NUM_PREDICTIONS, LOW_RANK)
        logs = F.linear(x, self.proj_logs.weight).clamp_min(MIN_LOG_STD)
        mu_res = F.linear(x, self.proj_else.weight)
        return logits, mus, mu_res, logs


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TalkerLayer() for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(H)


class EarTTSTalker(nn.Module):
    def __init__(self, config: EarTTSConfig, voices=("Aria",)):
        super().__init__()
        self.config = config
        self.embed_code = nn.Linear(CODE_DIM, H, bias=False)
        self.backbone = _Backbone()
        self.embed_subword = EmbedSubword()
        self.gated_fusion_audio_text = GatedFusion()
        self.mog_head = MogHead()
        # bespoke embeddings / tables
        self.bos_emb = param(H)
        self.null_emb = param(H)
        self.audio_prompt_projection_W = param(H, H)
        self.rvq_embs = param(NUM_QUANTIZERS, CODEBOOK, CODE_DIM)
        # top-level tts_model.* buffers (loaded via the talker node)
        self._control_codes = param(3, dtype=torch.int64)
        self.codec_silence_tokens = param(NUM_QUANTIZERS, dtype=torch.int64)
        self.audio_prompt_latents = nn.ParameterDict(
            {v: param(1, 37, H) for v in voices}
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

    def backbone_forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Gemma3-text decoder stack over ``inputs_embeds`` [b, t, H].

        Full-attention layers (``(i+1) % 6 == 0``) use rope_theta 1e6; the rest
        (sliding-attention) use rope_local_base_freq 1e4. For sequences shorter
        than the 7500 sliding window the two mask identically (plain causal), so
        only the rope base differs between layer types.
        """
        b, t, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(t, device=inputs_embeds.device)
        cos_g, sin_g = _rope_cos_sin(position_ids, ROPE_THETA_GLOBAL, HEAD_DIM)
        cos_l, sin_l = _rope_cos_sin(position_ids, ROPE_THETA_LOCAL, HEAD_DIM)
        x = inputs_embeds
        for i, layer in enumerate(self.backbone.layers):
            full = (i + 1) % SLIDING_WINDOW_PATTERN == 0
            cos, sin = (cos_g, sin_g) if full else (cos_l, sin_l)
            x = layer(x, cos, sin)
        x = _gemma_rmsnorm(x, self.backbone.norm.weight)
        return x

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
            cond = code_embeds.new_zeros((1, 1, H))
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
