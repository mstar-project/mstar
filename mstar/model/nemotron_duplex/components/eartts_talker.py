"""EarTTS talker (``tts_model.tts_model.*``): Gemma3-text transformer that
autoregressively emits RVQ codec codes, plus a mixture-of-gaussians (MoG) head,
a subword-conditioning encoder, and audio/text gated fusion.

Ported to exact checkpoint parity (28-layer Gemma3 backbone, head_dim 72 with
QK-norm; a 1-layer subword encoder; the MoG head; and the bespoke embedding
tables). The three top-level ``tts_model.*`` buffers (control codes, codec
silence tokens, per-voice audio-prompt latents) load here too. Forward is
Phase 5 (needs the EarTTS reference to verify code sampling / MoG numerics).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.components import RMSNorm
from mstar.model.components.mlp import GatedMLP
from mstar.model.nemotron_duplex.components._util import RawWeight, param
from mstar.model.nemotron_duplex.config import EarTTSConfig

H = 1152          # talker hidden
FF = 4608         # talker mlp intermediate
HEAD_DIM = 72
NUM_LAYERS = 28
CODE_DIM = 512    # rvq latent dim
NUM_QUANTIZERS = 31
CODEBOOK = 1024


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


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(H, H, bias=True)
        self.text_proj = nn.Linear(H, H, bias=True)
        self.final_norm = RMSNorm(H)
        self.gate = param(H)
        self.residual_scale = param()  # scalar


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
