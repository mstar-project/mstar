"""Qwen3-Omni Talker -- MoE transformer for codec token prediction.

The Talker is a smaller MoE transformer (1024 hidden, 20 layers) that runs
in streaming mode alongside the Thinker.  Key differences from the Thinker:

1. Standard 1-D RoPE (no 3-D MRoPE).
2. All layers are MoE with a shared expert (``Qwen3OmniTalkerSparseMoeBlock``).
3. No ``lm_head`` -- uses ``codec_head`` for codec token prediction.
4. Has ``codec_embedding`` for layer-0 codec tokens.
5. Has ``text_projection`` and ``hidden_projection`` MLPs that project
   Thinker hidden states into the Talker's embedding space.

Weight prefix: ``talker.``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mminf.engine.ar_engine import BatchedCacheManager
from mminf.model.qwen3_omni.components.attention import (
    Qwen3OmniAttention,
    Qwen3OmniRMSNorm,
)
from mminf.model.qwen3_omni.components.moe import Qwen3OmniTalkerSparseMoeBlock
from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig, TalkerTextConfig
from mminf.utils.flashinfer_utils import run_rms_norm

# ---------------------------------------------------------------------------
# Projection MLP (Thinker -> Talker)
# ---------------------------------------------------------------------------


class Qwen3OmniResizeMLP(nn.Module):
    """Projection MLP used for ``text_projection`` and ``hidden_projection``.

    Projects from ``thinker_hidden_size`` to ``talker_hidden_size`` using a
    two-layer MLP with SiLU activation::

        output = linear_fc2(silu(linear_fc1(x)))

    Weight names match HF checkpoint layout:
      ``linear_fc1.weight``, ``linear_fc1.bias``,
      ``linear_fc2.weight``, ``linear_fc2.bias``.
    """

    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(x)))


# ---------------------------------------------------------------------------
# Talker decoder layer
# ---------------------------------------------------------------------------


class Qwen3OmniTalkerLayer(nn.Module):
    """Single Talker transformer layer (pre-norm attention + MoE FFN)."""

    def __init__(self, config: TalkerTextConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = Qwen3OmniRMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = Qwen3OmniAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            # Talker reuses the same rope_theta as the Thinker (1e6)
            rope_theta=1_000_000.0,
            rms_norm_eps=rms_norm_eps,
            use_mrope=False,  # Standard 1-D RoPE, NOT 3-D MRoPE
        )
        self.post_attention_layernorm = Qwen3OmniRMSNorm(hidden_size, rms_norm_eps)

        # Every Talker layer is MoE (with shared expert + sigmoid gate)
        self.mlp = Qwen3OmniTalkerSparseMoeBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        # ---------- self-attention with pre-norm ----------
        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states,
            self.input_layernorm.weight,
            eps=self.input_layernorm.variance_epsilon,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache_handle=cache_handle,
        )
        hidden_states = residual + hidden_states

        # ---------- MoE FFN with pre-norm ----------
        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states,
            self.post_attention_layernorm.weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Talker model (backbone without head)
# ---------------------------------------------------------------------------


class Qwen3OmniTalkerLanguageModel(nn.Module):
    """Talker transformer backbone (embedding + N layers + final norm).

    This corresponds to the ``talker.model.*`` weight namespace.
    """

    def __init__(self, config: TalkerTextConfig):
        super().__init__()
        # NOTE: No embed_tokens here -- the HF Talker text model does not
        # have a text embedding table.  The Talker receives pre-computed
        # embeddings (projected Thinker states + codec embeddings).
        self.layers = nn.ModuleList(
            [
                Qwen3OmniTalkerLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3OmniRMSNorm(config.hidden_size, config.rms_norm_eps)

        # Codec embedding for layer-0 codec tokens
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states=hidden_states, cache_handle=cache_handle
            )

        cache_handle.advance_seq_lens()

        hidden_states = run_rms_norm(
            hidden_states, self.norm.weight, eps=self.norm.variance_epsilon
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level Talker (backbone + codec_head + projections)
# ---------------------------------------------------------------------------


class Qwen3OmniTalkerModel(nn.Module):
    """Complete Talker module with codec head and Thinker-to-Talker projections.

    Weight namespace::

        talker.model.embed_tokens.weight
        talker.model.layers.{i}.*
        talker.model.norm.weight
        talker.model.codec_embedding.weight
        talker.codec_head.weight
        talker.text_projection.linear_fc1.{weight,bias}
        talker.text_projection.linear_fc2.{weight,bias}
        talker.hidden_projection.linear_fc1.{weight,bias}
        talker.hidden_projection.linear_fc2.{weight,bias}
    """

    def __init__(self, config: Qwen3OmniModelConfig):
        super().__init__()
        talker_text = config.talker_text
        thinker_hidden_size = config.thinker_hidden_size

        # Transformer backbone
        self.model = Qwen3OmniTalkerLanguageModel(talker_text)

        # Codec head (replaces lm_head -- predicts codec tokens)
        self.codec_head = nn.Linear(
            talker_text.hidden_size, talker_text.vocab_size, bias=False
        )

        # Projection MLPs: Thinker hidden space -> Talker hidden space
        # linear_fc1: (thinker_hidden, intermediate) -> linear_fc2: (intermediate, talker_hidden)
        # HF uses text_config.intermediate_size as the intermediate dimension
        intermediate_size = talker_text.intermediate_size
        self.text_projection = Qwen3OmniResizeMLP(
            thinker_hidden_size, intermediate_size, talker_text.hidden_size
        )
        self.hidden_projection = Qwen3OmniResizeMLP(
            thinker_hidden_size, intermediate_size, talker_text.hidden_size
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        """Run the Talker backbone and return the final hidden states.

        The caller is responsible for applying ``self.codec_head`` to produce
        logits when needed.

        Args:
            input_embeds: [total_tokens, hidden_size] -- pre-embedded input
                (may combine codec embeddings and projected Thinker states).
            cache_handle: ``BatchedCacheManager`` for paged KV attention.

        Returns:
            hidden_states: [total_tokens, hidden_size] after final RMS norm.
        """
        return self.model(input_embeds=input_embeds, cache_handle=cache_handle)


# ---------------------------------------------------------------------------
# Code Predictor (lightweight transformer for residual codebook layers)
# ---------------------------------------------------------------------------


class Qwen3OmniCodePredictorLayer(nn.Module):
    """Single Code Predictor decoder layer (attention + dense MLP, no MoE).

    Uses QK-norm, standard 1D RoPE (no 3D MRoPE).
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_norm_eps: float, rope_theta: float):
        super().__init__()
        from mminf.model.qwen3_omni.components.moe import Qwen3OmniMLP

        self.input_layernorm = Qwen3OmniRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3OmniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            use_mrope=False,
        )
        self.post_attention_layernorm = Qwen3OmniRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3OmniMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states, cache_handle):
        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states, self.input_layernorm.weight,
            eps=self.input_layernorm.variance_epsilon,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states, cache_handle=cache_handle,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states, self.post_attention_layernorm.weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3OmniCodePredictorInnerModel(nn.Module):
    """Inner model (maps to ``talker.code_predictor.model.*`` in HF weights).

    Contains layers, norm, and per-layer codec embeddings.
    """

    def __init__(self, config: Qwen3OmniModelConfig):
        super().__init__()
        cp = config.code_predictor

        self.layers = nn.ModuleList([
            Qwen3OmniCodePredictorLayer(
                hidden_size=cp.hidden_size,
                intermediate_size=cp.intermediate_size,
                num_heads=cp.num_attention_heads,
                num_kv_heads=cp.num_key_value_heads,
                head_dim=cp.head_dim,
                rms_norm_eps=cp.rms_norm_eps,
                rope_theta=cp.rope_theta,
            )
            for _ in range(cp.num_hidden_layers)
        ])
        self.norm = Qwen3OmniRMSNorm(cp.hidden_size, eps=cp.rms_norm_eps)

        # Per-layer codec embeddings for residual layers 1..(G-1)
        # codec_embedding.{0} embeds layer-1 codes, ..., codec_embedding.{G-2} embeds layer-(G-1)
        num_residual = config.num_code_groups - 1
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(cp.vocab_size, cp.hidden_size)
            for _ in range(num_residual)
        ])

    def forward(self, *args):
        # This is never called; keeping for compatibility with nn.Module
        return

    def get_input_embeddings(self):
        """Return the codec embedding list (for HF compatibility)."""
        return self.codec_embedding


class Qwen3OmniCodePredictor(nn.Module):
    """Code Predictor: lightweight transformer for residual codebook layers.

    HF weight layout::

        talker.code_predictor.model.layers.{0-4}.*
        talker.code_predictor.model.norm.weight
        talker.code_predictor.model.codec_embedding.{0-30}.weight
        talker.code_predictor.lm_head.{0-30}.weight

    The Code Predictor runs 31 autoregressive steps (for layers 1-31)
    with NO persistent KV cache and in float32 for precision.
    """

    def __init__(self, config: Qwen3OmniModelConfig):
        super().__init__()
        cp = config.code_predictor

        self.model = Qwen3OmniCodePredictorInnerModel(config)

        # Per-layer output heads for residual layers 1..(G-1)
        num_residual = config.num_code_groups - 1
        self.lm_head = nn.ModuleList([
            nn.Linear(cp.hidden_size, cp.vocab_size, bias=False)
            for _ in range(num_residual)
        ])

    def forward(self, *args):
        # This is never called; keeping for compatibility with nn.Module
        return

    @property
    def codec_embedding(self):
        """Alias for submodule access."""
        return self.model.codec_embedding
