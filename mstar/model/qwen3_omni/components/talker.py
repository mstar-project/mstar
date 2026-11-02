"""Qwen3-Omni Talker -- MoE transformer for codec token prediction.

The Talker is a smaller MoE transformer (1024 hidden, 20 layers) that runs
in streaming mode alongside the Thinker.  Key differences from the Thinker:

1. Standard 1-D RoPE (no 3-D MRoPE).
2. All layers are MoE with a shared expert (``SparseMoeBlockWithSharedExpert``).
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

from mstar.distributed.communication import CommGroup
from mstar.model.components import ParallelSparseMoeBlockWithSharedExpert, RMSNorm, TopKRouter
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention
from mstar.model.qwen3_omni.config import TALKER_ATTN, TALKER_KV, TALKER_POS, Qwen3OmniModelConfig, TalkerTextConfig

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

    def __init__(
        self, config: TalkerTextConfig, layer_idx: int,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = Qwen3OmniAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=1_000_000.0,
            rms_norm_eps=rms_norm_eps,
            use_mrope=False,
            comm_group=comm_group,
            attn_key=TALKER_ATTN,
            kv_key=TALKER_KV,
            pos_key=TALKER_POS
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

        self.mlp = ParallelSparseMoeBlockWithSharedExpert(
            hidden_size=hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            shared_expert=ParallelGatedMLP(
                hidden_size=hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                activation="silu",
                comm_group=comm_group,
            ),
            comm_group=comm_group,
            router=TopKRouter(
                hidden_size, config.num_experts, config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob, fused_softmax_topk=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # ---------- self-attention with pre-norm ----------
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # ---------- MoE FFN with pre-norm ----------
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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

    def __init__(self, config: TalkerTextConfig, comm_group: CommGroup | None = None):
        super().__init__()
        # NOTE: No embed_tokens here -- the HF Talker text model does not
        # have a text embedding table.  The Talker receives pre-computed
        # embeddings (projected Thinker states + codec embeddings).
        self.layers = nn.ModuleList(
            [
                Qwen3OmniTalkerLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Codec embedding for layer-0 codec tokens. Kept replicated across TP
        # ranks: the codec vocab is small (~codec_vocab_size codes) and the
        # embed is read after the LLM all-reduce so the input token id is the
        # same on every rank, making the lookup output naturally replicated.
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_embeds: torch.Tensor,
        label: str = "main",
    ) -> torch.Tensor:
        hidden_states = input_embeds
        # The label and layer index are cursors on the shared resources: bind
        # the label once, advance the index per layer. Passing them as
        # arguments instead would make inductor specialize on the int.
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, decoder_layer in enumerate(self.layers):
            decoder_layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(hidden_states=hidden_states)

        hidden_states = self.norm(hidden_states)
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

    def __init__(
        self, config: Qwen3OmniModelConfig,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        talker_text = config.talker_text
        thinker_hidden_size = config.thinker_hidden_size

        # Transformer backbone (TP-sharded inside each attention + MoE).
        self.model = Qwen3OmniTalkerLanguageModel(talker_text, comm_group=comm_group)

        # Codec head (replaces lm_head -- predicts codec tokens). Kept
        # unsharded: vocab is small (talker_text.vocab_size ~ 16k) so the
        # weight replicates cheaply, and the all-reduce inside the last
        # MoE layer already produces a replicated ``last_hidden`` — running
        # ``codec_head`` unsharded gives the same ``[B, V]`` logits on
        # every rank without an extra all-gather.
        self.codec_head = nn.Linear(
            talker_text.hidden_size, talker_text.vocab_size, bias=False
        )

        # Thinker->Talker projection MLPs. Replicated: the Thinker emits a
        # replicated ``thinker_states`` chunk (its all-reduces close the
        # loop before streaming), so each Talker rank runs an identical
        # projection on identical input and produces identical output.
        # TP'ing these would buy nothing — the cross-rank reduction would
        # just re-sum what every rank already has.
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
        label: str = "main",
    ) -> torch.Tensor:
        """Run the Talker backbone and return the final hidden states.

        The caller is responsible for applying ``self.codec_head`` to produce
        logits when needed.

        Args:
            input_embeds: [total_tokens, hidden_size] -- pre-embedded input
                (may combine codec embeddings and projected Thinker states).
            label: the plan key every layer runs against.

        Returns:
            hidden_states: [total_tokens, hidden_size] after final RMS norm.
        """
        return self.model(
            input_embeds=input_embeds, label=label,
        )
