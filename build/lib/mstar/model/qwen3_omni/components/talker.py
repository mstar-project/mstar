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

import flashinfer
import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.engine.kv_cache_engine import BatchedCacheManager
from mstar.model.components import ParallelSparseMoeBlockWithSharedExpert, RMSNorm
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig, TalkerTextConfig
from mstar.utils.attention import decode_attn_nhd, fused_qk_norm_rope

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
        comm_group: TPCommGroup | None = None,
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
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        # ---------- self-attention with pre-norm ----------
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache_handle=cache_handle,
        )
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

    def __init__(self, config: TalkerTextConfig, comm_group: TPCommGroup | None = None):
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
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states=hidden_states, cache_handle=cache_handle
            )

        cache_handle.advance_seq_lens()

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
        comm_group: TPCommGroup | None = None,
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
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3OmniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            use_mrope=False,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            activation="silu",
        )

    def forward(self, hidden_states, cache_handle):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, cache_handle=cache_handle,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        self.norm = RMSNorm(cp.hidden_size, eps=cp.rms_norm_eps)

        # Per-layer codec embeddings for residual layers 1..(G-1)
        # codec_embedding.{0} embeds layer-1 codes, ..., codec_embedding.{G-2} embeds layer-(G-1)
        num_residual = config.num_code_groups - 1
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(cp.vocab_size, cp.hidden_size)
            for _ in range(num_residual)
        ])

    def forward(self, hidden_states: torch.Tensor, cache_handle: BatchedCacheManager):
        """
        Just runs the inner layers; does NOT run embedding.
        """
        for _layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(_layer_idx)
            hidden_states = decoder_layer(
                hidden_states,
                cache_handle
            )
        hidden_states = self.norm(hidden_states)
        cache_handle.advance_seq_len()
        return hidden_states

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

    The Code Predictor runs 15 autoregressive steps (for residual codebooks
    1..15) in float32 for precision.

    Two forward paths:
      * ``forward(embed, cache_manager)``  -- paged-FlashInfer eager path,
        used as a fallback when CUDA graphs are disabled.
      * ``forward_depth_unrolled(...)``   -- SDPA + dense-KV path that runs
        every layer with per-token position IDs and a static KV cache tensor.
        This is what the unrolled CUDA graph captures.
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

        # Stacked LM-head weight buffer, populated by
        # ``consolidate_stacked_weights()`` after the per-head ``nn.Linear``
        # weights are loaded. Registered as a non-persistent buffer (so it
        # doesn't appear in state_dicts) with a meta placeholder -- the real
        # tensor is assigned later once the per-head weights exist on the
        # target device. Shape: ``[num_residual, vocab, hidden]``.
        # The unrolled MTP graph indexes this with a Python-static int at
        # capture time (Option P), which is fixed-address and graph-safe.
        self.register_buffer(
            "lm_head_weight",
            torch.empty(num_residual, cp.vocab_size, cp.hidden_size),
            persistent=False,
        )

        self._num_residual = num_residual

    def consolidate_stacked_weights(self) -> None:
        """Stack per-head ``nn.Linear`` weights into ``lm_head_weight``.

        Must be called once after ``load_weights_from_hf_shards`` has
        populated the ``lm_head`` ModuleList. Replaces the placeholder
        ``lm_head_weight`` buffer with a real, contiguous tensor on the
        same device/dtype as the per-head weights. After this call, the
        unrolled graph path uses ``lm_head_weight[i]`` instead of
        ``lm_head[i].weight``. The per-head ``nn.Linear`` modules are kept
        (used by the eager fallback path) -- the duplicated memory is
        ~120 MB, small relative to the full Qwen3-Omni checkpoint.
        """
        stacked = torch.stack(
            [lm.weight.data for lm in self.lm_head], dim=0,
        ).contiguous()
        # Assigning via setattr updates the already-registered buffer in
        # place (see nn.Module.__setattr__); this replaces the meta
        # placeholder created in __init__ with the real device tensor.
        self.lm_head_weight = stacked

    def forward(self, *args):
        return self.model(*args)

    def get_embedding(self, group_idx: int):
        assert group_idx > 0
        return self.model.codec_embedding[group_idx-1]

    def get_lm_head(self, group_idx: int):
        assert group_idx > 0
        return self.lm_head[group_idx-1]

    @property
    def codec_embedding(self):
        """Alias for submodule access."""
        return self.model.codec_embedding

    @torch.compiler.disable
    def _apply_rope(self, q_rot, k_rot, flat_pos, rope_theta):
        # moved to its own function for torch.compile purposes
        return flashinfer.rope.apply_rope_pos_ids(
            q_rot, k_rot,
            pos_ids=flat_pos,
            rope_theta=rope_theta,
            interleave=False,
        )

    # ------------------------------------------------------------------
    # SDPA + dense-KV depth forward (for unrolled CUDA-graph capture)
    # ------------------------------------------------------------------
    def forward_depth_unrolled(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        """Run one "slice" of the code-predictor AR loop with SDPA + dense KV.

        This is the CUDA-graph-compatible replacement for the paged-FlashInfer
        attention path. It is safe to call repeatedly inside a single
        ``torch.cuda.graph`` capture because:

          * The KV cache is a preallocated dense tensor (no plan() call).
          * Position IDs come in directly (no PlanState bookkeeping).
          * Write/read slices use Python-static ``cache_pos`` / ``seq_len`` so
            the captured kernel sees identical shapes every replay.

        Args:
            inputs_embeds: ``[bs, seq_len, hidden_size]``. ``seq_len`` is 2 for
                the initial "prefill" call (``[last_hidden, c0_embed]``) and 1
                for every subsequent decode iteration.
            position_ids: ``[bs, seq_len]`` int tensor with the absolute
                position of each input token within the depth sequence.
            kv_cache: ``[n_layers, bs, 2, max_seq_len, n_kv_heads, head_dim]``
                dense tensor preallocated by the runner. The runner zeroes it
                at the start of every graph replay so stale state cannot leak.
            cache_pos: write offset into ``kv_cache``'s seq-len axis. The new
                K/V tokens land at ``[cache_pos : cache_pos + seq_len]``.

        Returns:
            ``[bs, seq_len, hidden_size]`` final hidden states after the code
            predictor's 5 layers + final RMSNorm. The caller is responsible
            for applying the per-codebook LM head.
        """
        hidden_states = inputs_embeds
        bs, seq_len, hidden_size = hidden_states.shape

        first_attn = self.model.layers[0].self_attn
        n_kv_heads = first_attn.num_kv_heads
        n_q_heads = first_attn.num_heads
        head_dim = first_attn.head_dim

        # FlashInfer's rope kernels require bf16; the rest of the code
        # predictor runs at the weights' native dtype (fp32 for quality per
        # the KVCacheEngine.has_autocast=False invariant).
        flat_pos = position_ids.reshape(-1)

        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            residual = hidden_states

            hidden_states = layer.input_layernorm(hidden_states)

            total_tokens = bs * seq_len
            qkv = F.linear(hidden_states, layer.self_attn.qkv_proj.weight)
            q_size = n_q_heads * head_dim
            kv_size = n_kv_heads * head_dim
            q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(total_tokens, n_q_heads, head_dim)
            k = k.view(total_tokens, n_kv_heads, head_dim)

            rope_theta = float(attn.rope_theta)
            q = fused_qk_norm_rope(
                q, attn.q_norm.weight,
                pos=flat_pos,
                eps=attn.q_norm.variance_epsilon,
                rope_theta=rope_theta
            )
            k = fused_qk_norm_rope(
                k, attn.k_norm.weight,
                pos=flat_pos,
                eps=attn.k_norm.variance_epsilon,
                rope_theta=rope_theta
            )

            q = q.view(bs, seq_len, n_q_heads, head_dim)
            k = k.view(bs, seq_len, n_kv_heads, head_dim)
            v = v.view(bs, seq_len, n_kv_heads, head_dim)

            # Append K, V into dense cache at cache_pos .. cache_pos+seq_len.
            kv_cache[layer_idx, :, 0, cache_pos:cache_pos + seq_len] = k
            kv_cache[layer_idx, :, 1, cache_pos:cache_pos + seq_len] = v

            # Read K, V for all positions up to and including the new tokens.
            valid_len = cache_pos + seq_len
            k_full = kv_cache[layer_idx, :, 0]
            v_full = kv_cache[layer_idx, :, 1]

            attn_out = decode_attn_nhd(
                q, k_full, v_full, valid_len
            )
            attn_out = attn_out.reshape(bs, seq_len, -1)
            hidden_states = residual + attn.o_proj(attn_out)

            # Post-attention RMSNorm + dense MLP.
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # Final norm.
        hidden_states = self.model.norm(hidden_states)
        return hidden_states
