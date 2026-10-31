"""Neural components for Qwen3-TTS Talker and residual CodePredictor.

The checkpoint has two coupled autoregressive axes:

* Across time, the 28-layer Talker predicts codec group 0 at 12 Hz and uses
  M*'s paged KV cache. Its attention/MLP projections are tensor-parallel.
* Within one time step, the 5-layer CodePredictor walks codec groups 1-15.
  This short depth axis uses a fixed local KV tensor and is kept replicated.
  Its inputs (the Talker hidden state and every codec embedding) live in the
  Talker width; ``small_to_mtp_projection`` maps them into the predictor
  width when the two differ (1.7B: 2048 -> 1024; 0.6B: identity).

Class/module names intentionally follow Hugging Face checkpoint namespaces so
``load_hf_weights`` can stream parameters without a model-specific state-dict
rewrite beyond the standard fused Q/K/V and gate/up rules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.components.qwen3_lm import Qwen3Attention
from mstar.model.qwen3_tts.config import (
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.utils.attention import (
    apply_rope_pos_ids,
    decode_attn_nhd,
    qk_norm_rope_kv_write,
    qk_norm_rope_kv_write_applies,
)

# ---------------------------------------------------------------------------
# Talker backbone: autoregression across audio frames
# ---------------------------------------------------------------------------


class Qwen3TTSResizeMLP(nn.Module):
    """Two-layer text-embedding projection used by the official Talker."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(hidden_states)))


class Qwen3TTSTalkerLayer(nn.Module):
    """Parameter container for one pre-norm Talker block, backed by M*
    tensor-parallel layers. ``Qwen3TTSTalkerLanguageModel`` runs it."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            comm_group=comm_group,
            attn_key=TALKER_ATTN,
            kv_key=TALKER_KV,
            pos_key=TALKER_POS,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
            comm_group=comm_group,
        )


class Qwen3TTSTalkerLanguageModel(nn.Module):
    """Backbone matching ``talker.model.*`` and the engine-managed KV cache."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Qwen3TTSTalkerLayer(config, comm_group=comm_group)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        # Which of the node's KV streams the layers write (`label`) and which
        # layer of it are cursors on the shared resources: bind the label once,
        # advance the index per layer. Passing them as arguments instead would
        # make inductor specialize on the int. The resources own the pages, so
        # module code stays clear of the allocator.
        self.layers[0].self_attn.attend.bind_step(label)
        # Each residual add is deferred into the next norm: one fused add+norm
        # kernel (in place) per norm instead of an add and a norm.
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            if residual is None:
                residual = hidden_states
                h = layer.input_layernorm(hidden_states)
            else:
                h, residual = layer.input_layernorm.forward_residual(hidden_states, residual)
            h = layer.self_attn(h)
            h, residual = layer.post_attention_layernorm.forward_residual(h, residual)
            hidden_states = layer.mlp(h)
        # Sequence lengths still advance once per forward, after every layer
        # wrote K/V for the same packed range — the runner does it now, on the
        # step this forward was declared from.
        normed, _ = self.norm.forward_residual(hidden_states, residual)
        return normed


class Qwen3TTSTalkerModel(nn.Module):
    """Talker backbone, text projection, and codec-group-0 output head."""

    def __init__(
        self,
        config: Qwen3TTSModelConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        talker = config.talker
        self.model = Qwen3TTSTalkerLanguageModel(talker, comm_group=comm_group)
        self.text_projection = Qwen3TTSResizeMLP(
            talker.text_hidden_size,
            talker.text_hidden_size,
            talker.hidden_size,
        )
        self.codec_head = nn.Linear(
            talker.hidden_size, talker.vocab_size, bias=False
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        return self.model(input_embeds, label=label)


# ---------------------------------------------------------------------------
# CodePredictor: autoregression across codec groups within one frame
# ---------------------------------------------------------------------------
# Qwen3-Omni has the same high-level residual-code loop, but currently uses a
# different attention primitive, checkpoint/config namespace, sampling policy,
# and CUDA-graph capture path.  Sharing the loop before the auxiliary-sampler
# work tracked in #199 would couple two independently validated model ports;
# factor the common implementation after that engine interface is available.


class Qwen3TTSCodePredictorLayer(nn.Module):
    """Parameter container for one replicated residual-code decoder block."""

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.input_layernorm = RMSNorm(cp.hidden_size, cp.rms_norm_eps)
        self.self_attn = ParallelAttention(
            hidden_size=cp.hidden_size,
            num_heads=cp.num_attention_heads,
            num_kv_heads=cp.num_key_value_heads,
            head_dim=cp.head_dim,
            rope_theta=cp.rope_theta,
            qk_norm=True,
            rms_norm_eps=cp.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            cp.hidden_size, cp.rms_norm_eps
        )
        self.mlp = ParallelGatedMLP(
            hidden_size=cp.hidden_size,
            intermediate_size=cp.intermediate_size,
            activation="silu",
        )


class Qwen3TTSCodePredictorInnerModel(nn.Module):
    """Depth decoder and one embedding table per residual codec group."""

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.layers = nn.ModuleList([
            Qwen3TTSCodePredictorLayer(config)
            for _ in range(cp.num_hidden_layers)
        ])
        self.norm = RMSNorm(cp.hidden_size, cp.rms_norm_eps)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(cp.vocab_size, config.talker.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ])


class Qwen3TTSCodePredictor(nn.Module):
    """Five-layer decoder that predicts codec groups 1 through 15.

    ``forward_depth_unrolled`` performs one position on the group-depth axis.
    Callers first write the Talker hidden state at position 0, then repeatedly
    feed the preceding codec embedding at positions 1-15. Keeping this as
    tensor-only code allows the complete depth loop to be CUDA-graph captured.

    Every depth input arrives in the Talker width and passes through
    ``small_to_mtp_projection`` (a biased linear layer on the 1.7B
    checkpoints, identity on the 0.6B where both widths are 1024), exactly as
    the reference ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``
    does before its decoder.
    """

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.config = cp
        self.model = Qwen3TTSCodePredictorInnerModel(config)
        self.small_to_mtp_projection: nn.Module = (
            nn.Linear(config.talker.hidden_size, cp.hidden_size, bias=True)
            if cp.hidden_size != config.talker.hidden_size
            else nn.Identity()
        )
        self.lm_head = nn.ModuleList([
            nn.Linear(cp.hidden_size, cp.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ])
        self.register_buffer(
            # Populated after weight loading. A single contiguous tensor lets
            # the piecewise loop select a residual head without traversing a
            # Python ModuleList during CUDA Graph replay.
            "lm_head_weight",
            torch.empty(
                config.num_code_groups - 1,
                cp.vocab_size,
                cp.hidden_size,
            ),
            persistent=False,
        )

    def consolidate_stacked_weights(self) -> None:
        """Pack the 15 loaded LM heads into graph-friendly contiguous storage."""
        self.lm_head_weight = torch.stack([
            head.weight.data for head in self.lm_head
        ]).contiguous()

    def forward_depth_unrolled(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        batch_rows: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        """Run all CodePredictor layers for the next depth tokens.

        ``kv_cache`` layout is ``[layers, batch, K/V, groups, kv_heads,
        head_dim]``. Unlike Talker cache, it is frame-local scratch space:
        every generated frame starts at ``cache_pos=0`` and overwrites it.

        ``inputs_embeds`` is ``[batch, tokens, talker_hidden]``, whose tokens
        take the cache positions from ``cache_pos``; it may be overwritten.
        ``position_ids`` and ``batch_rows`` give each token's position and row,
        ``[batch * tokens]`` in that order and contiguous, since the fused
        kernel indexes them. The returned hidden state is ``[batch, tokens,
        predictor_hidden]``, each token having attended its row's prefix.
        """
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        batch_size, seq_len, hidden_size = hidden_states.shape
        end = cache_pos + seq_len
        hidden_states = hidden_states.reshape(-1, hidden_size)

        # Each residual add is deferred into the next norm: one fused add+norm
        # kernel per norm. These run on every layer of every pass.
        residual = None
        for layer_idx, layer in enumerate(self.model.layers):
            if residual is None:
                residual = hidden_states
                h = layer.input_layernorm(hidden_states)
            else:
                h, residual = layer.input_layernorm.forward_residual(hidden_states, residual)
            attn = layer.self_attn
            q, k, v = attn._project_qkv(h)
            attn_output = self._norm_rope_attend(
                attn, q, k, v, position_ids, batch_rows, kv_cache[layer_idx], cache_pos, end,
            )
            attn_output = attn_output.reshape(batch_size * seq_len, -1)
            h, residual = layer.post_attention_layernorm.forward_residual(attn.o_proj(attn_output), residual)
            hidden_states = layer.mlp(h)

        normed, _ = self.model.norm.forward_residual(hidden_states, residual)
        return normed.view(batch_size, seq_len, hidden_size)

    @torch.compiler.disable
    def _norm_rope_attend(
        self,
        attn: ParallelAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor,
        batch_rows: torch.Tensor,
        kv_layer: torch.Tensor,
        cache_pos: int,
        end: int,
    ) -> torch.Tensor:
        """q/k norm, RoPE and this pass's write to the layer's ``[batch, K/V,
        groups, kv_heads, head_dim]`` cache, then attention over the prefix
        written so far. One kernel covers all but the attention where it
        applies: a depth position is also the token's cache slot."""
        seq_len = end - cache_pos
        batch_size = q.shape[0] // seq_len
        if qk_norm_rope_kv_write_applies(q, kv_layer):
            qk_norm_rope_kv_write(
                q, k, v, attn.q_norm.weight, attn.k_norm.weight, position_ids,
                kv_layer, batch_rows, position_ids, q.shape[0],
                float(attn.q_norm.variance_epsilon), float(attn.k_norm.variance_epsilon),
                float(attn.rope_theta),
            )
        else:
            q, k = attn._apply_qk_norm(q, k)
            q, k = apply_rope_pos_ids(q, k, position_ids.to(torch.int32), float(attn.rope_theta))
            kv_layer[:, 0, cache_pos:end].copy_(k.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
            kv_layer[:, 1, cache_pos:end].copy_(v.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim))
        return decode_attn_nhd(
            q.view(batch_size, seq_len, attn.num_heads, attn.head_dim),
            kv_layer[:, 0],
            kv_layer[:, 1],
            end,
        )
