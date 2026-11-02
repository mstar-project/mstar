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
from mstar.engine.resources.convenience import PackedAttentionCallable
from mstar.model.components import RMSNorm
from mstar.model.components.depth_decoder import DepthDecoder
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.qwen3_tts.config import (
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.utils.attention import apply_rope_pos_ids, decode_attn_nhd

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


class PackedStep:
    """What a decode step's attention layers rotate from when they take the
    packed prologue (``Qwen3TTSTalkerAttention``): the step's positions and the
    cos/sin tables over all positions. Empty on any other step."""

    def __init__(self) -> None:
        self.positions: torch.Tensor | None = None
        self.tables: tuple[torch.Tensor, torch.Tensor] | None = None


class Qwen3TTSTalkerAttention(ParallelAttention):
    """``ParallelAttention`` that takes the packed prologue on a decode replay:
    split, q/k norm, rotation and the cache write in one kernel (``attn.prologue``)."""

    def __init__(self, *, packed_step: PackedStep, **kwargs) -> None:
        super().__init__(**kwargs)
        self._packed_step = packed_step
        self._prologue_gammas: tuple[torch.Tensor, torch.Tensor] | None = None

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        self.attend = PackedAttentionCallable(kv=self.kv, attn=self.attn)

    def prologue_gammas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The kernel scales a norm by ``1 + w``; these norms scale by ``w``. Kept
        in fp32, where ``1 + (w - 1)`` is the checkpoint's weight."""
        if self._prologue_gammas is None:
            self._prologue_gammas = tuple(norm.weight.detach().float() - 1.0 for norm in (self.q_norm, self.k_norm))
        return self._prologue_gammas

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        step = self._packed_step
        if step.positions is None:
            return super().forward(hidden_states)
        qkv = self.qkv_proj(hidden_states)
        cos, sin = step.tables
        out, _ = self.attend.attend_packed(
            qkv, *self.prologue_gammas(), cos, sin, step.positions,
            self.num_heads, self.num_kv_heads, self.head_dim, self.q_norm.variance_epsilon, False,
        )
        return self.o_proj(out.reshape(hidden_states.shape[0], self.num_heads * self.head_dim))


class Qwen3TTSTalkerLayer(nn.Module):
    """One pre-norm Talker block backed by M* tensor-parallel layers."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
        packed_step: PackedStep | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3TTSTalkerAttention(
            packed_step=packed_step if packed_step is not None else PackedStep(),
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

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3TTSTalkerLanguageModel(nn.Module):
    """Backbone matching ``talker.model.*`` and the engine-managed KV cache."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.comm_group = comm_group if comm_group is not None else CommGroup.trivial()
        self._packed_step = PackedStep()
        self._packed: bool | None = None
        self._rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None
        self.layers = nn.ModuleList([
            Qwen3TTSTalkerLayer(config, comm_group=comm_group, packed_step=self._packed_step)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )

    @property
    def packed_prologue(self) -> bool:
        """Whether a decode replay's layers take the packed prologue, which asks
        the KV cache for flat write slots. One rank only: under TP the K/V heads
        can replicate past their count, which the kernel's layout does not hold.
        Settled on first read, once the layers are bound."""
        if self._packed is None:
            if not hasattr(self.layers[0].self_attn, "attend"):
                return False  # not bound yet
            attn = [layer.self_attn for layer in self.layers]
            self._packed = self.comm_group.world_size == 1 and all(a.attend.enable() for a in attn)
        return self._packed

    def rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin over every position, ``[positions, head_dim // 2]``, in fp32:
        the frequencies are too close together at this theta for bf16."""
        if self._rope_tables is None:
            config = self.config
            device = self.norm.weight.device
            steps = torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float64)
            inv_freq = config.rope_theta ** (-steps / config.head_dim)
            pos = torch.arange(config.max_position_embeddings, device=device, dtype=torch.float64)
            angle = (pos[:, None] * inv_freq[None, :]).to(torch.float32)
            self._rope_tables = (angle.cos(), angle.sin())
        return self._rope_tables

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``positions`` (one per token) sends the layers down the packed
        prologue: only a decode replay whose layers all take it passes them."""
        hidden_states = input_embeds
        self._packed_step.positions = positions
        self._packed_step.tables = self.rope_tables() if positions is not None else None
        # Which of the node's KV streams the layers write (`label`) and which
        # layer of it are cursors on the shared resources: bind the label once,
        # advance the index per layer. Passing them as arguments instead would
        # make inductor specialize on the int. The resources own the pages, so
        # module code stays clear of the allocator.
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states)
        # Sequence lengths still advance once per forward, after every layer
        # wrote K/V for the same packed range — the runner does it now, on the
        # step this forward was declared from.
        return self.norm(hidden_states)


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
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_embeds, label=label, positions=positions)


# ---------------------------------------------------------------------------
# CodePredictor: autoregression across codec groups within one frame
# ---------------------------------------------------------------------------
# The group loop is DepthDecoder's, shared with Qwen3-Omni's code predictor.


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


class Qwen3TTSCodePredictor(DepthDecoder):
    """Five-layer decoder that predicts codec groups 1 through 15.

    ``DepthDecoder.run_depth_loop`` walks the groups: position 0 holds the
    Talker hidden state, position g the embedding of group g's code. On
    platforms without the fused depth kernels it runs ``forward_depth_unrolled``,
    one position through every layer, per group.

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

    @property
    def input_projection(self) -> nn.Module | None:
        projection = self.small_to_mtp_projection
        return None if isinstance(projection, nn.Identity) else projection

    def consolidate_stacked_weights(self) -> None:
        """Pack the 15 loaded LM heads into graph-friendly contiguous storage."""
        self.lm_head_weight = torch.stack([
            head.weight.data for head in self.lm_head
        ]).contiguous()

    def forward_depth_unrolled(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        """Run all CodePredictor layers for one depth token.

        ``kv_cache`` layout is ``[layers, batch, K/V, groups, kv_heads,
        head_dim]``. Unlike Talker cache, it is frame-local scratch space:
        every generated frame starts at ``cache_pos=0`` and overwrites it.

        ``inputs_embeds`` is ``[batch, 1, talker_hidden]``; the returned
        hidden state is ``[batch, 1, predictor_hidden]``.
        """
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("CodePredictor decode expects exactly one token")
        end = cache_pos + seq_len
        flat_position_ids = position_ids.reshape(-1).to(torch.int32)

        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            qkv = attn.qkv_proj(hidden_states)
            q_size = attn.num_heads * attn.head_dim
            kv_size = attn.num_kv_heads * attn.head_dim
            q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
            k = k.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
            v = v.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
            q, k = attn._apply_qk_norm(q, k)
            # A stride-aware Triton kernel fuses both half rotations. The
            # previous tensor expression launched multiple elementwise and
            # concatenation kernels in every one of the 5 layers × 16 groups.
            q, k = apply_rope_pos_ids(
                q.reshape(batch_size * seq_len, attn.num_heads, attn.head_dim),
                k.reshape(
                    batch_size * seq_len,
                    attn.num_kv_heads,
                    attn.head_dim,
                ),
                flat_position_ids,
                float(attn.rope_theta),
            )
            q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
            k = k.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)

            # Append this depth position, then attend over the prefix already
            # generated within the same codec frame.
            kv_cache[layer_idx, :, 0, cache_pos:end].copy_(k)
            kv_cache[layer_idx, :, 1, cache_pos:end].copy_(v)
            attn_output = decode_attn_nhd(
                q,
                kv_cache[layer_idx, :, 0],
                kv_cache[layer_idx, :, 1],
                end,
            )
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            hidden_states = residual + attn.o_proj(attn_output)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        return self.model.norm(hidden_states)
