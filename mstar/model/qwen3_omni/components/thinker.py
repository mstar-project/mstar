"""Thinker MoE transformer with FlashInfer paged KV cache for Qwen3-Omni.

Architecture: embed_tokens -> N decoder layers -> final RMSNorm -> lm_head
Each decoder layer: input_layernorm -> attention -> residual -> post_attention_layernorm -> MLP/MoE -> residual

Key differences from Orpheus:
- MoE (SparseMoeBlock) on most layers, dense MLP on ``mlp_only_layers``
- QK-norm in attention (handled by ``Qwen3OmniAttention``)
- 3D MRoPE (passed through as ``cos_sin_3d``)
- Captures layer-0 embeddings and layer-N hidden states for Talker conditioning

Weight name prefix: ``thinker.``
  - thinker.model.embed_tokens.weight
  - thinker.model.layers.{i}.input_layernorm.weight
  - thinker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  - thinker.model.layers.{i}.self_attn.{q,k}_norm.weight
  - thinker.model.layers.{i}.block_sparse_moe.gate.weight
  - thinker.model.layers.{i}.block_sparse_moe.experts.{j}.{gate,up,down}_proj.weight
  - thinker.model.layers.{i}.mlp.{gate,up,down}_proj.weight (dense layers)
  - thinker.model.norm.weight
  - thinker.lm_head.weight
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import ExpertParallelSparseMoeBlock, ParallelSparseMoeBlock, RMSNorm
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_KV, THINKER_POS, Qwen3OmniModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThinkerMoeParallelConfig:
    """How the Thinker's routed experts split across its node group.

    Read from the deployment yaml's ``model_kwargs.thinker_moe``. Attention
    and dense MLPs stay TP-sharded either way.
    """

    # "tp": every rank holds a slice of every expert; "ep": whole experts per rank.
    parallel: str = "tp"
    # Comm group EP spans. Only "tp", the node group's TP ranks, exists today.
    ep_group: str = "tp"
    # Temporary: device-assert every rank routes to the same experts.
    debug_check_routing: bool = False
    # Log per-rank EP slot counts every N Thinker steps; 0 is off.
    log_expert_load_every: int = 0

    def __post_init__(self):
        if self.parallel not in ("tp", "ep"):
            raise ValueError(f"thinker_moe.parallel must be 'tp' or 'ep', got {self.parallel!r}")
        if self.ep_group != "tp":
            raise ValueError(f"thinker_moe.ep_group must be 'tp', got {self.ep_group!r}")
        if self.log_expert_load_every < 0:
            raise ValueError("thinker_moe.log_expert_load_every must be >= 0")
        if self.parallel != "ep" and (self.debug_check_routing or self.log_expert_load_every):
            raise ValueError(
                "thinker_moe.debug_check_routing and log_expert_load_every need parallel: ep"
            )

    @classmethod
    def from_yaml(cls, section: dict | None) -> "ThinkerMoeParallelConfig":
        section = dict(section or {})
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown thinker_moe keys: {sorted(unknown)}")
        return cls(**section)

    def resolve_ep_group(self, tp_group: CommGroup | None) -> CommGroup | None:
        # Only one choice for now; new mesh axes would be selected here.
        return tp_group


class Qwen3OmniThinkerLayer(nn.Module):
    """Single Thinker decoder layer: attention + MoE/dense MLP.

    Uses ``ParallelSparseMoeBlock``, or ``ExpertParallelSparseMoeBlock``
    under ``moe_parallel.parallel == "ep"``, for most layers, and a dense
    ``ParallelGatedMLP`` (SiLU SwiGLU) for layers in ``mlp_only_layers``.

    Args:
        config: top-level Qwen3-Omni model configuration.
        layer_idx: index of this layer in the stack.
        comm_group: TP communication group for MoE/MLP sharding.
        moe_parallel: TP or EP for the routed experts; TP when None.
    """

    def __init__(
        self, config: Qwen3OmniModelConfig, layer_idx: int,
        comm_group: CommGroup | None = None,
        moe_parallel: ThinkerMoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text
        moe_parallel = moe_parallel or ThinkerMoeParallelConfig()

        self.hidden_size = tc.hidden_size

        # Pre-attention layernorm
        self.input_layernorm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

        # Self-attention with QK-norm and 3D MRoPE
        self.self_attn = Qwen3OmniAttention(
            hidden_size=tc.hidden_size,
            num_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            rope_theta=tc.rope_theta,
            rms_norm_eps=tc.rms_norm_eps,
            use_mrope=True,
            comm_group=comm_group,
            attn_key=THINKER_ATTN,
            kv_key=THINKER_KV,
            pos_key=THINKER_POS
        )

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )

        # MoE or dense MLP depending on layer index.
        use_moe = (
            layer_idx not in tc.mlp_only_layers
            and tc.num_experts > 0
            and (layer_idx + 1) % tc.decoder_sparse_step == 0
        )
        if use_moe and moe_parallel.parallel == "ep":
            ep_group = moe_parallel.resolve_ep_group(comm_group)
            self.mlp = ExpertParallelSparseMoeBlock(
                hidden_size=tc.hidden_size,
                moe_intermediate_size=tc.moe_intermediate_size,
                num_experts=tc.num_experts,
                num_experts_per_tok=tc.num_experts_per_tok,
                norm_topk_prob=tc.norm_topk_prob,
                comm_group=ep_group,
                debug_check_routing=moe_parallel.debug_check_routing,
                # Counts are identical on every rank, so only rank 0 keeps them.
                track_expert_load=(
                    moe_parallel.log_expert_load_every > 0
                    and (ep_group is None or ep_group.rank == 0)
                ),
            )
        elif use_moe:
            self.mlp = ParallelSparseMoeBlock(
                hidden_size=tc.hidden_size,
                moe_intermediate_size=tc.moe_intermediate_size,
                num_experts=tc.num_experts,
                num_experts_per_tok=tc.num_experts_per_tok,
                norm_topk_prob=tc.norm_topk_prob,
                comm_group=comm_group,
            )
        else:
            self.mlp = ParallelGatedMLP(
                hidden_size=tc.hidden_size,
                intermediate_size=tc.intermediate_size,
                activation="silu",
                comm_group=comm_group,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE.

        Returns:
            hidden_states: [tokens, hidden_size]
        """
        # Pre-attention norm + self-attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
        )
        hidden_states = residual + hidden_states

        # Post-attention norm + MLP/MoE + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniThinkerTextModel(nn.Module):
    """Inner text model (maps to ``thinker.model.*`` in HF weights)."""

    def __init__(
        self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None,
        moe_parallel: ThinkerMoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3OmniThinkerLayer(config, layer_idx=i, comm_group=comm_group, moe_parallel=moe_parallel)
            for i in range(tc.num_hidden_layers)
        ])
        self.norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)


class Qwen3OmniThinkerModel(nn.Module):
    """Thinker: MoE transformer backbone for Qwen3-Omni.

    HF weight layout::

        thinker.model.embed_tokens.weight
        thinker.model.layers.{i}.*
        thinker.model.norm.weight
        thinker.lm_head.weight

    Produces:
    - Final hidden states (after all layers + final norm) for text logits
    - Layer-0 embeddings (before any transformer layers) for Talker conditioning
    - Layer-N hidden states (``accept_hidden_layer``) for Talker conditioning
    """

    def __init__(
        self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None,
        moe_parallel: ThinkerMoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size
        self.num_layers = tc.num_hidden_layers
        self.accept_hidden_layer = config.accept_hidden_layer

        self.model = Qwen3OmniThinkerTextModel(config, comm_group=comm_group, moe_parallel=moe_parallel)

        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

        self._log_expert_load_every = (moe_parallel or ThinkerMoeParallelConfig()).log_expert_load_every
        self._steps_since_load_log = 0
        self._discard_expert_load = False

    def _tracked_moe_layers(self) -> list[tuple[int, nn.Module]]:
        return [
            (i, layer.mlp) for i, layer in enumerate(self.model.layers)
            if getattr(layer.mlp, "track_expert_load", False)
        ]

    @torch.compiler.disable
    def maybe_log_expert_load(self, synthetic: bool = False) -> None:
        """Before each Thinker forward: after every N forwards, log EP load and reset it.

        Call from eager code. ``synthetic`` marks a
        warmup or capture step; its dummy routing is discarded before the
        next real step. Under CUDA graphs real counts include padding rows.
        """
        if not self._log_expert_load_every:
            return
        if synthetic:
            self._discard_expert_load = True
            return
        if self._discard_expert_load:
            self._discard_expert_load = False
            self._steps_since_load_log = 0
            for _, mlp in self._tracked_moe_layers():
                mlp.expert_load.zero_()
        # Counts forwards already run; this step's forward comes after.
        steps = self._steps_since_load_log
        self._steps_since_load_log += 1
        if steps < self._log_expert_load_every:
            return
        self._steps_since_load_log = 1
        layers = self._tracked_moe_layers()
        if not layers:
            return
        world_size = layers[0][1].comm_group.world_size
        # (layers, experts) -> (layers, ranks)
        per_expert = torch.stack([mlp.pop_expert_load() for _, mlp in layers])
        per_rank = per_expert.view(len(layers), world_size, -1).sum(-1)

        def imbalance(counts: torch.Tensor) -> float:
            mean = counts.double().mean().item()
            return counts.max().item() / mean if mean else 1.0

        totals = per_rank.sum(0)
        worst = max(range(len(layers)), key=lambda j: imbalance(per_rank[j]))
        hottest = per_expert.sum(0).topk(min(4, per_expert.shape[1]))
        logger.info(
            "Thinker EP load over %d steps, %d MoE layers: per-rank slots %s "
            "(max/mean %.2f); worst layer %d %s (max/mean %.2f); hottest experts %s",
            steps, len(layers), totals.tolist(), imbalance(totals),
            layers[worst][0], per_rank[worst].tolist(), imbalance(per_rank[worst]),
            dict(zip(hottest.indices.tolist(), hottest.values.tolist(), strict=True)),
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_embeds: torch.Tensor
    ):
        # NOTE: must ensure that visual_embeds is the same shape as hidden_states,
        # and zero where we do not have visual tokens!!
        hidden_states += visual_embeds
        return hidden_states

    def forward(
        self,
        input_embeds: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        label: str = "main",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_embeds: [tokens, hidden_size] -- pre-embedded input
                (token embeddings possibly merged with multimodal features).
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE,
                e.g. [24, 20, 20].
            label: the plan key every layer runs against. The stream's
                stored length and position counter advance when the runner
                commits the declared step (vision prefill declares its
                MRoPE 3D-grid span there).

        Returns:
            hidden_states: [tokens, hidden_size] -- final normed hidden states
            layer_0_embed: [tokens, hidden_size] -- input before any layers
            layer_n_hidden: [tokens, hidden_size] or None -- hidden states
                after ``accept_hidden_layer`` (for Talker conditioning)
        """
        hidden_states = input_embeds

        # Capture input embeddings BEFORE any transformer layers
        layer_0_embed = hidden_states.clone()
        layer_n_hidden = None

        # The label and layer index are cursors on the shared resources: bind
        # the label once, advance the index per layer. Passing them as
        # arguments instead would make inductor specialize on the int.
        self.model.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            decoder_layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states,
                cos_sin_3d=cos_sin_3d,
                mrope_section=mrope_section,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    deepstack_visual_embeds[layer_idx],
                )

            # Capture hidden states at the accept_hidden_layer for Talker
            if layer_idx == self.accept_hidden_layer:
                layer_n_hidden = hidden_states.clone()

        # Final layer norm
        hidden_states = self.model.norm(hidden_states)

        return hidden_states, layer_0_embed, layer_n_hidden
