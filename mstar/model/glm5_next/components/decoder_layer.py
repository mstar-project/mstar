"""GLM-5.3-Flash decoder layers: the mHC hybrid trunk layer + the plain MTP layer."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm5_next.components.attention import (
    Glm5NextKdaAttention,
    Glm5NextMLAAttention,
)
from mstar.model.glm5_next.components.language_model import (
    build_mlp_for_layer,
    build_rmsnorm,
)
from mstar.model.glm5_next.config import KDA_STATE, LINEAR_ATTENTION, Glm5NextModelConfig
from mstar.model.glm5_next.kda_state import Glm5NextKdaStateAccess
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection, update_streams


def build_hyper_connection(config: Glm5NextModelConfig) -> Glm5NextHyperConnection:
    return Glm5NextHyperConnection(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        hc_eps=config.hc_eps,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        rms_norm_eps=config.rms_norm_eps,
    )


class Glm5NextDecoderLayer(nn.Module):
    """One mHC trunk layer over streams ``(1, T, hc_mult, hidden)``."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear_attention = config.layer_types[layer_idx] == LINEAR_ATTENTION
        # Bound once at load (bind_resources): the KDA layer's view of the
        # engine's slot-state resource. None on an MLA layer.
        self._kda: Glm5NextKdaStateAccess | None = None
        if self.is_linear_attention:
            self.self_attn = Glm5NextKdaAttention(config, comm_group=comm_group)
            # Position of this layer inside the KDA state pool's layer axis.
            self.kda_pos = config.kda_layer_indices.index(layer_idx)
            self.kv_plane = None
        else:
            # Compact plane map: 3 -> 0, 7 -> 1, ..., 43 -> 10. The attention
            # module addresses its own plane so KDA layers never touch KV.
            self.kv_plane = config.full_attn_layer_indices.index(layer_idx)
            self.self_attn = Glm5NextMLAAttention(
                config, comm_group=comm_group, layer_idx=layer_idx,
                kv_plane=self.kv_plane,
            )
            self.kda_pos = None
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)
        self.attn_hc = build_hyper_connection(config)
        self.ffn_hc = build_hyper_connection(config)

    def bind_resources(self, resources: dict) -> None:
        if not self.is_linear_attention:
            return
        self._kda = Glm5NextKdaStateAccess(resources[KDA_STATE])
        # The conv tail is cat'd onto the projected activations (kda.prefill),
        # so the pool the model declared must match the dtype the layer was
        # loaded in — a YAML autocast override without a matching
        # model_kwargs.kda_conv_dtype would otherwise promote the conv math
        # silently.
        proj_dtype = self.self_attn.q_proj.weight.dtype
        pool_dtype = self._kda.conv_dtype
        if pool_dtype != proj_dtype:
            raise RuntimeError(
                f"KDA layer {self.layer_idx}: slot-state conv pool is "
                f"{pool_dtype} but the KDA projections are {proj_dtype}; pass "
                "model_kwargs.kda_conv_dtype matching the serve dtype"
            )

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        residual = hidden_streams
        post, comb, hidden = self.attn_hc(hidden_streams)
        hidden = self.input_layernorm(hidden.squeeze(0))
        if self.is_linear_attention:
            hidden = self._run_kda(hidden)
        else:
            hidden = self.self_attn(hidden)
        hidden_streams = update_streams(residual, hidden.unsqueeze(0), post, comb)

        residual = hidden_streams
        post, comb, hidden = self.ffn_hc(hidden_streams)
        hidden = self.post_attention_layernorm(hidden.squeeze(0))
        hidden = self.mlp(hidden)
        return update_streams(residual, hidden.unsqueeze(0), post, comb)

    # -- KDA routing ------------------------------------------------------

    def _run_kda(self, hidden: torch.Tensor) -> torch.Tensor:
        """Phase-routed KDA over the flat ``(T, hidden)`` batch, by the
        slot-state resource's plan for this step.
        """
        if self._kda is None:
            raise RuntimeError(
                f"KDA layer {self.layer_idx} has no slot-state resource bound; "
                "the submodule's bind_node_resources must run before a forward"
            )
        plan = self._kda.current_plan()
        if plan.mode == "step":
            recurrent, conv = self._kda.gather(self.kda_pos, plan.slot_index)
            out = self.self_attn.decode_step(hidden.unsqueeze(1), recurrent, conv)
            self._kda.scatter(self.kda_pos, plan.slot_index, recurrent, conv)
            return out.squeeze(1)
        return self._run_kda_prefill(hidden, plan.spans)

    @torch.compiler.disable
    def _run_kda_prefill(self, hidden: torch.Tensor, spans) -> torch.Tensor:
        """Per-request chunked prefill against in-place slot views."""
        outputs = []
        for span in spans:
            if span.q_len == 0:
                continue
            chunk = hidden[span.q_start : span.q_start + span.q_len].unsqueeze(0)
            recurrent, conv = self._kda.state_views(self.kda_pos, span.slot)
            out, _, _ = self.self_attn.prefill(
                chunk, recurrent_state=recurrent, conv_state=conv)
            outputs.append(out.squeeze(0))
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


class Glm5NextPlainDecoderLayer(nn.Module):
    """Plain-residual NoPE MLA + MoE layer — the MTP layer-45 structure."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        layer_idx: int,
        kv_plane: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kv_plane = kv_plane
        self.self_attn = Glm5NextMLAAttention(
            config, comm_group=comm_group, layer_idx=layer_idx, kv_plane=kv_plane)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
