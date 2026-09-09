"""Assembled GLM-5.3-Flash text backbone (45-layer hybrid + mHC + KDA state).

Forked from ``glm52/components/causal_lm.py``. Deltas: no rotary (NoPE —
no cos/sin computed or threaded), the residual enters as ``hc_mult``
replicated streams and exits through the unweighted ``hc_head`` mean, and
the per-layer cache planes are compact full-attention indices (each layer
addresses its own).

Engine forward protocol (resource-pool engine): flattened token batch in,
``(T, hidden)`` out. Every layer reaches its engine resources through the
references it bound at load (``bind_resources``): the MLA layers the KV +
attention resources, the KDA layers the slot-state resource whose per-step
plan says which slot each row reads. The engine plans before and commits
after the forward, so the model advances nothing itself.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm5_next.components.decoder_layer import Glm5NextDecoderLayer
from mstar.model.glm5_next.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.mhc import Glm5NextHyperHead, expand_streams


class Glm5NextLanguageModel(nn.Module):
    def __init__(
        self, config: Glm5NextModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        if not config.mhc:
            raise ValueError(
                "glm5_next is assembled mHC-only (the checkpoint always has "
                "hc tensors on layers 0..44); mhc=False has no layer shape"
            )
        self.hc_mult = config.hc_mult
        self.embed_tokens = build_embedding(config, comm_group=comm_group)
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = build_rmsnorm(config)
        # Unweighted stream mean (explicitly unlike DeepSeek-V4's weighted
        # head; the checkpoint ships no head weights). Parameter-free.
        self.hc_head = Glm5NextHyperHead()

    def forward(
        self,
        input_ids: torch.Tensor,
        return_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``input_ids (T,)`` flat batch -> ``(T, hidden)`` final hidden.

        The KDA layers read the slot-state resource's plan for this step;
        the full-attention layers address their own compact KV plane.

        ``return_prenorm``: additionally return the ``hc_head``-collapsed
        stream BEFORE the final norm as ``(normed, prenorm)``. Which stream
        the MTP ``hnorm`` pairs against is an OPEN three-way A/B for M2
        (post-``hc_head`` pre-norm — returned here — vs post-norm vs a
        single stream); port glm52's env-switch pattern, do not assume.
        """
        hidden_states = self.embed_tokens(input_ids)
        # (1, T, hc_mult, hidden): embedding replicated into all streams.
        # B=1 over the flat token batch is exact — mHC is per-token.
        hidden_streams = expand_streams(hidden_states.unsqueeze(0), self.hc_mult)
        for decoder_layer in self.layers:
            hidden_streams = decoder_layer(hidden_streams)
        collapsed = self.hc_head(hidden_streams).squeeze(0)
        normed = self.norm(collapsed)
        if return_prenorm:
            return normed, collapsed
        return normed


class Glm5NextForCausalLM(nn.Module):
    def __init__(
        self,
        config: Glm5NextModelConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Glm5NextLanguageModel(config, comm_group=comm_group)
        self.lm_head = build_lm_head(config, comm_group=comm_group)
        # Per-request KDA state is the engine's slot-state resource
        # (get_node_resources declares it; kda_state.py is the model's view).
        # The layer-45 draft module exists only when drafting is on, so
        # flag-off keeps the parameter set (and load) byte-identical to a
        # no-MTP build (glm52 flag pattern).
        self.mtp = None
        if config.mtp_num_draft_tokens > 0:
            from mstar.model.glm5_next.components.mtp import Glm5NextMTPModule

            self.mtp = Glm5NextMTPModule(config, comm_group=comm_group)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)

    def kda_conv_dtype(self) -> torch.dtype:
        """The KDA projection dtype: the slot-state conv tail must match it
        (the continue path ``cat``s the tail onto the projected activations
        bit-exactly). Read off the first KDA layer's q_proj."""
        for layer in self.model.layers:
            if layer.is_linear_attention:
                return layer.self_attn.q_proj.weight.dtype
        raise RuntimeError("no KDA layer in the schedule")  # unreachable by config

    def load_weights(self, weights, **kwargs) -> set[str]:
        from mstar.model.glm5_next.weight_loader import load_glm5_next_hf_weights

        loaded = load_glm5_next_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant_config=self.config.quantization_config,
            fp8_experts=(
                self.config.quantization_config is not None
                and self.config.moe_fp8_resident
            ),
            num_hidden_layers=self.config.num_hidden_layers,
            load_mtp=self.mtp is not None,
        )
        if self.mtp is not None:
            missing = {
                f"mtp.{name}" for name, _ in self.mtp.named_parameters()
            } - loaded
            if missing:
                raise RuntimeError(
                    f"MTP drafting is on but {len(missing)} mtp.* "
                    f"parameters received no checkpoint weights (e.g. "
                    f"{sorted(missing)[:3]}) — the weight stream is "
                    "missing the layer-45 keys, typically a read plan "
                    "built without load_mtp. Drafting from uninitialized "
                    "memory is silent 0.00 acceptance; refuse to serve."
                )
        return loaded
