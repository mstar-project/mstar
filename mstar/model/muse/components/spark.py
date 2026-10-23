"""Muse Spark Open's text stack: Glimmer's attention, residual gates, and a
mixture of experts in a narrower latent.

Ported from transformers' ``modeling_muse_spark.py`` (Apache-2.0), with the
residual stream kept in fp32 across layers as SGLang and vLLM serve it (the
reference rounds it to bf16 at every gate). Per layer, on the stream ``s``:

    x = rms(s) * (1 + w_in)                        -> bf16, attention
    s = alpha1 * s + beta1 * rms(attn(x))          (weightless post-norm)
    h = rms(s) * (1 + w_ffn)                       -> bf16, router and experts
    s = alpha2 * s + beta2 * rms(moe(h) * (1 + w_post))

A residual gate ``g`` gives ``beta = sigmoid(g / T)`` and ``alpha =
sqrt(max(sigmoid(-g / T) * (1 + beta), 1e-3))``, computed from the bf16
weight in fp32 on every forward (8192 values, fused by Inductor).

The experts are the shared EP block over a 4096-wide latent, declared with
its options: ``pre_expert_proj`` and its norm before them (``pre_experts``),
``post_expert_proj`` after (``post_experts``), and each expert's output
RMS-normed before its routing weight applies (``expert_norm``). The experts
are split across the LLM's TP ranks whole (the node group's ``ep_config``):
a rank's partial latent sums its own experts' normed rows, and one
all-reduce over the latent combines them.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import (
    ExpertParallelSparseMoeBlock,
    InputScaledRMSNorm,
    MoeParallelConfig,
    MoeParallelMixin,
    NoAuxTcRouter,
    RMSNorm,
    instantiate_moe_block,
)
from mstar.model.components.norm import rms_norm_no_weight
from mstar.model.muse.components.text import (
    MuseGlimmerAttention,
    MuseGlimmerForCausalLM,
    MuseGlimmerTextModel,
)
from mstar.model.muse.config import MuseSparkConfig
from mstar.model.muse.submodules import LLMSubmodule


class MuseSparkAttention(MuseGlimmerAttention):
    """Glimmer's attention with each head's output RMS-normalized (no
    weight) before the output gate."""

    def _attend_heads(self, q, k, v) -> torch.Tensor:
        out = super()._attend_heads(q, k, v)
        return rms_norm_no_weight(out.reshape(-1, self.num_heads, self.head_dim), self.eps)


class MuseSparkResidualGate(nn.Module):
    """``alpha * stream + beta * branch``, per channel, on the fp32 stream."""

    def __init__(self, hidden_size: int, temperature: float):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.temperature = temperature

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        w = self.gate.float() / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
        return alpha * stream + beta * branch.float()


def spark_moe(
    config: MuseSparkConfig, comm_group: CommGroup, moe_parallel: MoeParallelConfig | None,
) -> ExpertParallelSparseMoeBlock:
    """A layer's experts, split whole across the node group's ranks: its
    ``ep_config`` must say ``parallel: ep`` (`MuseSparkModel.validate_config_yaml`
    refuses any other at startup)."""
    hidden, latent = config.hidden_size, config.moe_hidden_size
    return instantiate_moe_block(
        hidden_size=latent,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_intermediate_size=config.moe_intermediate_size,
        router=NoAuxTcRouter(
            hidden, config.num_local_experts, config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            logit_scale=config.output_multiplier, fp32_weight=True,
        ),
        comm_group=comm_group,
        moe_parallel=moe_parallel,
        # replicated on every rank: the experts read the whole latent
        pre_experts=nn.Sequential(OrderedDict(
            proj=nn.Linear(hidden, latent, bias=False),
            norm=RMSNorm(latent, eps=config.rms_norm_eps, gemma_mode=True),
        )),
        post_experts=nn.Linear(latent, hidden, bias=False),
        expert_norm=InputScaledRMSNorm(latent, eps=config.post_norm_eps),
        # Decode and every captured prefill bucket fit in one call.
        chunk_tokens=LLMSubmodule.PREFILL_TOKEN_BUCKETS[-1],
    )


class MuseSparkDecoderLayer(nn.Module):
    def __init__(
        self, config: MuseSparkConfig, sliding: bool, comm_group: CommGroup,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = MuseSparkAttention(config, sliding, comm_group)
        self.mlp = spark_moe(config, comm_group, moe_parallel)
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps, gemma_mode=True)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps, gemma_mode=True)
        self.post_feedforward_layernorm = InputScaledRMSNorm(hidden, eps=config.post_norm_eps)
        self.post_attention_residual_gate = MuseSparkResidualGate(hidden, config.residual_gate_temperature)
        self.post_feedforward_residual_gate = MuseSparkResidualGate(hidden, config.residual_gate_temperature)
        self.post_norm_eps = config.post_norm_eps

    def forward(
        self, stream: torch.Tensor, dtype: torch.dtype, token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The fp32 stream in and out; the sublayers run in ``dtype``."""
        a = self.self_attn(self.input_layernorm(stream).to(dtype))
        stream = self.post_attention_residual_gate(stream, rms_norm_no_weight(a, self.post_norm_eps))
        f = self.mlp(self.pre_feedforward_layernorm(stream).to(dtype), token_valid=token_valid)
        return self.post_feedforward_residual_gate(stream, self.post_feedforward_layernorm(f))


class MuseSparkTextModel(MoeParallelMixin, MuseGlimmerTextModel):
    LAYER = MuseSparkDecoderLayer

    def __init__(
        self, config: MuseSparkConfig, comm_group: CommGroup | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        super().__init__(config, comm_group, moe_parallel=moe_parallel)
        self.init_moe_parallel(moe_parallel)

    def forward(
        self, hidden: torch.Tensor, *, label: str, num_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``num_valid_tokens``: ``(1,)``, the real tokens ahead of a replay's
        padding rows, which the experts then skip (``ep_config.mask_padding``)."""
        for idx in self._bind_through:
            self.layers[idx].self_attn.attend.bind_step(label)
        token_valid = self.moe_token_valid(hidden, num_valid_tokens)
        stream = hidden.float()
        for i, layer in enumerate(self.layers):
            # every layer is a layer of the one cache, so its stack index
            layer.self_attn.attend.set_layer_idx(i)
            stream = layer(stream, hidden.dtype, token_valid)
        # both references normalize a bf16-rounded stream here
        return self.norm(stream.to(hidden.dtype))


class MuseSparkForCausalLM(MuseGlimmerForCausalLM):
    TEXT_MODEL = MuseSparkTextModel
