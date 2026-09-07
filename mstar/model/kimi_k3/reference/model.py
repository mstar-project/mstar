"""Whole-model reference forward (spec B): embedding, the 93-layer loop with Block
Attention Residuals, output mixing, final norm, lm_head. Supports incremental decode via
explicit per-layer state (KDA conv/recurrent state, MLA latent cache); Attention
Residuals never persist across steps.

Weights are taken from an HF ``KimiLinearForCausalLM`` module tree (``from_hf_model``), so
this file validates the *forward logic* independently of any loader.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.model.kimi_k3.config import KimiK3TextConfig
from mstar.model.kimi_k3.reference.attn_res import attn_res_read, attn_res_score_weight
from mstar.model.kimi_k3.reference.kda import KDAState, KDAWeights, kda_layer_forward
from mstar.model.kimi_k3.reference.mla import MLAWeights, mla_forward_dense, rms_norm
from mstar.model.kimi_k3.reference.moe import LatentMoEWeights, latent_moe_forward, shared_experts_mlp


@dataclass
class DenseMLPWeights:
    gate_up: torch.Tensor  # [2*inter, hidden]
    down: torch.Tensor  # [hidden, inter]
    situ_beta: float = 4.0
    situ_linear_beta: float | None = 25.0


@dataclass
class LayerWeights:
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    sa_res_w: torch.Tensor | None  # folded score weight for the pre-attention read
    mlp_res_w: torch.Tensor | None  # folded score weight for the pre-FFN read
    attn: KDAWeights | MLAWeights
    ffn: LatentMoEWeights | DenseMLPWeights


@dataclass
class ModelWeights:
    embed: torch.Tensor  # [V, H]
    layers: list[LayerWeights]
    output_res_w: torch.Tensor | None
    norm: torch.Tensor
    lm_head: torch.Tensor  # [V, H]
    config: KimiK3TextConfig

    @classmethod
    def from_hf_model(cls, hf, cfg: KimiK3TextConfig, expert_dtype: torch.dtype = torch.float32) -> ModelWeights:
        core = hf.model
        layers = []
        for i, layer in enumerate(core.layers):
            attn = KDAWeights.from_hf_module(layer.self_attn) if cfg.is_kda_layer(i) \
                else MLAWeights.from_hf_module(layer.self_attn)
            if cfg.is_moe_layer(i):
                ffn = LatentMoEWeights.from_hf_module(layer.block_sparse_moe, dtype=expert_dtype)
            else:
                mlp = layer.mlp
                ffn = DenseMLPWeights(
                    gate_up=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0),
                    down=mlp.down_proj.weight,
                    situ_beta=getattr(mlp.act_fn, "beta", 1.0),
                    situ_linear_beta=getattr(mlp.act_fn, "linear_beta", None),
                )
            use_res = cfg.use_attn_res
            sa_res_w = mlp_res_w = None
            if use_res:
                sa_res_w = attn_res_score_weight(
                    layer.self_attention_res_norm.weight, layer.self_attention_res_proj.weight)
                mlp_res_w = attn_res_score_weight(layer.mlp_res_norm.weight, layer.mlp_res_proj.weight)
            layers.append(LayerWeights(
                input_layernorm=layer.input_layernorm.weight,
                post_attention_layernorm=layer.post_attention_layernorm.weight,
                sa_res_w=sa_res_w, mlp_res_w=mlp_res_w, attn=attn, ffn=ffn,
            ))
        return cls(
            embed=core.embed_tokens.weight,
            layers=layers,
            output_res_w=attn_res_score_weight(
                core.output_attn_res_norm.weight, core.output_attn_res_proj.weight,
            ) if cfg.use_attn_res else None,
            norm=core.norm.weight,
            lm_head=hf.lm_head.weight,
            config=cfg,
        )


@dataclass
class ModelState:
    """Per-request state: one entry per layer (KDAState for KDA layers, latent cache
    ``[T_past, 576]`` for MLA layers)."""
    per_layer: list[KDAState | torch.Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> ModelState:
        return cls([None] * num_layers)


def reference_forward(
    w: ModelWeights,
    input_ids: torch.Tensor,
    state: ModelState | None = None,
    return_hidden: bool = False,
) -> tuple[torch.Tensor, ModelState]:
    """``input_ids [T]`` (new tokens) -> ``(logits [T, V], new_state)``."""
    cfg = w.config
    state = state or ModelState.empty(cfg.num_hidden_layers)
    new_state = ModelState.empty(cfg.num_hidden_layers)
    h = w.embed[input_ids]  # [T, H]
    t, hidden = h.shape
    eps = cfg.rms_norm_eps
    prefix = h
    blocks = h.new_zeros(t, 0, hidden)
    for i, layer in enumerate(w.layers):
        # pre-attention read (identity when the stack is empty)
        x = attn_res_read(prefix, blocks, layer.sa_res_w, eps) if cfg.use_attn_res else prefix
        if cfg.is_attn_res_block_start(i):
            blocks = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
            prefix = None
        x = rms_norm(x, layer.input_layernorm, eps)
        if cfg.is_kda_layer(i):
            a, st = kda_layer_forward(layer.attn, x, state.per_layer[i])
            new_state.per_layer[i] = st
        else:
            a, latent_new = mla_forward_dense(layer.attn, x, state.per_layer[i])
            prev = state.per_layer[i]
            new_state.per_layer[i] = latent_new if prev is None else torch.cat([prev, latent_new], 0)
        prefix = a if prefix is None else prefix + a
        if cfg.use_attn_res:
            x = attn_res_read(prefix, blocks, layer.mlp_res_w, eps)
        else:
            x = prefix
        x = rms_norm(x, layer.post_attention_layernorm, eps)
        if isinstance(layer.ffn, LatentMoEWeights):
            f = latent_moe_forward(layer.ffn, x)
        else:
            f = shared_experts_mlp(
                x, layer.ffn.gate_up, layer.ffn.down, layer.ffn.situ_beta, layer.ffn.situ_linear_beta)
        prefix = prefix + f
    out = attn_res_read(prefix, blocks, w.output_res_w, eps) if cfg.use_attn_res else prefix + 0
    out = rms_norm(out, w.norm, eps)
    if return_hidden:
        return out, new_state
    logits = F.linear(out, w.lm_head)
    return logits, new_state


@torch.no_grad()
def reference_generate(
    w: ModelWeights, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None,
) -> list[int]:
    """Greedy decoding with incremental state; returns the generated ids."""
    logits, state = reference_forward(w, prompt_ids)
    out: list[int] = []
    nxt = int(logits[-1].argmax())
    for _ in range(max_new_tokens):
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        logits, state = reference_forward(w, torch.tensor([nxt], device=prompt_ids.device), state)
        nxt = int(logits[-1].argmax())
    return out
