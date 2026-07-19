"""Kimi-K2.7 / DeepSeek-V3 MLA attention — naive / materialized path.

MLA compresses q and k/v through low-rank latents, then (in the naive path)
projects the latent back up to full per-head K/V and runs ordinary attention.
This avoids the weight-absorbed path (``W_UK``/``W_UV``) and its bespoke kernel —
throughput caveat noted — so it drops straight onto mstar's paged
``run_attention`` ``[tokens, heads, head_dim]`` interface, matching vLLM's
``DeepseekV2Attention`` (the non-absorbed class).

Per-token shape story (H heads, Dnope=qk_nope, Drope=qk_rope, Dqk=Dnope+Drope,
Dv=v_head_dim, L=kv_lora_rank):
  - q: ``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj`` -> ``[T,H,Dqk]``, split
    into ``q_nope[..,Dnope]`` / ``q_pe[..,Drope]``.
  - kv: ``kv_a_proj_with_mqa`` -> ``[L | Drope]``; the ``L`` slice is RMS-normed
    and ``kv_b_proj``-ed to per-head ``[k_nope[..,Dnope] | v[..,Dv]]``; the trailing
    ``Drope`` slice is the single shared MQA rope key ``k_pe[T,1,Drope]``.
  - YARN RoPE rotates only ``q_pe`` (per head) and ``k_pe`` (broadcast to H heads).
  - assemble ``k = [k_nope | k_pe_broadcast] -> [T,H,Dqk]``; zero-pad ``q``/``k``
    (Dqk) and ``v`` (Dv) up to ``padded_head_dim`` (M6 mitigation — FlashInfer SM90
    rejects ``head_dim_vo`` not in {64,128,256}); fold the scale boost into ``q``
    (``run_attention`` uses the fixed ``1/sqrt(padded_head_dim)`` scale), attend,
    slice the output back to ``Dv``, ``o_proj``.

Cache config for this node: ``num_kv_heads == num_qo_heads == num_attention_heads``,
``head_dim == padded_head_dim`` (256 for the real Dqk=192, 64 for the reduced
Dqk=24). Weight-absorbed MLA (native latent dims, no pad) and the
``fused_qkv_a_proj`` weight fusion are deferred to the perf backlog.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k2_7.components.rope import KimiYarnRotaryEmbedding, yarn_get_mscale
from mstar.model.kimi_k2_7.config import KimiK2Config


class KimiMLAAttention(nn.Module):
    """Multi-head Latent Attention (naive/materialized)."""

    def __init__(self, config: KimiK2Config, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()

        # TODO(M6): shard num_heads across TP; naive path assumes local == total.
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        # FlashInfer SM90 rejects head_dim_vo not in {64,128,256}, so q/k/v are
        # zero-padded to this width for the paged run_attention (M6 mitigation);
        # the attention output is sliced back to v_head_dim. See config docstring.
        self.padded_head_dim = config.padded_head_dim
        h = self.num_heads

        # Q: two-stage low-rank (q_a down -> norm -> q_b up). Down-projections are
        # replicated (small rank); up-projections shard over heads under TP.
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)

        # KV: shared latent + decoupled rope key.
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = RowParallelLinear(
            comm_group, h * config.v_head_dim, config.hidden_size,
            bias=False, input_is_parallel=True, reduce_results=True)

        rope = config.rope_scaling
        self.rotary = KimiYarnRotaryEmbedding(
            rotary_dim=config.qk_rope_head_dim,
            base=config.rope_theta,
            factor=rope["factor"],
            original_max_position_embeddings=rope["original_max_position_embeddings"],
            beta_fast=rope.get("beta_fast", 32),
            beta_slow=rope.get("beta_slow", 1),
            mscale=rope.get("mscale", 1.0),
            mscale_all_dim=rope.get("mscale_all_dim", 0.0),
        )
        # Softmax-scale boost folded into q because run_attention applies a fixed
        # 1/sqrt(head_dim) scale and exposes no custom sm_scale. DeepSeek's intended
        # softmax scale is ``qk_head_dim**-0.5 * mscale**2``; run_attention now runs
        # over the PADDED head dim, so it uses ``padded_head_dim**-0.5``. The
        # zero-pad dims contribute 0 to q·k, so to recover the intended scale we
        # fold ``mscale**2 * sqrt(padded_head_dim / qk_head_dim)`` into q:
        #   scores = (q*boost)·k * padded_head_dim**-0.5
        #          = q·k * mscale**2 * sqrt(padded/qk) * padded**-0.5
        #          = q·k * mscale**2 * qk**-0.5   (the DeepSeek scale).
        mscale = yarn_get_mscale(rope["factor"], rope.get("mscale_all_dim", 0.0))
        self.softmax_scale_boost = (
            mscale * mscale * math.sqrt(self.padded_head_dim / self.qk_head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        h = self.num_heads

        # --- Q ---
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- KV latent ---
        latent = self.kv_a_proj_with_mqa(hidden_states)  # (T, L + Drope)
        kv_a, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_a))
        kv = kv.view(num_tokens, h, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # shared MQA rope key

        # --- RoPE (only the pe slices) ---
        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe)

        # --- assemble full q / k (k_pe broadcast over heads) ---
        q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk)
        k_pe = k_pe.expand(num_tokens, h, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (T, H, Dqk)

        # --- zero-pad q/k (Dqk) and v (Dv) up to padded_head_dim for the paged
        # run_attention (FlashInfer SM90 requires head_dim_vo in {64,128,256}) ---
        qk_pad = self.padded_head_dim - self.qk_head_dim
        q = F.pad(q, [0, qk_pad])  # (T, H, Dpad)
        k = F.pad(k, [0, qk_pad])  # (T, H, Dpad)
        v = F.pad(v, [0, self.padded_head_dim - self.v_head_dim])  # (T, H, Dpad)

        # --- softmax boost folded into q (compensates padded_head_dim scale),
        # attend, strip the pad + v-pad, project ---
        q = q * self.softmax_scale_boost
        attn = cache_handle.run_attention(q=q, k=k, v=v)  # (T, H, Dpad)
        attn = attn[..., : self.v_head_dim].reshape(num_tokens, h * self.v_head_dim)
        return self.o_proj(attn)
