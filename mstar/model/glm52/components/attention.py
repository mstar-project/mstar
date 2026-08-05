"""GLM-5.2 MLA attention with absorbed and naive fallback paths.

Same structure as ``kimi_k2_7/components/attention.py`` (both are
DeepSeek-family MLA) with GLM geometry — q_lora 2048, nope 192, rope 64,
v_head 256, 64 heads — and plain (non-Yarn) RoPE, so every mscale term is 1.
qk_head_dim is exactly 256, so the FlashInfer pad on the naive path is a
no-op for the full model (reduced test configs still exercise it: 24 -> 64).

Phase C: FULL indexer layers (``is_full_indexer_layer``) instantiate a
``Glm52Indexer`` whose checkpoint weights now load; the naive path accepts
a precomputed ``dsa_selection`` and restricts softmax to the selected set
(reference semantics: masked dense attention, -1 entries excluded). The
absorbed/cache_handle sparse path — paged indexer k-cache, per-request
block-table index mapping — is the marked engine follow-up. At ctx <= 2048
top-2048 selection of <= 2048 tokens is the identity, so the dense path IS
the exact DSA computation in that regime.
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
from mstar.model.glm52.components.indexer import Glm52Indexer, is_full_indexer_layer
from mstar.model.glm52.components.rope import Glm52RotaryEmbedding
from mstar.model.glm52.config import Glm52ModelConfig


def dsa_selection_to_mask(
    dsa_selection: torch.Tensor, num_keys: int, dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``(T, num_keys)`` mask from per-query top-k rows (-1 = padding)."""
    num_tokens = dsa_selection.shape[0]
    mask = torch.full(
        (num_tokens, num_keys), float("-inf"), dtype=dtype, device=device)
    valid = dsa_selection >= 0
    rows = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(valid)
    mask[rows[valid], dsa_selection[valid].long()] = 0.0
    return mask


def masked_reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Dense ``(T, H, D)`` attention with an additive ``(T, T)`` mask.

    Matches ``run_attention`` semantics: scale is 1/sqrt(padded head dim)
    (any intended softmax-scale correction is pre-folded into ``q``).
    """
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale + mask
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


class Glm52MLAAttention(nn.Module):
    def __init__(
        self,
        config: Glm52ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()

        self.tp_size = comm_group.world_size
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} is not divisible by "
                f"tp_size={self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.padded_head_dim = config.padded_head_dim
        h = self.total_num_heads

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = RowParallelLinear(
            comm_group, h * config.v_head_dim, config.hidden_size,
            bias=False, input_is_parallel=True, reduce_results=True)

        self.rotary = Glm52RotaryEmbedding(
            rotary_dim=config.qk_rope_head_dim, base=config.rope_theta)

        # DSA indexer only on FULL layers — SHARED layers carry no indexer
        # weights in the checkpoint (they reuse the last FULL selection),
        # and a dormant module would sit uninitialized after ``to_empty``.
        self.indexer = (
            Glm52Indexer(config)
            if layer_idx is not None and is_full_indexer_layer(config, layer_idx)
            else None
        )
        # run_attention uses 1/sqrt(padded_head_dim); fold the intended
        # qk_head_dim**-0.5 into q on the padded path. No Yarn -> no mscale.
        self.softmax_scale_boost = math.sqrt(self.padded_head_dim / self.qk_head_dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        self.mla_absorb = config.mla_absorb
        if self.mla_absorb:
            self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
            self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
            self.register_buffer("fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L+Drope, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        dsa_selection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``dsa_selection``: optional ``(T, index_topk)`` int rows from
        ``Glm52Indexer.compute_selection`` (-1 = padding). When given, the
        naive path restricts softmax to the selected keys of the current
        batch; when None, dense behavior is unchanged. The absorbed path
        does not take a selection yet (engine follow-up)."""
        if self.mla_absorb:
            if dsa_selection is not None:
                raise NotImplementedError(
                    "dsa_selection on the absorbed path is the Phase C engine "
                    "follow-up; only the naive reference path consumes it"
                )
            return self._forward_absorbed(hidden_states, cache_handle, position_ids)
        num_tokens = hidden_states.shape[0]
        h = self.num_heads

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        latent = self.kv_a_proj_with_mqa(hidden_states)  # (T, L + Drope)
        kv_a, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_a))
        kv = kv.view(num_tokens, h, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # shared MQA rope key

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe)

        q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk)
        k_pe = k_pe.expand(num_tokens, h, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (T, H, Dqk)

        qk_pad = self.padded_head_dim - self.qk_head_dim
        q = F.pad(q, [0, qk_pad])  # (T, H, Dpad)
        k = F.pad(k, [0, qk_pad])  # (T, H, Dpad)
        v = F.pad(v, [0, self.padded_head_dim - self.v_head_dim])  # (T, H, Dpad)

        q = q * self.softmax_scale_boost
        if dsa_selection is None:
            attn = cache_handle.run_attention(q=q, k=k, v=v)  # (T, H, Dpad)
        else:
            # Reference sparse path: softmax over the selected keys only,
            # via an additive mask over the current batch (a gather of the
            # selected k/v rows is semantically identical; -1 excluded).
            mask = dsa_selection_to_mask(
                dsa_selection, num_tokens, dtype=q.dtype, device=q.device)
            attn = masked_reference_attention(q, k, v, mask)  # (T, H, Dpad)
        attn = attn[..., : self.v_head_dim].reshape(num_tokens, h * self.v_head_dim)
        return self.o_proj(attn)

    def _forward_absorbed(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run MLA over the compressed latent cache after folding kv_b into Q/O."""
        if self.w_kc is None or self.w_vc is None:
            raise RuntimeError(
                "mla_absorb forward requires process_weights_after_loading() to "
                "have built w_kc/w_vc from kv_b_proj first"
            )
        if self.fused_qkv_a_proj_weight is None:
            raise RuntimeError(
                "mla_absorb forward requires process_weights_after_loading() to "
                "have built fused_qkv_a_proj_weight first"
            )
        num_tokens = hidden_states.shape[0]
        h = self.num_heads

        fused = F.linear(hidden_states, self.fused_qkv_a_proj_weight)
        q_c, kv_a, k_pe = fused.split(
            [self.q_a_proj.out_features, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # FlashInfer RMSNorm needs 64-byte input alignment; decode split views can
        # be contiguous yet start at an unaligned offset, so clone kv_a.
        q_c = q_c.contiguous()
        kv_a = kv_a.clone(memory_format=torch.contiguous_format)

        q = self.q_b_proj(self.q_a_layernorm(q_c))
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)  # (T,1,L)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # (T,1,Drope) shared MQA key

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe)

        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        attn_latent = cache_handle.run_attention_mla(
            q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe)

        out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
        return self.o_proj(out.reshape(num_tokens, h * self.v_head_dim))

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build absorbed Q/O projections from the local-head ``kv_b_proj`` shard."""
        if not self.mla_absorb:
            return
        del device  # protocol arg; kv_b_proj.weight already carries the right device
        w = self.kv_b_proj.weight  # (H_local*(Dnope+Dv), L)
        h, d_nope, d_v, latent = (
            self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank)
        w = w.view(h, d_nope + d_v, latent)
        w_kc, w_vc = w.split([d_nope, d_v], dim=1)
        self.w_kc = w_kc.contiguous()  # (H_local, Dnope, L)
        self.w_vc = w_vc.contiguous()  # (H_local, Dv,    L)

        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0).contiguous()
