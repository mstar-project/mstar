"""Gated NoPE Multi-head Latent Attention layer (spec D), tensor-parallel over query
heads; the 576-wide per-token latent is replicated (every rank writes the same cache).

Paged path: the layer is bound to an MLA KV resource (``kv_key``) and an MLA attention
resource (``attn_key``); it writes the latent through the KV resource and attends in the
absorbed form (``q_nope @ W_UK`` against ``[ckv | kpe]``, output ``@ W_UV``).
Dense path (``forward_dense``): explicit latent cache for one sequence, used by tests and
the eager reference.

``kv_b_proj`` is kept as a parameter (checkpoint name) and absorbed lazily into
``W_UK_T [H_local, nope, latent]`` / ``W_UV [H_local, latent, v]`` buffers on first use
after loading.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mstar.model.kimi_k3.components.common import KimiRMSNorm, ReplicatedLinear, replicated_loader


class ParallelMLAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        use_output_gate: bool = True,
        norm_eps: float = 1e-6,
        comm_group: CommGroup | None = None,
        attn_key: str = "mla_attn",
        kv_key: str = "mla_kv",
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp = comm_group.world_size
        assert num_heads % tp == 0
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale = self.qk_head_dim ** -0.5
        self.use_output_gate = use_output_gate
        self._attn_key, self._kv_key = attn_key, kv_key
        self.attn = None
        self.kv = None

        self.q_a_proj = ReplicatedLinear(hidden_size, q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(q_lora_rank, eps=norm_eps)
        self.q_b_proj = ColumnParallelLinear(comm_group, q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(kv_lora_rank, eps=norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False,
        )
        if use_output_gate:
            self.g_proj = ColumnParallelLinear(comm_group, hidden_size, num_heads * v_head_dim, bias=False)
        self.o_proj = RowParallelLinear(
            comm_group, num_heads * v_head_dim, hidden_size, bias=False, input_is_parallel=True, reduce_results=True,
        )
        self.q_a_proj.weight.weight_loader = replicated_loader
        self.kv_a_proj_with_mqa.weight.weight_loader = replicated_loader
        self._absorbed: tuple[torch.Tensor, torch.Tensor] | None = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.q_a_proj.weight.weight_loader = replicated_loader
        self.kv_a_proj_with_mqa.weight.weight_loader = replicated_loader
        self._absorbed = None
        return result

    def bind_resources(self, resources: dict) -> None:
        self.attn = resources.get(self._attn_key)
        self.kv = resources.get(self._kv_key)

    def absorb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(W_UK [H_local, nope, latent], W_UV [H_local, latent, v])`` from ``kv_b_proj``."""
        if self._absorbed is None:
            w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
            self._absorbed = (w_uk, w_uv)
        return self._absorbed

    # ---------------------------------------------------------------- pieces
    def _query(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.shape[0]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        w_uk, _ = self.absorb()
        q_lat = torch.einsum("thn,hnl->thl", q_nope, w_uk.to(q_nope.dtype))
        return q_lat, q_pe

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        ckv = self.kv_a_proj_with_mqa(x)
        c, k_pe = ckv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return torch.cat([self.kv_a_layernorm(c), k_pe], dim=-1)

    def _finish(self, x: torch.Tensor, o_lat: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        _, w_uv = self.absorb()
        attn = torch.einsum("thl,hlv->thv", o_lat, w_uv.to(o_lat.dtype)).reshape(t, self.num_heads * self.v_head_dim)
        if self.use_output_gate:
            attn = attn * torch.sigmoid(self.g_proj(x))
        return self.o_proj(attn)

    # ---------------------------------------------------------------- paths
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.attn is not None and self.kv is not None, "bind_resources first, or use forward_dense"
        q_lat, q_pe = self._query(x)
        self.kv.write_kv(self.latent(x))
        o_lat = self.attn.run(q_lat, kv_cache_layer=self.kv.layer_view(), q_pe=q_pe)
        return self._finish(x, o_lat)

    def forward_dense(
        self, x: torch.Tensor, latent_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal attention of the new tokens ``x [T, hidden]`` over ``latent_cache``
        (``[T_past, latent]``) plus themselves; returns ``(out, new_latents [T, latent])``."""
        t = x.shape[0]
        q_lat, q_pe = self._query(x)
        latent_new = self.latent(x)
        lat = latent_new if latent_cache is None else torch.cat([latent_cache, latent_new], 0)
        c, k_pe = lat.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        scores = (
            torch.einsum("qhl,kl->hqk", q_lat.float(), c.float())
            + torch.einsum("qhr,kr->hqk", q_pe.float(), k_pe.float())
        ) * self.scale
        offset = lat.shape[0] - t
        qi = torch.arange(t, device=x.device)[:, None]
        kj = torch.arange(lat.shape[0], device=x.device)[None, :]
        scores = scores.masked_fill(~(kj <= qi + offset)[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_lat = torch.einsum("hqk,kl->qhl", probs, c.float()).to(x.dtype)
        return self._finish(x, o_lat), latent_new


__all__ = ["ParallelMLAAttention", "F"]
