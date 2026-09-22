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
from mstar.model.components.distributed.merged_linear import COLUMN, REPLICATED, MergedParallelLinear
from mstar.model.kimi_k3.components.common import KimiRMSNorm, fused_decode_kernels
from mstar.model.kimi_k3.components.mla_out_kernel import mla_out
from mstar.utils.streams import Fork

# rows up to which the one-launch output kernel beats the einsum path (bench/kernels/mla_out_crossover.py)
MLA_OUT_MAX_ROWS = 32

# the checkpoint's q_a_proj / kv_a_proj_with_mqa land in the merged in_proj by segment name; its
# g_proj goes through the same ``(".in_proj", ".g_proj", "g")`` rule as the KDA layers'
MLA_IN_PROJ_PARAMS = [(".in_proj", ".q_a_proj", "q_a"), (".in_proj", ".kv_a_proj_with_mqa", "kv_a")]


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
        self._fork = Fork()

        # the three projections of the layer input (the replicated LoRA-A factors of q and kv,
        # the column-parallel output gate) run as one GEMM; see MergedParallelLinear
        segments = [("q_a", q_lora_rank, REPLICATED), ("kv_a", kv_lora_rank + qk_rope_head_dim, REPLICATED)]
        if use_output_gate:
            segments.append(("g", num_heads * v_head_dim, COLUMN))
        self.in_proj = MergedParallelLinear(comm_group, hidden_size, segments)
        self.q_a_layernorm = KimiRMSNorm(q_lora_rank, eps=norm_eps)
        self.q_b_proj = ColumnParallelLinear(comm_group, q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(kv_lora_rank, eps=norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False,
        )
        self.o_proj = RowParallelLinear(
            comm_group, num_heads * v_head_dim, hidden_size, bias=False, input_is_parallel=True, reduce_results=True,
        )
        self._absorbed: tuple[torch.Tensor, torch.Tensor] | None = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
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
    def _project(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The input projections in one GEMM: ``q_a``, ``kv_a`` (both replicated) and ``g``
        (column-parallel, when the output gate is on); views of the merged output."""
        return self.in_proj.project(x)

    def _query(self, q_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = q_a.shape[0]
        q = self.q_b_proj(self.q_a_layernorm(q_a)).view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        w_uk, _ = self.absorb()
        q_lat = torch.einsum("thn,hnl->thl", q_nope, w_uk.to(q_nope.dtype))
        return q_lat, q_pe

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """The new tokens' compressed latents ``[T, kv_lora_rank + rope]`` from the layer input."""
        return self._latent(self._project(x)["kv_a"])

    def _latent(self, ckv: torch.Tensor) -> torch.Tensor:
        c, k_pe = ckv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return torch.cat([self.kv_a_layernorm(c), k_pe], dim=-1)

    def _finish(self, g: torch.Tensor | None, o_lat: torch.Tensor) -> torch.Tensor:
        t = o_lat.shape[0]
        _, w_uv = self.absorb()
        w_uv = w_uv.to(o_lat.dtype)
        gate = g if self.use_output_gate else None
        if fused_decode_kernels() and t <= MLA_OUT_MAX_ROWS and o_lat.is_cuda \
                and o_lat.dtype in (torch.bfloat16, torch.float16) and o_lat.stride(-1) == 1 and w_uv.stride(-1) == 1:
            # one launch: the per-head product and the gate, written as the [T, H * V] matrix o_proj
            # reads (the einsum's head-major result costs a copy for more than one row). Only for a
            # few rows: the kernel is one program per (row, head), so its time grows with the rows
            # (about 1 us a row at 12 heads) while the einsum path stays at some 35 us up to a
            # thousand rows. Measured on an H100: even at 32 rows, 25 times slower at 1024 (a prefill
            # step ran it at 7 ms a layer before this bound).
            return self.o_proj(mla_out(o_lat, w_uv, gate))
        attn = torch.einsum("thl,hlv->thv", o_lat, w_uv).reshape(t, self.num_heads * self.v_head_dim)
        if gate is not None:
            attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn)

    # ---------------------------------------------------------------- paths
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.attn is not None and self.kv is not None, "bind_resources first, or use forward_dense"
        mixed = self._project(x)
        # the query path and the latent write only share the projection: two streams under a capture
        (q_lat, q_pe), _ = self._fork.run(
            lambda: self._query(mixed["q_a"]), lambda: self.kv.write_kv(self._latent(mixed["kv_a"])))
        o_lat = self.attn.run(q_lat, kv_cache_layer=self.kv.layer_view(), q_pe=q_pe)
        return self._finish(mixed.get("g"), o_lat)

    def forward_dense(
        self, x: torch.Tensor, latent_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal attention of the new tokens ``x [T, hidden]`` over ``latent_cache``
        (``[T_past, latent]``) plus themselves; returns ``(out, new_latents [T, latent])``."""
        t = x.shape[0]
        mixed = self._project(x)
        q_lat, q_pe = self._query(mixed["q_a"])
        latent_new = self._latent(mixed["kv_a"])
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
        return self._finish(mixed.get("g"), o_lat), latent_new


__all__ = ["MLA_IN_PROJ_PARAMS", "ParallelMLAAttention", "F"]
