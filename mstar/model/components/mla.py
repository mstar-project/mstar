"""Multi-head latent attention (DeepSeek-V2/V3's MLA) over the engine's paged KV cache.

``config`` carries the MLA dims: ``hidden_size``, ``num_attention_heads``, ``q_lora_rank``,
``kv_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``qk_head_dim``, ``v_head_dim``,
``padded_head_dim``, ``rms_norm_eps``, ``softmax_scale``, ``mla_absorb`` (the backend), and
optionally ``mla_fused_prep`` and the fp8 block-scale fields ``dense_fp8_block`` reads:

* absorbed: once the weights load, ``kv_b_proj`` folds into the query (``w_kc``) and the output
  (``w_vc``), and the two down projections into one GEMM. Attention runs over the cache's compressed
  rows (the ``kv_lora_rank`` latent and the decoupled RoPE key) through the MLA attention resource.
* naive: per-head K/V padded to the attention backend's head size, the reduced-dims parity path.

The decoupled RoPE is the position resource's (``pos_key``; the model's ``PositionConfig`` sets
its layout and table), at the positions the caller passes. Optional ``config`` fields:

* ``mla_use_nope``: no rotation, and no position resource (``pos_key=None``). With
  ``qk_rope_head_dim`` > 0 the decoupled dims are kept unrotated (Kimi K3); with 0 there are none
  (GLM-5.3), and the cache row's rope slot, ``mla_cache_kpe`` wide (0 or the 64 FlashInfer's MLA
  kernel may need), is zeros. Its query part is ``kpe_query_fill``.
* ``mla_use_output_gate``: ``o_proj`` reads the attention output times ``sigmoid(g_proj(x))``
  (Kimi K3).
* ``mla_latent_norm_eps``: the q and kv latent norms' eps, where it isn't ``rms_norm_eps``
  (DeepSeek-V3's and Kimi's references build them at 1e-6).

``kv_label`` names the cache stream a multi-label model writes and attends this layer under, and
``cache_layer`` the KV layer, where a model packs its attention layers' planes. A model's sparse
attention plugs in by overriding ``_select`` and ``_attend_selected`` (absorbed) or ``_attend_naive``
(naive); the keyword arguments ``forward`` takes beyond its own reach them unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.fp8_linear import COLUMN, ROW, Fp8Linear, dense_fp8_block, fp8_block_linear
from mstar.model.components.norm import RMSNorm
from mstar.utils import mla_prep
from mstar.utils.mla_latent_norm import mla_latent_norm, mla_latent_norm_available


class MLAAttention(nn.Module):
    def __init__(
        self,
        config,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        *,
        kv_key: str,
        attn_key: str,
        pos_key: str | None,
        kv_label: str | None = None,
        cache_layer: int | None = None,
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
        self.use_rope = not getattr(config, "mla_use_nope", False)
        if self.use_rope and (pos_key is None or self.qk_rope_head_dim == 0):
            raise ValueError("MLAAttention rotates its decoupled dims through a position "
                             "resource: pass pos_key and qk_rope_head_dim, or set mla_use_nope")
        # the rope slot of a cache row: the decoupled key, or zeros without one
        self.mla_cache_kpe = getattr(config, "mla_cache_kpe", self.qk_rope_head_dim)
        if self.qk_rope_head_dim and self.mla_cache_kpe != self.qk_rope_head_dim:
            raise ValueError(f"mla_cache_kpe={self.mla_cache_kpe} must be qk_rope_head_dim="
                             f"{self.qk_rope_head_dim} when the decoupled dims exist")
        # the query's rope part against a zero slot; a model may mark rows with that slot
        self.kpe_query_fill = 0.0
        latent_eps = getattr(config, "mla_latent_norm_eps", None)
        if latent_eps is None:
            latent_eps = config.rms_norm_eps

        # dense_fp8: the checkpoint's fp8 weights and block scales, except kv_b_proj, which
        # the absorbed path folds into bf16 w_kc / w_vc.
        self.fp8_block = fp8 = dense_fp8_block(config)
        kv_a_out = config.kv_lora_rank + config.qk_rope_head_dim

        self.q_a_proj = (
            nn.Linear(config.hidden_size, config.q_lora_rank, bias=False) if fp8 is None
            else Fp8Linear(config.hidden_size, config.q_lora_rank, fp8))
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=latent_eps)
        self.q_b_proj = (
            ColumnParallelLinear(comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)
            if fp8 is None
            else Fp8Linear(config.q_lora_rank, h * self.qk_head_dim, fp8, comm_group,
                           shard=COLUMN))

        self.kv_a_proj_with_mqa = (
            nn.Linear(config.hidden_size, kv_a_out, bias=False) if fp8 is None
            else Fp8Linear(config.hidden_size, kv_a_out, fp8))
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=latent_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = (
            RowParallelLinear(
                comm_group, h * config.v_head_dim, config.hidden_size,
                bias=False, input_is_parallel=True, reduce_results=True) if fp8 is None
            else Fp8Linear(h * config.v_head_dim, config.hidden_size, fp8, comm_group,
                           shard=ROW, reduce_results=True))
        self.g_proj = (
            ColumnParallelLinear(comm_group, config.hidden_size, h * config.v_head_dim, bias=False)
            if getattr(config, "mla_use_output_gate", False) else None)

        self.layer_idx = layer_idx
        # run_attention uses 1/sqrt(padded_head_dim); fold the intended
        # softmax_scale into q on the padded path.
        self.softmax_scale = config.softmax_scale
        self.softmax_scale_boost = math.sqrt(self.padded_head_dim) * self.softmax_scale

        self.mla_absorb = config.mla_absorb
        self.mla_fused_prep = (
            getattr(config, "mla_fused_prep", False) and self.mla_absorb and self.use_rope
            and mla_prep.supports(config.q_lora_rank, config.kv_lora_rank, self.qk_rope_head_dim))
        if self.mla_absorb:
            self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
            self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
            self.register_buffer("fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L+Drope, hidden)
            self.register_buffer("fused_qkv_a_proj_scale_inv", None, persistent=False)  # dense_fp8
        # engine-built, bound once at load (bind_resources)
        self._kv_key, self._attn_key, self._pos_key = kv_key, attn_key, pos_key
        self._kv_label, self._cache_layer = kv_label, cache_layer
        self._kv = None
        self._attn = None
        self._pos = None

    def bind_resources(self, resources: dict) -> None:
        self._kv = resources[self._kv_key]
        self._attn = resources[self._attn_key]
        self._pos = resources[self._pos_key] if self.use_rope else None
        if self.mla_fused_prep and self._pos.cos_sin_table is None:
            raise RuntimeError(
                "mla_fused_prep rotates from the position resource's cos/sin table: set "
                "rope_table_dtype in the model's PositionConfig")

    def _rope(
        self, position_ids: torch.Tensor, q_pe: torch.Tensor, k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q_pe`` / ``k_pe`` at ``position_ids`` through the position resource. Its
        kernel rotates whole rows in place, so the slices split off q and the latent are
        copied out first. Unrotated (``mla_use_nope``), they pass through."""
        if not self.use_rope:
            return q_pe, k_pe
        return self._pos.apply_qk(
            q_pe.clone(memory_format=torch.contiguous_format),
            k_pe.clone(memory_format=torch.contiguous_format), pos_ids=position_ids)

    @property
    def cache_layer_idx(self) -> int:
        """The KV layer this attention writes and reads: ``cache_layer``, else its own index
        (an MTP module's is past the trunk's, sharing the trunk's page table)."""
        if self._cache_layer is not None:
            return self._cache_layer
        return 0 if self.layer_idx is None else self.layer_idx

    # The layer index is read only here, outside dynamo: a traced frame that
    # reads it guards on it and compiles once per layer.
    @torch.compiler.disable
    def _write_and_attend(self, q, k, v, q_pe=None):
        layer_idx = self.cache_layer_idx
        label = {} if self._kv_label is None else {"label": self._kv_label}
        self._kv.write_kv(k, v, layer_idx=layer_idx, **label)
        layer = self._kv.layer_view(layer_idx)
        if q_pe is None:
            return self._attn.run(q, kv_cache_layer=layer, **label)
        return self._attn.run(q, kv_cache_layer=layer, q_pe=q_pe, **label)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **sparse,
    ) -> torch.Tensor:
        """``sparse``: the subclass's selection inputs, passed to its hooks."""
        if self.mla_absorb:
            return self._forward_absorbed(hidden_states, position_ids, **sparse)
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

        q_pe, k_pe = self._rope(position_ids, q_pe, k_pe)

        q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk)
        k_pe = k_pe.expand(num_tokens, h, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (T, H, Dqk)

        qk_pad = self.padded_head_dim - self.qk_head_dim
        q = F.pad(q, [0, qk_pad])  # (T, H, Dpad)
        k = F.pad(k, [0, qk_pad])  # (T, H, Dpad)
        v = F.pad(v, [0, self.padded_head_dim - self.v_head_dim])  # (T, H, Dpad)

        q = q * self.softmax_scale_boost
        attn = self._attend_naive(q, k, v, **sparse)  # (T, H, Dpad)
        attn = attn[..., : self.v_head_dim].reshape(num_tokens, h * self.v_head_dim)
        return self.o_proj(self._gate(attn, hidden_states))

    def _attend_naive(self, q, k, v, **sparse) -> torch.Tensor:
        """Naive-path attention of ``q`` over the padded ``k`` / ``v``: the cache's, by default."""
        del sparse
        return self._write_and_attend(q, k, v)

    def _forward_absorbed(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **sparse,
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

        if self.fp8_block is None:
            fused = F.linear(hidden_states, self.fused_qkv_a_proj_weight)
        else:
            fused = fp8_block_linear(hidden_states, self.fused_qkv_a_proj_weight,
                                     self.fused_qkv_a_proj_scale_inv, self.fp8_block)
        if (self.mla_fused_prep and (self._attends_written or not self._selects(**sparse))
                and fused.is_cuda and fused.dtype == torch.bfloat16):
            attn_latent = self._fused_prep_and_attend(fused, hidden_states, position_ids, **sparse)
        else:
            attn_latent = self._prep_and_attend(fused, hidden_states, position_ids, **sparse)
        return self.o_proj(self._gate(self._out(attn_latent), hidden_states))

    def _out(self, attn_latent: torch.Tensor) -> torch.Tensor:
        """``attn_latent (T, H_local, L)`` through ``w_vc``: ``(T, H_local * Dv)``. Eager, bmm
        writes ``(T, H, Dv)`` in place of einsum's ``(H, T, Dv)``, so the reshape stays a view
        (the same bits on H100); compiled, Inductor lays out einsum's, and under autograd (an
        ``out=`` has no backward) einsum runs too."""
        num_tokens, h = attn_latent.shape[:2]
        if torch.compiler.is_compiling() or attn_latent.requires_grad or self.w_vc.requires_grad:
            out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
            return out.reshape(num_tokens, h * self.v_head_dim)
        out = attn_latent.new_empty(num_tokens, h, self.v_head_dim)
        torch.bmm(attn_latent.transpose(0, 1), self.w_vc.transpose(1, 2), out=out.transpose(0, 1))
        return out.view(num_tokens, h * self.v_head_dim)

    def _gate(self, attn: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """``attn * sigmoid(g_proj(hidden_states))`` with ``mla_use_output_gate``, else ``attn``."""
        if self.g_proj is None:
            return attn
        return attn * torch.sigmoid(self.g_proj(hidden_states))

    def forward_block_row(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, row: int,
        block: int, rows: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One query per request at ``row`` of the step's block: ``hidden_states (N, hidden)``,
        ``position_ids (N,)``. The step plans ``block`` causal queries per request; the other
        rows attend with zero queries and write back ``rows``, this step's cache rows so far
        (rows past ``row`` are overwritten before a query reads them). Returns ``(N, hidden)``.
        """
        n, h = hidden_states.shape[0], self.num_heads
        if not self.mla_absorb:
            return self._block_row_naive(hidden_states, position_ids, row, block, rows)
        if self.fp8_block is None:
            fused = F.linear(hidden_states, self.fused_qkv_a_proj_weight)
        else:
            fused = fp8_block_linear(hidden_states, self.fused_qkv_a_proj_weight,
                                     self.fused_qkv_a_proj_scale_inv, self.fp8_block)
        if self.qk_rope_head_dim == 0:
            q_c, latent = self._latent_rows(fused)
            q_nope = self.q_b_proj(q_c).view(n, h, self.qk_head_dim)
            q_pe_all = q_nope.new_full((n, block, h, self.mla_cache_kpe), self.kpe_query_fill)
        else:
            q_c, kv_a, k_pe = fused.split(
                [self.q_a_proj.out_features, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            q = self.q_b_proj(self.q_a_layernorm(q_c.contiguous())).view(n, h, self.qk_head_dim)
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            kv_c = self.kv_a_layernorm(kv_a.clone(memory_format=torch.contiguous_format))
            q_pe, k_pe = self._rope(position_ids, q_pe, k_pe.reshape(n, 1, self.qk_rope_head_dim))
            latent = torch.cat([kv_c, k_pe.squeeze(1)], dim=-1)
            q_pe_all = q.new_zeros(n, block, h, self.qk_rope_head_dim)
            q_pe_all[:, row] = q_pe
        latents = rows.setdefault("latent", latent.new_zeros(n, block, latent.shape[-1]))
        latents[:, row] = latent
        q_nope_all = q_nope.new_zeros(n, block, h, self.kv_lora_rank)
        q_nope_all[:, row] = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)
        attn = self._write_and_attend(
            q_nope_all.view(n * block, h, -1), latents.view(n * block, -1), None,
            q_pe=q_pe_all.view(n * block, h, -1))
        out = self._out(attn.view(n, block, h, -1)[:, row])
        return self.o_proj(self._gate(out, hidden_states))

    def _block_row_naive(self, hidden_states, position_ids, row, block, rows):
        """``forward_block_row`` on the naive path: padded per-head K/V rows instead of latents."""
        n, h = hidden_states.shape[0], self.num_heads
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(n, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_a)).view(n, h, -1)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_pe, k_pe = self._rope(position_ids, q_pe, k_pe.reshape(n, 1, self.qk_rope_head_dim))
        pad = self.padded_head_dim
        q = F.pad(torch.cat([q_nope, q_pe], dim=-1), [0, pad - self.qk_head_dim])
        k = F.pad(torch.cat([k_nope, k_pe.expand(n, h, -1)], dim=-1), [0, pad - self.qk_head_dim])
        k_rows = rows.setdefault("k", k.new_zeros(n, block, h, pad))
        v_rows = rows.setdefault("v", k.new_zeros(n, block, h, pad))
        k_rows[:, row] = k
        v_rows[:, row] = F.pad(v, [0, pad - self.v_head_dim])
        q_all = q.new_zeros(n, block, h, pad)
        q_all[:, row] = q * self.softmax_scale_boost
        attn = self._write_and_attend(
            q_all.view(n * block, h, pad), k_rows.view(n * block, h, pad), v_rows.view(n * block, h, pad))
        attn = attn.view(n, block, h, pad)[:, row, :, : self.v_head_dim]
        return self.o_proj(self._gate(attn.reshape(n, h * self.v_head_dim), hidden_states))

    def _prep_and_attend(
        self,
        fused: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **sparse,
    ) -> torch.Tensor:
        """Norms, RoPE, cache write and attention over the latent, op by op. The
        post-q_a_layernorm ``q_c`` reaches ``_select`` too: a sparse selection may query from it
        (GLM-5.x's DSA indexers do)."""
        num_tokens = hidden_states.shape[0]
        h = self.num_heads
        if self.qk_rope_head_dim == 0:
            q_c, latent = self._latent_rows(fused)
            q_nope = self.q_b_proj(q_c).view(num_tokens, h, self.qk_head_dim)
            q_pe = q_nope.new_full((num_tokens, h, self.mla_cache_kpe), self.kpe_query_fill)
        else:
            q_c, kv_a, k_pe = fused.split(
                [self.q_a_proj.out_features, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            # FlashInfer RMSNorm needs 64-byte input alignment; decode split views can
            # be contiguous yet start at an unaligned offset, so clone kv_a.
            q_c = q_c.contiguous()
            kv_a = kv_a.clone(memory_format=torch.contiguous_format)

            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)
            q = q.view(num_tokens, h, self.qk_head_dim)
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)  # (T,1,L)
            k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # (T,1,Drope) shared MQA key

            q_pe, k_pe = self._rope(position_ids, q_pe, k_pe)
            latent = None

        selection = self._select(hidden_states, q_c, position_ids, **sparse)

        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        if latent is None:
            latent = torch.cat([kv_c, k_pe], dim=-1).squeeze(1)  # the cache rows (T, L + Drope)
        if selection is None:
            # dense: the paged kernel path over every cached row
            return self._write_and_attend(q_nope, latent, None, q_pe=q_pe)
        return self._attend_selected(selection, q_nope, q_pe, latent, **sparse)

    def _latent_rows(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """No decoupled dims: ``fused (T, q_lora + L)`` -> ``q_a_layernorm`` of the q half
        ``(T, q_lora)`` and the cache rows ``kv_a_layernorm(kv half) || zeros(mla_cache_kpe)``
        ``(T, L + kpe)``. One kernel on CUDA (``mla_latent_norm``)."""
        q_norm, kv_norm = self.q_a_layernorm, self.kv_a_layernorm
        if mla_latent_norm_available(fused, q_norm.weight, kv_norm.weight):
            return mla_latent_norm(
                fused, q_norm.weight, q_norm.variance_epsilon,
                kv_norm.weight, kv_norm.variance_epsilon, self.mla_cache_kpe,
            )
        q_c, kv_a = fused.split([self.q_a_proj.out_features, self.kv_lora_rank], dim=-1)
        # FlashInfer RMSNorm's alignment, as in _prep_and_attend
        q_c = q_c.contiguous()
        kv_a = kv_a.clone(memory_format=torch.contiguous_format)
        kv_c = kv_norm(kv_a)
        latent = torch.cat([kv_c, kv_c.new_zeros(kv_c.shape[0], self.mla_cache_kpe)], dim=-1)
        return q_norm(q_c), latent

    def _fused_prep_and_attend(
        self, fused: torch.Tensor, hidden_states: torch.Tensor, position_ids: torch.Tensor,
        **sparse,
    ) -> torch.Tensor:
        """``_prep_and_attend`` in two kernels: q_a's norm before q_b, then kv_a's norm, both
        RoPEs (from the position resource's cos/sin table) and the latent's cache write inside
        the attention call. A selecting subclass selects from the normed ``q_c`` in between."""
        q_c = mla_prep.q_norm(
            fused, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon)
        selection = (self._select(hidden_states, q_c, position_ids, **sparse)
                     if self._attends_written else None)
        q = self.q_b_proj(q_c).view(fused.shape[0], self.num_heads, self.qk_head_dim)
        q_nope = torch.einsum("thd,hdl->thl", q[..., : self.qk_nope_head_dim], self.w_kc)
        return self._write_fused_and_attend(q_nope, q, fused, position_ids, selection, **sparse)

    # outside dynamo for the layer index, as _write_and_attend
    @torch.compiler.disable
    def _write_fused_and_attend(self, q_nope, q, fused, position_ids, selection=None, **sparse):
        layer = self._kv.layer_view(self.cache_layer_idx)
        pages, offsets = self._kv.write_slots()
        q_pe = mla_prep.kv_write_q_rope(
            fused, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon,
            self._pos.cos_sin_table, position_ids, q, layer, pages, offsets)
        if selection is None:
            return self._attn.run(q_nope, kv_cache_layer=layer, q_pe=q_pe)
        return self._attend_written(selection, q_nope, q_pe, **sparse)

    # A selecting subclass that implements ``_attend_written`` takes the fused prep too.
    _attends_written = False

    def _selects(self, **sparse) -> bool:
        """Whether this forward may select rows (else the fused prep may serve it, or with
        ``_attends_written`` either way)."""
        del sparse
        return False

    def _select(self, hidden_states, q_c, position_ids, **sparse) -> torch.Tensor | None:
        """The rows each query attends, or None for every cached row (the default)."""
        del hidden_states, q_c, position_ids, sparse
        return None

    def _attend_selected(self, selection, q_nope, q_pe, latent, **sparse) -> torch.Tensor:
        """Write the step's cache rows ``latent (T, L + kpe)``, then absorbed attention over
        ``_select``'s rows: ``(T, H_local, kv_lora_rank)``."""
        raise NotImplementedError(f"{type(self).__name__} selects rows but attends none")

    def _attend_written(self, selection, q_nope, q_pe, **sparse) -> torch.Tensor:
        """``_attend_selected`` over cache rows the fused prep already wrote."""
        raise NotImplementedError(f"{type(self).__name__} selects rows but attends none")

    # The absorbed forward reads only w_kc / w_vc / fused_qkv_a_proj_weight;
    # these source projections would otherwise stay resident for the process
    # lifetime (~2.8 GB per rank on GLM-5.2's TP8 full model — KV pages lost).
    _ABSORBED_SOURCE_PROJS = ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj")

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build absorbed Q/O projections from the local-head ``kv_b_proj`` shard,
        then release the source weights the absorbed forward never reads."""
        if not self.mla_absorb:
            return
        del device  # protocol arg; kv_b_proj.weight already carries the right device
        if self.absorbed_sources_released:
            return  # fused and freed already: nothing left to fold from
        # detached: derived inference buffers, which need no autograd graph (and bmm's out=
        # in _out takes none)
        w = self.kv_b_proj.weight.detach()  # (H_local*(Dnope+Dv), L)
        h, d_nope, d_v, latent = (
            self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank)
        w = w.view(h, d_nope + d_v, latent)
        w_kc, w_vc = w.split([d_nope, d_v], dim=1)
        # clone, not contiguous(): with one local head the split slices are
        # already contiguous views, and a view would pin the storage we free
        self.w_kc = w_kc.clone(memory_format=torch.contiguous_format)  # (H_local, Dnope, L)
        self.w_vc = w_vc.clone(memory_format=torch.contiguous_format)  # (H_local, Dv,    L)

        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0).contiguous()
        if self.fp8_block is not None:
            # q_a's rows end on a scale block boundary, so kv_a's blocks stack right after
            assert self.q_a_proj.out_features % self.fp8_block[0] == 0
            self.fused_qkv_a_proj_scale_inv = torch.cat(
                [self.q_a_proj.weight_scale_inv, self.kv_a_proj_with_mqa.weight_scale_inv],
                dim=0).contiguous()
        self._release_absorbed_sources()

    @property
    def absorbed_sources_released(self) -> bool:
        return (
            self.mla_absorb and self.w_kc is not None
            and self.kv_b_proj.weight.numel() == 0
        )

    def _release_absorbed_sources(self) -> None:
        for name in self._ABSORBED_SOURCE_PROJS:
            proj = getattr(self, name)
            if isinstance(proj, Fp8Linear):
                proj.release()
                continue
            param = proj.weight
            # keep the Parameter (its weight_loader, dtype, device) and the
            # Linear's in/out_features the forward still reads; drop only the
            # storage — the same ``.data`` rebinding restore_fp32_params uses
            param.data = param.data.new_empty(0)
