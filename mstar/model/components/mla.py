"""Multi-head latent attention (DeepSeek-V2/V3's MLA) through a handle on the model's latent or KV array.

``config`` carries the MLA dims: ``hidden_size``, ``num_attention_heads``, ``q_lora_rank``,
``kv_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``qk_head_dim``, ``v_head_dim``,
``padded_head_dim``, ``rms_norm_eps``, ``softmax_scale``, ``mla_absorb`` (the backend), and
optionally the fp8 fields ``dense_fp8_block`` reads and ``kernels`` (``reference``: the fp8
linears dequantized and the latent prep op by op). The backends:

* absorbed: once the weights load, ``kv_b_proj`` folds into the query (``w_kc``) and the output
  (``w_vc``), and the two down projections into one GEMM. Attention reads the array's compressed
  entries, the ``kv_lora_rank`` latent and the decoupled RoPE key.
* naive: per-head K/V padded to the attention backend's head size, the reduced-dims parity path.

The model hands the layer the handle on the array ``kv_key`` names (``bind_state``), and the layer
adds, reads and rotates through it at its own layer id. The decoupled RoPE is the array's, at the
positions the caller passes. ``mla_use_nope`` (optional) drops the decoupled dims
(``qk_rope_head_dim`` 0), and an entry is then the latent alone. The absorbed path's fused prep is
the state op `mla_prep_attend`. A model's sparse attention plugs in by overriding ``_select``
(absorbed), whose picks go to the read as ``selected``, or ``_attend_naive`` (naive); the keyword
arguments ``forward`` takes beyond its own reach them unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.state import PagedLatent, rope, state_op
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.fp8_linear import (
    COLUMN,
    ROW,
    Fp8Linear,
    dense_fp8_block,
    fp8_block_linear,
    fp8_reference,
)
from mstar.model.components.norm import RMSNorm, rms_norm
from mstar.utils import mla_prep
from mstar.utils.mla_latent_norm import mla_latent_norm, mla_latent_norm_available


def _prep_attend(kv, layer, fused, q, q_nope, kv_norm_weight, eps, pos_ids):
    rank, rope_dim = kv_norm_weight.shape[0], kv.array.entry.tensors["kpe"].shape[0]
    kv_a, k_pe = fused[:, fused.shape[1] - rank - rope_dim:].split([rank, rope_dim], dim=-1)
    # FlashInfer RMSNorm's alignment, and the RoPE kernel rotates in place, as in `MLAAttention._prep_and_attend`
    kv_c = rms_norm(kv_a.clone(memory_format=torch.contiguous_format), kv_norm_weight, eps)
    q_pe, k_pe = rope(
        kv, q[..., -rope_dim:].clone(memory_format=torch.contiguous_format),
        k_pe.reshape(-1, 1, rope_dim).clone(memory_format=torch.contiguous_format), at=pos_ids,
    )
    kv.add(layer, ckv=kv_c, kpe=k_pe.squeeze(1))
    return kv.read(layer, q_nope, q_pe=q_pe)


@state_op(
    adds=("kv",), reads=("kv",), formats=(PagedLatent(dtypes={torch.bfloat16}, devices={"cuda"}, rope_table=True),),
    reference=_prep_attend,
)
def mla_prep_attend(kv, layer, fused, q, q_nope, kv_norm_weight, eps, pos_ids, *, target):
    """kv_a's RMSNorm, both RoPEs and this layer's latent write in one launch, then the layer's read. ``fused`` is
    ``(T, q_lora + rank + rope_dim)`` rows of ``[q_a | kv_a | k_pe]``, ``q`` q_b's ``(T, H, nope + rope_dim)`` and
    ``q_nope`` its absorbed ``(T, H, rank)``."""
    q_pe = mla_prep.kv_write_q_rope(
        fused, kv_norm_weight, eps, target.rope_table, pos_ids, q, target.cache, target.page_of_token,
        target.slot_of_token,
    )
    return kv.read(layer, q_nope, q_pe=q_pe)


class MLAAttention(nn.Module):
    def __init__(
        self,
        config,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        *,
        kv_key: str,
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
        if not self.use_rope and self.qk_rope_head_dim:
            raise ValueError("mla_use_nope means no decoupled dims: qk_rope_head_dim must be 0")

        # dense_fp8: the checkpoint's fp8 weights and block scales, except kv_b_proj, which
        # the absorbed path folds into bf16 w_kc / w_vc.
        self.fp8_block = fp8 = dense_fp8_block(config)
        self.fp8_reference = ref = fp8_reference(config)
        kv_a_out = config.kv_lora_rank + config.qk_rope_head_dim

        self.q_a_proj = (
            nn.Linear(config.hidden_size, config.q_lora_rank, bias=False) if fp8 is None
            else Fp8Linear(config.hidden_size, config.q_lora_rank, fp8, reference=ref))
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = (
            ColumnParallelLinear(comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)
            if fp8 is None
            else Fp8Linear(config.q_lora_rank, h * self.qk_head_dim, fp8, comm_group,
                           shard=COLUMN, reference=ref))

        self.kv_a_proj_with_mqa = (
            nn.Linear(config.hidden_size, kv_a_out, bias=False) if fp8 is None
            else Fp8Linear(config.hidden_size, kv_a_out, fp8, reference=ref))
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = (
            RowParallelLinear(
                comm_group, h * config.v_head_dim, config.hidden_size,
                bias=False, input_is_parallel=True, reduce_results=True) if fp8 is None
            else Fp8Linear(h * config.v_head_dim, config.hidden_size, fp8, comm_group,
                           shard=ROW, reduce_results=True, reference=ref))

        self.layer_idx = layer_idx
        # run_attention uses 1/sqrt(padded_head_dim); fold the intended
        # softmax_scale into q on the padded path.
        self.softmax_scale = config.softmax_scale
        self.softmax_scale_boost = math.sqrt(self.padded_head_dim) * self.softmax_scale

        self.mla_absorb = config.mla_absorb
        # the absorbed path's norms, RoPEs and cache write in two kernels (mla_prep.py). Each
        # has its ops' bits alone, but a captured MTP step writes different bits into each
        # request's last live cache rows with them, so the reference keeps the ops
        self.mla_fused_prep = (
            not ref and self.mla_absorb and self.use_rope
            and mla_prep.supports(config.q_lora_rank, config.kv_lora_rank, self.qk_rope_head_dim))
        if self.mla_absorb:
            self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
            self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
            self.register_buffer("fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L+Drope, hidden)
            self.register_buffer("fused_qkv_a_proj_scale_inv", None, persistent=False)  # dense_fp8
            self.lora_unservable = dict.fromkeys(
                self._ABSORBED_SOURCE_PROJS, "folded into the absorbed MLA weights at load (mla_absorb)")
        self._kv_key = kv_key
        # the model's id for this layer, a plain attribute that `.to(device)` leaves on the CPU, and the handle on the
        # array ``kv_key`` names, where the model declares it
        self.layer = None if layer_idx is None else torch.tensor(layer_idx, device="cpu")
        self.state = None

    def bind_state(self, handles: dict) -> None:
        self.state = handles.get(self._kv_key)

    def _rope(
        self, position_ids: torch.Tensor, q_pe: torch.Tensor, k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q_pe`` / ``k_pe`` at ``position_ids`` through the handle. Its kernel rotates
        whole rows in place, so the slices split off q and the latent are copied out first.
        Without decoupled dims (``mla_use_nope``), they pass through."""
        if not self.use_rope:
            return q_pe, k_pe
        return rope(
            self.state, q_pe.clone(memory_format=torch.contiguous_format),
            k_pe.clone(memory_format=torch.contiguous_format), at=position_ids,
        )

    def _write_and_attend(self, q, k, v, q_pe=None, *, keep=None, selected=None):
        """``keep`` and ``selected`` reach the handle's add and read. ``v`` None: ``k`` is the absorbed
        path's latent rows, with no rope query ``q_pe`` where there are no decoupled dims."""
        if v is not None:
            self.state.add(self.layer, keep=keep, k=k, v=v)
            return self.state.read(self.layer, q)
        if q_pe is None:
            self.state.add(self.layer, keep=keep, ckv=k)
            return self.state.read(self.layer, q)
        self.state.add(self.layer, keep=keep, ckv=k[:, :self.kv_lora_rank], kpe=k[:, self.kv_lora_rank:])
        return self.state.read(self.layer, q, q_pe=q_pe, selected=selected)

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
        return self.o_proj(attn)

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
                                     self.fused_qkv_a_proj_scale_inv, self.fp8_block,
                                     use_reference=self.fp8_reference)
        attn_latent = self._attend_state(fused, hidden_states, position_ids, **sparse)
        return self.o_proj(self._out(attn_latent))

    def _attend_state(self, fused, hidden_states, position_ids, **sparse) -> torch.Tensor:
        """The absorbed path on the latent array's handle: `mla_prep_attend` on a layer that picks no positions, whose
        fused kernel the handle takes or leaves, and the calls op by op otherwise."""
        if not (self.mla_fused_prep and fused.is_cuda and fused.dtype == torch.bfloat16) or self._selects(**sparse):
            return self._prep_and_attend(fused, hidden_states, position_ids, **sparse)
        q_c = mla_prep.q_norm(fused, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon)
        q = self.q_b_proj(q_c).view(fused.shape[0], self.num_heads, self.qk_head_dim)
        q_nope = torch.einsum("thd,hdl->thl", q[..., : self.qk_nope_head_dim], self.w_kc)
        return mla_prep_attend(
            self.state, self.layer, fused, q, q_nope, self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon, position_ids,
        )

    def _out(self, attn_latent: torch.Tensor) -> torch.Tensor:
        """``attn_latent (T, H_local, L)`` through ``w_vc``: ``(T, H_local * Dv)``. Eager, bmm
        writes ``(T, H, Dv)`` in place, so the reshape stays a view (einsum's bits on H100);
        compiled, or under autograd (``out=`` has no backward), einsum runs."""
        num_tokens, h = attn_latent.shape[:2]
        if torch.compiler.is_compiling() or attn_latent.requires_grad or self.w_vc.requires_grad:
            out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
            return out.reshape(num_tokens, h * self.v_head_dim)
        out = attn_latent.new_empty(num_tokens, h, self.v_head_dim)
        torch.bmm(attn_latent.transpose(0, 1), self.w_vc.transpose(1, 2), out=out.transpose(0, 1))
        return out.view(num_tokens, h * self.v_head_dim)

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
                                     self.fused_qkv_a_proj_scale_inv, self.fp8_block,
                                     use_reference=self.fp8_reference)
        if self.qk_rope_head_dim == 0:
            q_c, latent = self._latent_rows(fused)
            q_nope = self.q_b_proj(q_c).view(n, h, self.qk_head_dim)
            q_pe_all = None
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
            q_pe=None if q_pe_all is None else q_pe_all.view(n * block, h, -1), keep=False)
        out = self._out(attn.view(n, block, h, -1)[:, row])
        return self.o_proj(out)

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
            q_all.view(n * block, h, pad), k_rows.view(n * block, h, pad), v_rows.view(n * block, h, pad), keep=False)
        attn = attn.view(n, block, h, pad)[:, row, :, : self.v_head_dim]
        return self.o_proj(attn.reshape(n, h * self.v_head_dim))

    def _prep_and_attend(
        self,
        fused: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **sparse,
    ) -> torch.Tensor:
        """Norms, RoPE, cache write and attention over the latent, op by op. The
        post-q_a_layernorm ``q_c`` reaches ``_select`` too: a sparse selection may query from it."""
        num_tokens = hidden_states.shape[0]
        h = self.num_heads
        if self.qk_rope_head_dim == 0:
            q_c, latent = self._latent_rows(fused)
            q_nope = self.q_b_proj(q_c).view(num_tokens, h, self.qk_head_dim)
            q_pe = None
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
        return self._write_and_attend(q_nope, latent, None, q_pe=q_pe, selected=selection)

    def _latent_rows(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """No decoupled dims: ``fused (T, q_lora + L)`` -> ``q_a_layernorm`` of the q half
        ``(T, q_lora)`` and the entries, ``kv_a_layernorm`` of the kv half ``(T, L)``. One kernel
        on CUDA (``mla_latent_norm``)."""
        q_norm, kv_norm = self.q_a_layernorm, self.kv_a_layernorm
        if mla_latent_norm_available(fused, q_norm.weight, kv_norm.weight):
            return mla_latent_norm(
                fused, q_norm.weight, q_norm.variance_epsilon, kv_norm.weight, kv_norm.variance_epsilon, 0,
            )
        q_c, kv_a = fused.split([self.q_a_proj.out_features, self.kv_lora_rank], dim=-1)
        # FlashInfer RMSNorm's alignment, as in _prep_and_attend
        return q_norm(q_c.contiguous()), kv_norm(kv_a.clone(memory_format=torch.contiguous_format))

    def _selects(self, **sparse) -> bool:
        """Whether this forward may pick rows, which the fused prep doesn't serve."""
        del sparse
        return False

    def _select(self, hidden_states, q_c, position_ids, **sparse) -> torch.Tensor | None:
        """The rows each query attends, which the handle's read takes as ``selected``, or None
        for every cached row (the default)."""
        del hidden_states, q_c, position_ids, sparse
        return None

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
