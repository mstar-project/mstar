"""Kimi-K2.7 / DeepSeek-V3 MLA attention — naive path + weight-absorbed path.

MLA compresses q and k/v through low-rank latents. Two forwards live here, picked
by ``config.mla_absorb`` (default ``True`` -> absorbed):

* **weight-absorbed** (``mla_absorb=True``, DEFAULT): folds ``kv_b_proj``'s
  up-projection into the Q path (``W_UK``) and the O path (``W_UV``) at load (plus
  the ``fused_qkv_a_proj`` down-proj fusion), via
  :meth:`KimiMLAAttention.process_weights_after_loading`, so attention runs as MQA
  over the COMPRESSED latent (``kv_c | k_pe``, one KV head) via
  ``cache_handle.run_attention_mla`` — a ~57x per-token KV shrink, numerically
  identical to naive up to fp rounding. See :meth:`_forward_absorbed`. Served by
  ``engine/cache_manager.py::MlaAbsorbCacheManager`` over a 4D latent paged cache.
  That backend currently uses a torch SDPA-over-latent path (correct + memory-lean
  but EAGER-ONLY, no CUDA-graph capture); the FlashInfer MLA kernel + CUDA-graph
  capture for real-1T throughput is a follow-up.
* **naive / materialized** (``mla_absorb=False``): projects the latent back up to
  full per-head K/V and runs ordinary attention, dropping straight onto mstar's
  paged ``run_attention`` ``[tokens, heads, head_dim]`` interface (matching vLLM's
  ``DeepseekV2Attention``, the non-absorbed class). Zero-pads q/k/v to
  ``padded_head_dim`` and folds a softmax boost into q. The M4-golden parity
  reference + opt-out fallback (production should keep this until the MLA kernel
  lands).

Per-token shape story (H heads, Dnope=qk_nope, Drope=qk_rope, Dqk=Dnope+Drope,
Dv=v_head_dim, L=kv_lora_rank):
  - q: ``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj`` -> ``[T,H,Dqk]``, split
    into ``q_nope[..,Dnope]`` / ``q_pe[..,Drope]``.
  - kv: ``kv_a_proj_with_mqa`` -> ``[L | Drope]``; the ``L`` slice is RMS-normed
    and ``kv_b_proj``-ed to per-head ``[k_nope[..,Dnope] | v[..,Dv]]``; the trailing
    ``Drope`` slice is the single shared MQA rope key ``k_pe[T,1,Drope]``.
  - YARN RoPE rotates only ``q_pe`` (per head) and ``k_pe`` (broadcast to H heads).
  - assemble ``k = [k_nope | k_pe_broadcast] -> [T,H,Dqk]``; zero-pad ``q``/``k``
    (Dqk) and ``v`` (Dv) up to ``padded_head_dim`` (FlashInfer SM90 rejects
    ``head_dim_vo`` not in {64,128,256}); fold the scale boost into ``q``
    (``run_attention`` uses the fixed ``1/sqrt(padded_head_dim)`` scale), attend,
    slice the output back to ``Dv``, ``o_proj``.

Cache config for this node: ``num_kv_heads == num_qo_heads == num_attention_heads``,
``head_dim == padded_head_dim`` (256 for the real Dqk=192, 64 for the reduced
Dqk=24). Under tensor parallelism each rank materializes only its
``num_attention_heads // tp_size`` local heads (K/V and Q both shard on the head
axis — there is no separate KV-head group in the naive path), and the paged cache
reports the matching per-rank count.

The absorbed cache config (``num_kv_heads == 1``, ``head_dim == kv_lora_rank +
qk_rope_head_dim``, no pad) is reported by ``kimi_model.py::get_kv_cache_config``
when ``mla_absorb`` is set.

The ``fused_qkv_a_proj`` weight fusion (``q_a_proj`` + ``kv_a_proj_with_mqa`` -> one
GEMM) is applied in the ABSORBED path only: ``process_weights_after_loading`` concats
the two replicated latent down-projection weights into ``fused_qkv_a_proj_weight`` and
:meth:`_forward_absorbed` runs a single ``F.linear`` then splits. The naive ``forward``
keeps the two separate calls unchanged.
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

        # MLA shards on the head dim under TP, mirroring vLLM
        # ``DeepseekV2MLAAttention``: the query/kv UP-projections (``q_b_proj`` /
        # ``kv_b_proj``) are ColumnParallel and ``o_proj`` is RowParallel, so each
        # rank owns a contiguous block of ``num_heads // tp_size`` attention heads.
        # The latent DOWN-projections (``q_a_proj`` / ``kv_a_proj_with_mqa``) and
        # their RMSNorms are REPLICATED (small shared latent, no head structure).
        # ``num_heads`` below is this rank's LOCAL head count used for every
        # forward reshape / RoPE / pad / run_attention; the parallel linears are
        # given the TOTAL width and divide by tp_size internally, and their
        # per-rank ``weight_loader`` slices this rank's head block — so one weight
        # path serves tp=1 and tp>1. The paged cache reports the matching per-rank
        # head count: ``KVCacheConfig.shard`` divides ``num_qo/kv_heads`` by the
        # node's instance world size (tp*sp), exactly like the Orpheus TP path.
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
        # FlashInfer SM90 rejects head_dim_vo not in {64,128,256}, so q/k/v are
        # zero-padded to this width for the paged run_attention; the attention
        # output is sliced back to v_head_dim. See config docstring.
        self.padded_head_dim = config.padded_head_dim
        # Parallel linears take the TOTAL head width (they divide by tp_size);
        # the forward uses ``self.num_heads`` (local).
        h = self.total_num_heads

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
        # DeepSeek's intended softmax scale (uses the PRE-absorption qk_head_dim,
        # not the latent width). The naive path reaches it by folding
        # softmax_scale_boost into q; the absorbed path hands this to the latent
        # cache backend (planned in Phase B; the reduced-config test mock reads it).
        self.softmax_scale = self.qk_head_dim ** -0.5 * mscale * mscale

        # Weight-absorbed MLA (config.mla_absorb): W_UK/W_UV are split out of
        # kv_b_proj post-load by process_weights_after_loading(). persistent=False:
        # derived tensors, never part of the checkpoint / state_dict.
        self.mla_absorb = config.mla_absorb
        if self.mla_absorb:
            self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
            self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
            # The two replicated latent down-projections (q_a_proj +
            # kv_a_proj_with_mqa) both read hidden_states; they fuse into one GEMM
            # for the absorbed forward. Built post-load; q_a_proj/kv_a_proj_with_mqa
            # remain the checkpoint load targets. persistent=False: derived tensor.
            self.register_buffer("fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L+Drope, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.mla_absorb:
            return self._forward_absorbed(hidden_states, cache_handle, position_ids)
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

    def _forward_absorbed(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Weight-absorbed MLA forward (see module docstring; ``config.mla_absorb``).

        kv_b_proj is folded into Q (``W_UK`` = ``w_kc``) and O (``W_UV`` = ``w_vc``)
        so the KV latent stays COMPRESSED and attention is MQA over ``[kv_c | k_pe]``
        (one KV head). Math identity vs naive: ``q_nope · k_nope ==
        (q_nope @ W_UK) · kv_c`` and ``attn · v == (attn · kv_c) @ W_UV``. The
        softmax scale (``self.softmax_scale``) is applied by the latent cache
        backend (planned in Phase B; the reduced-config test mock reads it), so —
        unlike naive — nothing is folded into q and nothing is padded.
        """
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

        # --- fused latent down-projection (q_a_proj + kv_a_proj_with_mqa in one
        # GEMM), then split into the q latent, the kv latent, and the rope key ---
        fused = F.linear(hidden_states, self.fused_qkv_a_proj_weight)
        q_c, kv_a, k_pe = fused.split(
            [self.q_a_proj.out_features, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # q_c / kv_a feed the FlashInfer RMSNorm kernel, which requires a
        # 64-byte-aligned input pointer. These are mid-tensor split views: q_c is at
        # offset 0 (always aligned); kv_a sits at q_lora_rank*dtype bytes (e.g.
        # reduced dims -> 96, not 64-aligned). ``.contiguous()`` fixes the multi-token
        # (prefill) case but is a NO-OP for a size-1 (decode, T=1) row — a [1,K] view
        # is already "contiguous" — leaving kv_a at the misaligned offset. So force a
        # fresh contiguous allocation for kv_a via ``clone``. (Real dims align by
        # chance; this keeps reduced-config / decode correct too.)
        q_c = q_c.contiguous()
        kv_a = kv_a.clone(memory_format=torch.contiguous_format)

        # --- Q (norm -> up), split nope/rope (same as naive) ---
        q = self.q_b_proj(self.q_a_layernorm(q_c))
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- KV latent kept COMPRESSED (no kv_b up-projection); norm the kv_c slice ---
        kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)  # (T,1,L)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # (T,1,Drope) shared MQA key

        # --- RoPE on the pe slices only (q_pe per head, k_pe single shared key) ---
        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe)

        # --- absorb W_UK into q_nope: (T,H,Dnope) x (H,Dnope,L) -> (T,H,L) ---
        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        # --- MQA over the latent: key=[kv_c|k_pe], value=kv_c -> (T,H,L) ---
        attn_latent = cache_handle.run_attention_mla(
            q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe)

        # --- absorb W_UV into the output: (T,H,L) x (H,Dv,L) -> (T,H,Dv) -> o_proj ---
        out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
        return self.o_proj(out.reshape(num_tokens, h * self.v_head_dim))

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build the absorbed projections from ``kv_b_proj`` (no-op unless mla_absorb).

        Splits ``kv_b_proj.weight`` ``[H_local*(Dnope+Dv), L]`` per local head into
        ``w_kc = W_UK [H_local, Dnope, L]`` (absorbed into Q) and
        ``w_vc = W_UV [H_local, Dv, L]`` (absorbed into O). Each rank's ``kv_b_proj``
        (ColumnParallelLinear) already holds only its local heads, so this is
        TP-correct with no extra sharding — one path serves tp=1 and tp>1.

        Named to match the generic post-load walker protocol so the serve path picks
        it up automatically (Phase B). Idempotent. ``device`` is accepted for
        protocol compatibility but unused — ``kv_b_proj.weight`` is already resident.
        """
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

        # Fuse the two replicated latent down-projections into one GEMM weight for
        # the absorbed forward: [q_a_proj ; kv_a_proj_with_mqa] -> a single
        # (q_lora + kv_lora + Drope, hidden) matmul that splits back into the q
        # latent, the kv latent, and the shared rope key.
        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0).contiguous()
