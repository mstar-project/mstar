"""GLM-5.2 MLA attention with absorbed and naive fallback paths.

Same structure as ``kimi_k2_7/components/attention.py`` (both are
DeepSeek-family MLA) with GLM geometry — q_lora 2048, nope 192, rope 64,
v_head 256, 64 heads — and plain (non-Yarn) RoPE, so every mscale term is 1.
qk_head_dim is exactly 256, so the FlashInfer pad on the naive path is a
no-op for the full model (reduced test configs still exercise it: 24 -> 64).

Phase C: FULL indexer layers (``is_full_indexer_layer``) instantiate a
``Glm52Indexer`` whose checkpoint weights now load; the naive path accepts
a precomputed ``dsa_selection`` and restricts softmax to the selected set
(reference semantics: masked dense attention, -1 entries excluded). At
ctx <= index_topk, top-k selection of <= topk tokens is the identity, so
the dense path IS the exact DSA computation in that regime.

Engine half (``dsa_ctx``, absorbed path only): FULL layers append this
chunk's index keys to the per-request k-store and — once any request's
context exceeds ``index_topk`` — compute the selection SHARED layers then
reuse (``_dsa_update``). Beyond topk, decode queries run
``_run_sparse_absorbed``: gather the selected <= topk latent vectors from
the paged MLA cache through the request's page table and run dense MQA
over the gathered set — the same math as ``MlaAbsorbCacheManager._sdpa_mla``
restricted to the selected rows (-1 padding excluded; the spec marks
gather + dense semantically identical to the FlashMLA sparse kernel,
dsa-indexer-spec.md section 4). While every context fits topk the paged
``run_attention_mla`` path runs UNTOUCHED — bit-identical to flag-off.
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
from mstar.model.glm52.dsa import Glm52DsaForwardContext


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
        self.layer_idx = layer_idx  # k-store key + selection provenance
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
        dsa_ctx: Glm52DsaForwardContext | None = None,
    ) -> torch.Tensor:
        """``dsa_selection``: optional ``(T, index_topk)`` int rows from
        ``Glm52Indexer.compute_selection`` (-1 = padding). When given, the
        naive path restricts softmax to the selected keys of the current
        batch; when None, dense behavior is unchanged.

        ``dsa_ctx``: the engine-threaded per-forward DSA context (absorbed
        path only) — see the module docstring and ``_dsa_update``."""
        if dsa_ctx is not None and not self.mla_absorb:
            # The sparse gather path reads the paged 576-dim latent cache;
            # the naive backend stores padded per-head K/V instead. The
            # naive path stays what it is: the reduced-test parity fallback.
            raise RuntimeError(
                "dsa_long_context requires mla_absorb: the sparse gather path "
                "consumes the paged MLA latent cache, which only the absorbed "
                "backend maintains"
            )
        if self.mla_absorb:
            if dsa_selection is not None:
                raise NotImplementedError(
                    "dsa_selection is the naive-path reference hook; the "
                    "absorbed path takes the engine dsa_ctx instead"
                )
            return self._forward_absorbed(
                hidden_states, cache_handle, position_ids, dsa_ctx)
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
        dsa_ctx: Glm52DsaForwardContext | None = None,
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

        # Keep the post-q_a_layernorm latent: it is ALSO the indexer's query
        # input (the shared 2048-dim bottleneck, dsa-indexer-spec.md section
        # 6 item 1). Same ops in the same order as the previous fused
        # expression — bit-identical.
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)  # (T,1,L)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # (T,1,Drope) shared MQA key

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe)

        # None-guarded here (not just inside _dsa_update) so the default
        # flag-off path never crosses the compiler.disable boundary — dynamo
        # specializes on dsa_ctx=None and folds the branch away.
        selection = (
            self._dsa_update(dsa_ctx, hidden_states, q_c, position_ids)
            if dsa_ctx is not None else None
        )

        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        if selection is None:
            # Identity regime (or DSA off): the paged path, untouched.
            attn_latent = cache_handle.run_attention_mla(
                q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe)
        else:
            attn_latent = self._run_sparse_absorbed(
                cache_handle, dsa_ctx, selection, q_nope, q_pe, kv_c, k_pe)

        out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
        return self.o_proj(out.reshape(num_tokens, h * self.v_head_dim))

    # Host-side per-request work (dict mutation, python loops over spans):
    # keep dynamo out of it, same as the cache-manager plan/run methods —
    # under the engine's warmup torch.compile these run eagerly via the
    # disable wrapper instead of graph-breaking token by token.
    @torch.compiler.disable
    def _dsa_update(
        self,
        dsa_ctx: Glm52DsaForwardContext,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Maintain the indexer k-store and return the selection to apply.

        FULL layers (``self.indexer`` set) always append this chunk's
        roped+normed keys — history must be complete from token 0 for the
        step that first crosses topk — and, when any request is beyond topk,
        score their per-request history (current chunk included: the causal
        window is self-inclusive) and publish the selection on ``dsa_ctx``.
        SHARED layers publish nothing and reuse the most recent FULL layer's
        rows (IndexShare, spec section 3). Returns None in the identity
        regime so the caller keeps the untouched dense path.
        """
        if self.indexer is None:
            if not dsa_ctx.needs_selection:
                return None
            if dsa_ctx.last_selection is None:
                raise RuntimeError(
                    f"SHARED layer {self.layer_idx} needs a DSA selection but no "
                    "FULL layer ran before it — the skip formula guarantees "
                    "layer 0 is FULL, so the threading order is broken"
                )
            return dsa_ctx.last_selection

        keys = self.indexer.compute_k(hidden_states, position_ids)
        for span in dsa_ctx.spans:
            dsa_ctx.k_store.append(
                span.request_id,
                self.layer_idx,
                keys[span.q_start : span.q_start + span.q_len],
                start_pos=span.ctx_start,
            )
        if not dsa_ctx.needs_selection:
            return None

        rows = []
        for span in dsa_ctx.spans:
            token_slice = slice(span.q_start, span.q_start + span.q_len)
            history = dsa_ctx.k_store.history(
                span.request_id, self.layer_idx, span.ctx_start + span.q_len)
            rows.append(self.indexer.compute_selection(
                q_c[token_slice],
                hidden_states[token_slice],
                position_ids[token_slice],
                history,
            ))
        selection = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        dsa_ctx.last_selection = selection
        dsa_ctx.last_selection_layer = self.layer_idx
        return selection

    @torch.compiler.disable
    def _run_sparse_absorbed(
        self,
        cache_handle: BatchedCacheManager,
        dsa_ctx: Glm52DsaForwardContext,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse absorbed MLA: gather the selected latents, dense MQA over them.

        Replaces ``run_attention_mla`` beyond topk, so it also does that
        call's cache write: scatter this chunk's 576-dim latents to their
        page slots (same ``page_indices[pos // page_size]`` mapping the
        MlaAbsorbCacheManager plan uses) BEFORE gathering — the causal
        window includes self, so the current token's latent must be
        readable. Then, per query, gather the selected <= topk latents
        (request-local indices -> page slots; -1 padding excluded by the
        gather itself) and run ``_sdpa_mla``'s math over the gathered set:
        fp32 MQA, softmax over exactly the selected rows, value = the
        ckv slice. Gathered-row order is irrelevant up to fp addition order.

        v1 is decode-shaped (q_len == 1 per request; the submodule guard
        keeps prefill within topk) and loops requests on the host — fine at
        decode batch sizes, and the batched-gather kernel belongs to the
        fp8 paged-pool follow-up.
        """
        latent_cache = cache_handle.kv_cache[cache_handle.layer_idx]
        page_size = cache_handle.kv_cache_config.page_size
        latent = torch.cat([kv_c, k_pe], dim=-1).squeeze(1)  # (T, L + Drope)
        latent_dim = q_nope.shape[-1]  # ckv width (post-w_kc absorption)
        query = torch.cat([q_nope, q_pe], dim=-1).float()  # (T, H, L + Drope)

        out = torch.empty_like(q_nope)
        for span in dsa_ctx.spans:
            if span.q_len != 1:
                raise RuntimeError(
                    f"request {span.request_id!r}: sparse attention beyond "
                    "index_topk is decode-only in v1 (q_len == 1); prefill "
                    "beyond topk is refused by the submodule guard"
                )
            pages = torch.tensor(
                span.page_indices, dtype=torch.long, device=latent.device)
            pos = span.ctx_start  # this decode token's absolute position
            latent_cache[pages[pos // page_size], pos % page_size] = (
                latent[span.q_start].to(latent_cache.dtype))

            row = selection[span.q_start]
            picked = row[row >= 0].long()  # request-local positions, causal
            gathered = latent_cache[
                pages[picked // page_size], picked % page_size
            ].float()  # (n, L + Drope)

            scores = torch.einsum(
                "hd,kd->hk", query[span.q_start], gathered) * self.softmax_scale
            attn = scores.softmax(dim=-1)
            out[span.q_start] = torch.einsum(
                "hk,kd->hd", attn, gathered[:, :latent_dim]).to(out.dtype)
        return out

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
