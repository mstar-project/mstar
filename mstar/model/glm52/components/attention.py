"""GLM-5.2 MLA attention with absorbed and naive fallback paths."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.norm import RMSNorm
from mstar.model.glm52.components.indexer import Glm52Indexer, is_full_indexer_layer
from mstar.model.glm52.components.rope import Glm52RotaryEmbedding
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, Glm52ModelConfig
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
    """Dense ``(T, H, D)`` attention with an additive ``(T, T)`` mask."""
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
        # And only on the DSA engine path: with dsa_long_context off no
        # forward ever hands the layer a dsa_ctx, so a FULL layer's indexer
        # (replicated, not TP-sharded) would be read and held for nothing;
        # the loader skips its keys to match (Glm52ForCausalLM.load_weights).
        self.layer_idx = layer_idx  # k-store key + selection provenance
        self.dsa_long_context = config.dsa_long_context
        self.indexer = (
            Glm52Indexer(config)
            if self.dsa_long_context
            and layer_idx is not None
            and is_full_indexer_layer(config, layer_idx)
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
        # engine-built, bound once at load (bind_resources)
        self._kv = None
        self._attn = None

    def bind_resources(self, resources: dict) -> None:
        self._kv = resources[KV_RESOURCE]
        self._attn = resources[ATTN_RESOURCE]

    @property
    def cache_layer_idx(self) -> int:
        """The KV layer this attention writes and reads: its own index (the
        MTP module's is num_hidden_layers, sharing the trunk's page table)."""
        return 0 if self.layer_idx is None else self.layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_selection: torch.Tensor | None = None,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``dsa_selection``: optional ``(T, index_topk)`` int rows from
        ``Glm52Indexer.compute_selection`` (-1 = padding).
        """
        if dsa_ctx is not None and not self.mla_absorb:
            # The sparse gather path reads the paged 576-dim latent cache;
            # the naive backend stores padded per-head K/V instead. The
            # naive path stays what it is: the reduced-test parity fallback.
            raise RuntimeError(
                "dsa_long_context requires mla_absorb: the sparse gather path "
                "consumes the paged MLA latent cache, which only the absorbed "
                "backend maintains"
            )
        if dsa_ctx is not None and not self.dsa_long_context:
            # built flag-off: no indexer was constructed and its weights were
            # never loaded, so a FULL layer here would pass for SHARED
            raise RuntimeError(
                "dsa_ctx handed to an attention layer built with dsa_long_context "
                "off — its indexer does not exist"
            )
        if self.mla_absorb:
            if dsa_selection is not None:
                raise NotImplementedError(
                    "dsa_selection is the naive-path reference hook; the "
                    "absorbed path takes the engine dsa_ctx instead"
                )
            return self._forward_absorbed(
                hidden_states, position_ids, dsa_ctx, rope_cos_sin=rope_cos_sin)
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

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe, cos_sin=rope_cos_sin)

        q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk)
        k_pe = k_pe.expand(num_tokens, h, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (T, H, Dqk)

        qk_pad = self.padded_head_dim - self.qk_head_dim
        q = F.pad(q, [0, qk_pad])  # (T, H, Dpad)
        k = F.pad(k, [0, qk_pad])  # (T, H, Dpad)
        v = F.pad(v, [0, self.padded_head_dim - self.v_head_dim])  # (T, H, Dpad)

        q = q * self.softmax_scale_boost
        if dsa_selection is None:
            layer_idx = self.cache_layer_idx
            self._kv.write_kv(k, v, layer_idx=layer_idx)
            attn = self._attn.run(
                q, kv_cache_layer=self._kv.layer_view(layer_idx))  # (T, H, Dpad)
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
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
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

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe, cos_sin=rope_cos_sin)

        # None-guarded here (not just inside _dsa_update) so the default
        # flag-off path never crosses the compiler.disable boundary — dynamo
        # specializes on dsa_ctx=None and folds the branch away.
        selection = (
            self._dsa_update(dsa_ctx, hidden_states, q_c, position_ids)
            if dsa_ctx is not None else None
        )

        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        if selection is None:
            # Identity regime (or DSA off): the paged kernel path, untouched.
            layer = self._kv.layer_view(self.cache_layer_idx)
            self._attn.write_latent(torch.cat([kv_c, k_pe], dim=-1).squeeze(1), layer)
            attn_latent = self._attn.run(q_nope, q_pe, layer)
        else:
            attn_latent = self._run_sparse_absorbed(
                dsa_ctx, selection, q_nope, q_pe, kv_c, k_pe)

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
        """Maintain the indexer k-store and return the selection to apply."""
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
        dsa_ctx: Glm52DsaForwardContext,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse absorbed MLA: gather the selected latents, dense MQA over them."""
        latent_cache = self._kv.layer_view(self.cache_layer_idx)
        page_size = self._kv.config.page_size
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

    # The absorbed forward reads only w_kc / w_vc / fused_qkv_a_proj_weight;
    # these source projections would otherwise stay resident for the process
    # lifetime (~2.8 GB per rank on the TP8 full model — KV pages lost).
    _ABSORBED_SOURCE_PROJS = ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj")

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build absorbed Q/O projections from the local-head ``kv_b_proj`` shard,
        then release the source weights the absorbed forward never reads."""
        if not self.mla_absorb:
            return
        del device  # protocol arg; kv_b_proj.weight already carries the right device
        if self.absorbed_sources_released:
            return  # fused and freed already: nothing left to fold from
        w = self.kv_b_proj.weight  # (H_local*(Dnope+Dv), L)
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
        self._release_absorbed_sources()

    @property
    def absorbed_sources_released(self) -> bool:
        return (
            self.mla_absorb and self.w_kc is not None
            and self.kv_b_proj.weight.numel() == 0
        )

    def _release_absorbed_sources(self) -> None:
        for name in self._ABSORBED_SOURCE_PROJS:
            param = getattr(self, name).weight
            # keep the Parameter (its weight_loader, dtype, device) and the
            # Linear's in/out_features the forward still reads; drop only the
            # storage — the same ``.data`` rebinding restore_fp32_params uses
            param.data = param.data.new_empty(0)
