"""GLM-5.2 MLA attention: the shared MLA (components/mla.py) and DSA, its sparse attention."""
from __future__ import annotations

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.mla import MLAAttention
from mstar.model.glm52.components.indexer import Glm52Indexer, is_full_indexer_layer
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


class Glm52MLAAttention(MLAAttention):
    """The shared MLA with GLM-5.2's DSA: on FULL layers an indexer picks each query's top
    ``index_topk`` keys, SHARED layers reuse the last FULL layer's pick, and the absorbed
    path attends just those latents (``dsa_ctx``). ``dsa_selection`` is the naive path's
    reference hook: given rows, the mask over the current batch."""

    def __init__(
        self,
        config: Glm52ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, comm_group, layer_idx, kv_key=KV_RESOURCE, attn_key=ATTN_RESOURCE)
        # DSA indexer only on FULL layers — SHARED layers carry no indexer
        # weights in the checkpoint (they reuse the last FULL selection),
        # and a dormant module would sit uninitialized after ``to_empty``.
        # And only on the DSA engine path: with dsa_long_context off no
        # forward ever hands the layer a dsa_ctx, so a FULL layer's indexer
        # (replicated, not TP-sharded) would be read and held for nothing;
        # the loader skips its keys to match (Glm52ForCausalLM.load_weights).
        # layer_idx is the k-store key and the selection's provenance.
        self.dsa_long_context = config.dsa_long_context
        self.indexer = (
            Glm52Indexer(config)
            if self.dsa_long_context
            and layer_idx is not None
            and is_full_indexer_layer(config, layer_idx)
            else None
        )

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
        if self.mla_absorb and dsa_selection is not None:
            raise NotImplementedError(
                "dsa_selection is the naive-path reference hook; the "
                "absorbed path takes the engine dsa_ctx instead"
            )
        return super().forward(hidden_states, position_ids, rope_cos_sin,
                               dsa_selection=dsa_selection, dsa_ctx=dsa_ctx)

    def _attend_naive(self, q, k, v, dsa_selection=None, dsa_ctx=None) -> torch.Tensor:
        if dsa_selection is None:
            return self._write_and_attend(q, k, v)
        # Reference sparse path: softmax over the selected keys only,
        # via an additive mask over the current batch (a gather of the
        # selected k/v rows is semantically identical; -1 excluded).
        mask = dsa_selection_to_mask(dsa_selection, q.shape[0], dtype=q.dtype, device=q.device)
        return masked_reference_attention(q, k, v, mask)

    def _select(self, hidden_states, q_c, position_ids, dsa_ctx=None, dsa_selection=None):
        # None-guarded here (not just inside _dsa_update) so the default
        # flag-off path never crosses the compiler.disable boundary — dynamo
        # specializes on dsa_ctx=None and folds the branch away.
        if dsa_ctx is None:
            return None
        return self._dsa_update(dsa_ctx, hidden_states, q_c, position_ids)

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
    def _attend_selected(
        self,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        dsa_selection: torch.Tensor | None = None,
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

            # fp32 outside the engine's bf16 autocast, which would run both einsums in bf16
            with torch.autocast(query.device.type, enabled=False):
                scores = torch.einsum(
                    "hd,kd->hk", query[span.q_start], gathered) * self.softmax_scale
                attn = scores.softmax(dim=-1)
                out[span.q_start] = torch.einsum(
                    "hk,kd->hd", attn, gathered[:, :latent_dim]).to(out.dtype)
        return out
