"""GLM-5.3-Flash's MLA layer: the shared MLA (``components/mla.py``) without decoupled dims, plus
the DSA k-pool indexer each full-attention layer ships. Its KDA layers are the shared
``ParallelKimiDeltaAttention`` (``components/distributed/kda.py``)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.dsa import paged
from mstar.engine.resources.dsa.config import INDEX_PLANES
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.model.components.mla import MLAAttention
from mstar.model.glm5_next.config import ATTN, DSA, KV_CACHE, LABEL, Glm5NextModelConfig

# the reference indexer's k_norm is LayerNorm(head_dim, eps=1e-6) with a bias, not rms_norm_eps
_INDEXER_K_NORM_EPS = 1e-6


class Glm5NextIndexer(nn.Module):
    """DSA k-pool indexer parameters for one full-attention layer."""

    def __init__(self, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.kpool = config.index_kpool

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=_INDEXER_K_NORM_EPS)
        # k-pool compression: an additive prior per pool member and the member gate (F.linear, no bias)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, config.hidden_size))

    def keys_and_gate(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each token's index key ``(T, D)`` and its pool gate ``(T, D)``."""
        k = self.k_norm(self.wk(hidden_states))
        return k, F.linear(hidden_states, self.index_kpool_compress_gate)

    def query_and_weights(
        self, q_c: torch.Tensor, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The score's query heads ``(T, H, D)`` and fp32 head weights ``(T, H)``, which carry
        the head-count and softmax scales (relu commutes with a positive scale)."""
        q = self.wq_b(q_c).view(q_c.shape[0], self.n_heads, self.head_dim)
        scale = self.n_heads ** -0.5 * self.head_dim ** -0.5
        return q, self.weights_proj(hidden_states).float() * scale


class Glm5NextMLAAttention(MLAAttention):
    """The shared MLA with no decoupled dims (``mla_use_nope``, ``qk_rope_head_dim`` 0), absorbed,
    over this layer's plane of the paged latent cache: the compact full-attention index ``kv_plane``
    (3 -> 0, 7 -> 1, ..., 43 -> 10; the MTP draft plane comes after), so the caller threads no
    cursor through the stack. Under dsa_long_context its DSA resource selects each row's
    slots past index_topk (``_select`` / ``_attend_selected``)."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        kv_plane: int | None = None,
    ) -> None:
        if not config.mla_absorb:
            raise ValueError(
                "Glm5NextMLAAttention serves the absorbed path only (mla_absorb=True)"
            )
        super().__init__(config, comm_group, layer_idx, kv_key=KV_CACHE, attn_key=ATTN,
                         pos_key=None, kv_label=LABEL, cache_layer=kv_plane)
        self.kv_plane = kv_plane
        if config.mtp_num_draft_tokens > 0:
            # With MTP the rows a query must not see (rejected drafts left as holes, or a sparse
            # plan's padding) carry a very negative rope slot: a query rope part of ones adds 0
            # to a real key (its slot is 0).
            self.kpe_query_fill = 1.0
        # every full-attention layer, the MTP layer's too, ships its own indexer
        self.indexer = Glm5NextIndexer(config)
        # dsa_long_context: this layer's index plane (its latent plane's, on the same pages; the
        # MTP layer selects with its own indexer too), and the DSA resource whose context the
        # submodule sets per forward
        self.index_plane = kv_plane if config.dsa_long_context else None
        self._dsa: DSAManager | None = None
        self.dsa_group = (comm_group or CommGroup.trivial()) if config.dsa_shard_prefill else None

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        self._dsa = resources.get(DSA)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states (T, hidden)`` flat engine layout, out the same."""
        return super().forward(hidden_states, None)

    def _select(self, hidden_states, q_c, position_ids):
        """DSA past index_topk: write the step's index keys and select each row's slots (prefill
        rows too: each selects over its own causal prefix); None without a context."""
        ctx = self._dsa.ctx if self._dsa is not None else None
        if ctx is None:
            return None
        k, gate = self.indexer.keys_and_gate(hidden_states)
        paged.write_index(self._kv, self.index_plane, LABEL, k, ctx, gate=gate,
                          ape=self.indexer.index_kpool_compress_ape)
        q, w = self.indexer.query_and_weights(q_c, hidden_states)
        return paged.select(q, w, self._kv.aux_view(INDEX_PLANES, self.index_plane), ctx,
                            self.dsa_group)

    def _attend_selected(self, slots, q_nope, q_pe, latent):
        """Write the step's latents, then attend each row over just its selected slots."""
        self._kv.write_kv(latent, None, layer_idx=self.kv_plane, label=LABEL)
        return paged.sparse_attend(q_nope, q_pe, self._kv.layer_view(self.kv_plane), slots,
                                   self._dsa.ctx, self.softmax_scale)

    def _select_block(self, hidden_states, q_c, position_ids, row, rows):
        """The draft row's slots under dsa_long_context: its index key joins the block's rows
        so far (``rows["index"]``), then every block row selects (only ``row`` has a query; the
        rest go unused)."""
        ctx = self._dsa.ctx if self._dsa is not None else None
        if ctx is None:
            return None
        keys, gates = rows["index"]
        n, block = keys.shape[:2]
        keys[:, row], gates[:, row] = self.indexer.keys_and_gate(hidden_states)
        paged.write_index(self._kv, self.index_plane, LABEL, keys.view(n * block, -1), ctx,
                          gate=gates.view(n * block, -1),
                          ape=self.indexer.index_kpool_compress_ape)
        q, w = self.indexer.query_and_weights(q_c, hidden_states)
        qs = q.new_zeros(n, block, *q.shape[1:])
        ws = w.new_zeros(n, block, w.shape[-1])
        qs[:, row], ws[:, row] = q, w
        return paged.select(qs.view(n * block, *q.shape[1:]), ws.view(n * block, -1),
                            self._kv.aux_view(INDEX_PLANES, self.index_plane), ctx,
                            self.dsa_group)

    def forward_block_row(
        self, hidden_states: torch.Tensor, row: int, latents: torch.Tensor,
        index_rows: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``MLAAttention.forward_block_row`` over ``latents (N, block, L + kpe)``, the step's
        cache rows so far, which the MTP step allocates. Under dsa_long_context ``index_rows``
        are the block's index keys and gates so far, kept the same way."""
        return super().forward_block_row(
            hidden_states, None, row, latents.shape[1], {"latent": latents, "index": index_rows})
