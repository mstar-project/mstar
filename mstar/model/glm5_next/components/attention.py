"""GLM-5.3-Flash's MLA layer: the shared MLA (``components/mla.py``) without decoupled dims, plus
the DSA k-pool indexer each full-attention layer ships. Its KDA layers are the shared
``ParallelKimiDeltaAttention`` (``components/distributed/kda.py``)."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.mla import MLAAttention
from mstar.model.glm5_next.config import KV_CACHE, Glm5NextModelConfig

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


class Glm5NextMLAAttention(MLAAttention):
    """The shared MLA with no decoupled dims (``mla_use_nope``, ``qk_rope_head_dim`` 0), absorbed,
    through the latent array's handle at its model layer id."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
    ) -> None:
        if not config.mla_absorb:
            raise ValueError(
                "Glm5NextMLAAttention serves the absorbed path only (mla_absorb=True)"
            )
        super().__init__(config, comm_group, layer_idx, kv_key=KV_CACHE)
        # every full-attention layer, the MTP layer's too, ships its own indexer
        self.indexer = Glm5NextIndexer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states (T, hidden)`` flat engine layout, out the same."""
        return super().forward(hidden_states, None)

    def forward_block_row(
        self, hidden_states: torch.Tensor, row: int, latents: torch.Tensor,
    ) -> torch.Tensor:
        """``MLAAttention.forward_block_row`` over ``latents (N, block, L)``, the step's
        cache rows so far, which the MTP step allocates."""
        return super().forward_block_row(
            hidden_states, None, row, latents.shape[1], {"latent": latents})
