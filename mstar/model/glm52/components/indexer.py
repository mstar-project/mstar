"""GLM-5.2's DSA indexer: the index keys and queries its FULL layers hand the DSA resource
(``mstar/engine/resources/dsa``), which scores them and picks each query's top-k."""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.glm52.config import POS_RESOURCE, Glm52ModelConfig


def is_full_indexer_layer(config: Glm52ModelConfig, layer_idx: int) -> bool:
    """FULL layers run the indexer; SHARED layers reuse the last FULL selection."""
    skip = (
        max(layer_idx - config.index_skip_topk_offset + 1, 0)
        % config.index_topk_freq
        != 0
    )
    return not skip


class Glm52Indexer(nn.Module):
    """DSA indexer projections for one FULL layer: the key each token stores, and each query
    row's heads and head weights."""

    def __init__(self, config: Glm52ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = torch.nn.LayerNorm(self.head_dim, eps=1e-6)  # weight + bias

        # The main MLA's RoPE table and positions, with the indexer's own pairing
        # (indexer_rope_interleave=True => GPT-J).
        self.rope_interleave = config.indexer_rope_interleave
        self._pos = None  # engine-built, bound once at load
        # Both the softmax scale (head_dim^-0.5) and n_heads^-0.5 fold into
        # the per-head weights, as the reference implementation does; the
        # bf16/fp32 path carries no dequant scale to fold in alongside them.
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5

    def bind_resources(self, resources: dict) -> None:
        self._pos = resources[POS_RESOURCE]

    def _rope_first_dims(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the FIRST ``rope_dim`` dims of ``(T, H, D)`` ``x``: the position resource's
        table is ``rope_dim`` wide, so the rest of each head passes through. A copy: the
        kernel rotates in place."""
        x, _ = self._pos.apply_qk(
            x.clone(memory_format=torch.contiguous_format), None, pos_ids=positions,
            interleave=self.rope_interleave)
        return x

    def compute_k(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-token index key ``(T, head_dim)`` — what the indexer cache stores."""
        k = self.k_norm(self.wk(hidden_states))  # (T, D)
        return self._rope_first_dims(k.unsqueeze(1), positions).squeeze(1)

    def compute_query(
        self,
        q_c: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row's index query ``(T, H, D)`` and head weights ``(T, H)``, fp32: what the DSA
        resource scores the stored keys with."""
        num_tokens = q_c.shape[0]
        q = self.wq_b(q_c).view(num_tokens, self.n_heads, self.head_dim)
        q = self._rope_first_dims(q, positions).float()
        w = self.weights_proj(hidden_states).float() * self.weight_scale  # (T, H)
        return q, w
