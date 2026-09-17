"""GLM-5.2 DSA sparse-attention indexer (CPU-testable reference)."""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.glm52.components.rope import Glm52RotaryEmbedding
from mstar.model.glm52.config import Glm52ModelConfig


def is_full_indexer_layer(config: Glm52ModelConfig, layer_idx: int) -> bool:
    """FULL layers run the indexer; SHARED layers reuse the last FULL selection."""
    skip = (
        max(layer_idx - config.index_skip_topk_offset + 1, 0)
        % config.index_topk_freq
        != 0
    )
    return not skip


class Glm52Indexer(nn.Module):
    """DSA indexer for one FULL layer: k projection/cache side + selection side."""

    def __init__(self, config: Glm52ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk = config.index_topk

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = torch.nn.LayerNorm(self.head_dim, eps=1e-6)  # weight + bias

        # Same interleaved rotation, theta, and absolute positions as the
        # main MLA rope (indexer_rope_interleave=True => GPT-J pairing).
        self.rotary = Glm52RotaryEmbedding(
            rotary_dim=config.qk_rope_head_dim, base=config.rope_theta)
        # Both the softmax scale (head_dim^-0.5) and n_heads^-0.5 fold into
        # the per-head weights, as the reference implementation does; the
        # bf16/fp32 path carries no dequant scale to fold in alongside them.
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5

    def _rope_first_dims(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the FIRST ``rope_dim`` dims of ``(T, H, D)`` ``x``."""
        pe = x[..., : self.rope_dim]
        pe, _ = self.rotary(positions, pe, pe)
        return torch.cat([pe, x[..., self.rope_dim :]], dim=-1)

    def compute_k(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-token index key ``(T, head_dim)`` — what the indexer cache stores."""
        k = self.k_norm(self.wk(hidden_states))  # (T, D)
        return self._rope_first_dims(k.unsqueeze(1), positions).squeeze(1)

    def compute_selection(
        self,
        q_c: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k_history: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k prefix positions per query token, ``(T, topk)`` int32, -1 padded."""
        num_tokens = q_c.shape[0]
        num_keys = k_history.shape[0]
        if num_keys <= int(positions.max()):
            raise ValueError(
                f"k_history has {num_keys} rows but positions reach "
                f"{int(positions.max())}; the causal window includes self"
            )

        q = self.wq_b(q_c).view(num_tokens, self.n_heads, self.head_dim)
        q = self._rope_first_dims(q, positions)
        w = self.weights_proj(hidden_states) * self.weight_scale  # (T, H)

        # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]): per-head ReLU
        # BEFORE the weighted sum; the raw weights get no softmax/sigmoid.
        dots = torch.einsum("thd,sd->ths", q, k_history).relu()
        scores = torch.einsum("th,ths->ts", w, dots)

        # Causal window INCLUDING self: candidates are positions 0..p_t.
        key_pos = torch.arange(num_keys, device=scores.device)
        scores = scores.masked_fill(
            key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))

        selection = torch.full(
            (num_tokens, self.topk), -1, dtype=torch.int32, device=scores.device)
        for t in range(num_tokens):
            n = min(self.topk, int(positions[t]) + 1)
            selection[t, :n] = scores[t].topk(n).indices.to(torch.int32)
        return selection
