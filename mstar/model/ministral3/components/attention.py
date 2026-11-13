"""Ministral 3 attention over engine-owned KV, attention, and position resources."""

import math

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.attention import ParallelAttention


class Ministral3Attention(ParallelAttention):
    """GQA without QK norms, with query scaling after engine-provided YaRN.

    ``hidden_size`` is the residual width, independent of ``num_heads * head_dim``.
    The owning model declares YaRN on its position resource and passes the same
    original context length here. Rotation parameters come from that resource.
    Only the reference's query-only scaling is model-specific:
    ``1 + beta * log(1 + floor(position / original_context_length))``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        original_max_position_embeddings: int,
        llama_4_scaling_beta: float = 0.1,
        comm_group: CommGroup | None = None,
        attn_key: str = "attn",
        kv_key: str = "kv",
        pos_key: str = "rope",
    ):
        if original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be positive")
        if not math.isfinite(llama_4_scaling_beta) or llama_4_scaling_beta < 0:
            raise ValueError("llama_4_scaling_beta must be finite and nonnegative")
        super().__init__(
            comm_group=comm_group,
            hidden_size=num_heads * head_dim,
            input_hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=False,
            attn_key=attn_key,
            kv_key=kv_key,
            pos_key=pos_key,
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        self.llama_4_scaling_beta = llama_4_scaling_beta

    def bind_resources(self, resources: dict) -> None:
        for key in (self._attn_key, self._kv_key, self._pos_key):
            if resources.get(key) is None:
                raise ValueError(f"Ministral 3 attention requires resource {key!r}")
        super().bind_resources(resources)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.pos.pos_ids(label)
        if positions is None:
            raise RuntimeError(f"Ministral 3 attention has no planned positions for {label!r}")
        q, k = self.pos.apply_qk(q, k, label=label)
        # Eager resources can retain a buffer wider than the live token rows.
        positions = positions[:q.shape[0]].float()
        scale = 1 + self.llama_4_scaling_beta * torch.log(
            1 + torch.floor(positions / self.original_max_position_embeddings)
        )
        # Match the reference: round the scale to Q's dtype before multiplying.
        return q * scale.to(q.dtype)[:, None, None], k
