"""Scaled dot-product attention with a dense additive bias."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# SDPA sends a dense bias to cuDNN, which builds a graph per new shape (~1 s of host time each)
DENSE_BIAS_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def dense_bias_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, attn_mask=bias)`` off cuDNN."""
    with sdpa_kernel(DENSE_BIAS_BACKENDS):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
