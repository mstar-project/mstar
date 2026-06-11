"""Reusable transformer building blocks.

Components in this package are model-agnostic and meant to be shared
across model implementations. Anything model-specific (vision encoders,
audio codecs, multimodal preprocessing, etc.) stays in the per-model
``components/`` directory.

The TP-aware versions of these blocks (parallel linears, etc.) will
land here too in a follow-up; the current shapes intentionally leave
room for that.
"""
from mstar.model.components.attention import Attention
from mstar.model.components.decoder_layer import DecoderLayer, GatedDecoderLayer
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.components.mlp import MLP, FusedGatedMLP, GatedMLP
from mstar.model.components.moe import (
    ParallelSparseMoeBlock,
    ParallelSparseMoeBlockWithSharedExpert,
    SparseMoeBlock,
    SparseMoeBlockWithSharedExpert,
    TopKRouter,
    dispatch_experts_fused,
)
from mstar.model.components.norm import AdaRMSNorm, RMSNorm

__all__ = [
        "Attention",
    "DecoderLayer",
    "GatedDecoderLayer",
    "FusedColumnLinear",
    "FusedGatedMLP",
    "MLP",
    "GatedMLP",
    "ParallelSparseMoeBlock",
    "ParallelSparseMoeBlockWithSharedExpert",
    "SparseMoeBlock",
    "SparseMoeBlockWithSharedExpert",
    "TopKRouter",
    "dispatch_experts_fused",
    "AdaRMSNorm",
    "RMSNorm",
]
