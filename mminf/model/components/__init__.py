"""Reusable transformer building blocks.

Components in this package are model-agnostic and meant to be shared
across model implementations. Anything model-specific (vision encoders,
audio codecs, multimodal preprocessing, etc.) stays in the per-model
``components/`` directory.

The TP-aware versions of these blocks (parallel linears, etc.) will
land here too in a follow-up; the current shapes intentionally leave
room for that.
"""
from mminf.model.components.attention import Attention
from mminf.model.components.decoder_layer import DecoderLayer, GatedDecoderLayer
from mminf.model.components.linear import FusedColumnLinear
from mminf.model.components.mlp import MLP, FusedGatedMLP, GatedMLP
from mminf.model.components.moe import (
    ParallelSparseMoeBlock,
    ParallelSparseMoeBlockWithSharedExpert,
    SparseMoeBlock,
    SparseMoeBlockWithSharedExpert,
    TopKRouter,
    dispatch_experts_fused,
)
from mminf.model.components.norm import AdaRMSNorm, RMSNorm

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
