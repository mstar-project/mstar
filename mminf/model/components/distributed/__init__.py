"""Tensor-parallel building blocks.

Parallel linears (``ColumnParallelLinear``, ``RowParallelLinear``,
``MergedColumnParallelLinear``, ``QKVParallelLinear``), vocab-parallel
embedding (``VocabParallelEmbedding``), and the composed parallel
``Attention`` / ``GatedMLP`` blocks. Each parallel parameter carries a
``weight_loader`` attribute used by the model-level weight loader to
slice checkpoint tensors per-rank on load.

For vocab parallelism, pair ``VocabParallelEmbedding`` with
``ColumnParallelLinear(gather_output=True)`` on the LM head: the
embedding all-reduces shard contributions before the first transformer
layer, and the LM head all-gathers logits along the vocab dim before
returning, so the sampler stays vocab-oblivious.
"""
from mminf.model.components.distributed.attention import ParallelAttention
from mminf.model.components.distributed.embedding import VocabParallelEmbedding
from mminf.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mminf.model.components.distributed.mlp import ParallelGatedMLP

__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "ParallelAttention",
    "ParallelGatedMLP",
    "VocabParallelEmbedding",
]
