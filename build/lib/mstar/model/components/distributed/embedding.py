"""Vocabulary-parallel embedding for TP'd token LM stacks.

Embedding weight ``[V, H]`` is row-sharded across TP ranks (each rank
holds ``V/tp`` rows). Forward masks tokens outside the local rank's slice
to zero, runs ``F.embedding`` on the local shard, then sums across ranks
with a single ``all_reduce``. The output is ``[..., H]`` replicated on
every rank — i.e. the LM stack's subsequent layers see the same input
they would in non-TP mode, no further coordination required.

Pairs with ``ColumnParallelLinear(gather_output=True)`` as the LM head:
the head produces ``[B, V/tp]`` per rank and ``gather_output`` collects
to ``[B, V]`` so the sampler stays vocab-oblivious.

Requires ``vocab_size`` divisible by ``tp_size``. Padding-to-divisible is
a follow-up.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.distributed.utils import divide


class VocabParallelEmbedding(nn.Module):
    """Row-parallel token embedding.

    Each rank holds rows ``[tp_rank * V/tp : (tp_rank + 1) * V/tp]`` of
    the full ``[V, H]`` embedding matrix. The forward zeroes
    contributions for tokens outside this rank's slice and ``all_reduce``
    sums shards into the replicated full embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        comm_group: TPCommGroup | None = None,
        padding_idx: int | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        self.tp_rank = comm_group.rank
        self.tp_size = comm_group.world_size

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        # ``padding_idx`` is plumbed for API parity with ``nn.Embedding`` but
        # only affects training (zero gradient for the padding row). Inference
        # is unaffected — the masking below already zeroes any out-of-slice
        # token, including a padding row that lives on another rank.
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition, embedding_dim, dtype=dtype,
            )
        )
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.weight.weight_loader = self.weight_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        """Slice ``[V, H]`` HF embedding into this rank's ``[V/tp, H]``."""
        assert loaded_shard_id is None, (
            f"VocabParallelEmbedding.weight_loader doesn't take a shard id; "
            f"got {loaded_shard_id!r}"
        )
        shard = loaded_weight.narrow(
            0, self.vocab_start_index, self.num_embeddings_per_partition,
        )
        assert param.data.shape == shard.shape, (
            f"weight_loader shape mismatch: param {tuple(param.data.shape)} "
            f"vs shard {tuple(shard.shape)}"
        )
        param.data.copy_(shard)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_, self.weight)

        # Tokens this rank owns: True at position ``i`` iff
        # ``vocab_start_index <= input_[i] < vocab_end_index``.
        mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)

        # Translate to local-shard coordinates; clamp out-of-range so
        # ``F.embedding`` doesn't index-fault on tokens owned by other ranks
        # (their contribution is zeroed out below anyway).
        local_input = (input_ - self.vocab_start_index).clamp_(
            0, self.num_embeddings_per_partition - 1,
        )
        local_output = F.embedding(local_input, self.weight)
        # Broadcast the mask along the embedding dim so zeroed rows have
        # no contribution to the all-reduce.
        local_output = local_output * mask.unsqueeze(-1).to(local_output.dtype)
        return self.comm_group.all_reduce(local_output)
