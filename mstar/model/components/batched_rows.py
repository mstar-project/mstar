"""``forward`` / ``forward_batched`` for nodes whose computation runs on a stacked batch."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mstar.model.submodule_base import ModelInputsFromEngine


class BatchedRows:
    """Mixin for a node whose :meth:`run_batch` takes a stacked ``[B, ...]`` batch and
    returns one ``[B, ...]`` tensor per output key; ``forward`` and ``forward_batched``
    then hand each request its own row.

    List it before ``NodeSubmodule`` in the bases. Each request's row drops the batch dim
    by default, so an edge's shape does not depend on how its requests were batched; a
    node whose consumers expect a leading batch dim of 1 sets ``keep_batch_dim``.
    """

    output_keys: tuple[str, ...]
    keep_batch_dim: bool = False

    def run_batch(self, **kwargs) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def _row(self, tensor: torch.Tensor, i: int) -> torch.Tensor:
        return tensor[i:i + 1] if self.keep_batch_dim else tensor[i]

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs):
        out = self.run_batch(**kwargs)
        if self.keep_batch_dim:
            # A batch of one is already [1, ...]; a stacked batch passes through whole.
            return {key: [out[key]] for key in self.output_keys}
        return {key: [out[key][0]] for key in self.output_keys}

    def forward_batched(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs):
        out = self.run_batch(**kwargs)
        return {
            rid: {key: [self._row(out[key], i)] for key in self.output_keys}
            for i, rid in enumerate(engine_inputs.request_ids)
        }
