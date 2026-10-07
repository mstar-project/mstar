"""Outputs of a chunked row's non-final chunks, held until its final chunk.

A chunk of a split input is a full forward pass, but its node is not done:
nothing is routed until the final chunk, which emits each output edge once,
joined per the submodule's ``ChunkedPrefillOutputPolicy``.
"""

from collections.abc import Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.model.submodule_base import ChunkedPrefillOutputMode, ChunkedPrefillOutputPolicy

_DEFAULT = ChunkedPrefillOutputPolicy()


class ChunkOutputAccumulator:
    def __init__(self):
        # (rid, node) -> edge -> each held chunk's tensors, in chunk order
        self._held: dict[tuple[int, str], dict[str, list[list[torch.Tensor]]]] = {}

    def hold(self, rid: int, node_name: str, outputs: NameToTensorList) -> None:
        """Keep a non-final chunk's outputs, in chunk order."""
        held = self._held.setdefault((rid, node_name), {})
        for name, tensors in outputs.items():
            held.setdefault(name, []).append(list(tensors))

    def release(
        self, rid: int, node_name: str, outputs: NameToTensorList,
        policies: Mapping[str, ChunkedPrefillOutputPolicy] = {},
    ) -> NameToTensorList:
        """The final chunk's outputs joined with the held ones, each edge per
        its policy (CONCAT on dim 0 when unlisted)."""
        held = self._held.pop((rid, node_name), None)
        if held is None:
            return outputs
        merged: NameToTensorList = {}
        for name in {**held, **outputs}:
            chunks = held.get(name, []) + ([list(outputs[name])] if name in outputs else [])
            policy = policies.get(name, _DEFAULT)
            if policy.mode is ChunkedPrefillOutputMode.LIST or len(chunks) == 1:
                merged[name] = [t for chunk in chunks for t in chunk]
                continue
            widths = {len(chunk) for chunk in chunks}
            if len(widths) != 1:
                raise ValueError(
                    f"output {name!r} of {node_name!r}: chunks emitted {sorted(widths)} "
                    "tensors; CONCAT joins the same number from every chunk"
                )
            merged[name] = [
                torch.cat([chunk[i] for chunk in chunks], dim=policy.dim)
                for i in range(widths.pop())
            ]
        return merged

    def drop(self, rid: int) -> None:
        for key in [key for key in self._held if key[0] == rid]:
            del self._held[key]
