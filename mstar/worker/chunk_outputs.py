"""Outputs of a chunked row's non-final chunks, held until its final chunk.

A chunk of a split input is a full forward pass, but its node is not done:
nothing is routed until the final chunk, which emits each output edge once,
as if the input had run in one step.
"""

import torch

from mstar.communication.tensors import NameToTensorList


class ChunkOutputAccumulator:
    def __init__(self):
        self._held: dict[tuple[int, str], NameToTensorList] = {}

    def hold(self, rid: int, node_name: str, outputs: NameToTensorList) -> None:
        """Keep a non-final chunk's outputs, in chunk order."""
        held = self._held.setdefault((rid, node_name), {})
        for name, tensors in outputs.items():
            held.setdefault(name, []).extend(tensors)

    def release(
        self, rid: int, node_name: str, outputs: NameToTensorList,
    ) -> NameToTensorList:
        """The final chunk's outputs with the held ones in front, each edge
        concatenated along dim 0 so the consumer sees one tensor."""
        held = self._held.pop((rid, node_name), None)
        if held is None:
            return outputs
        merged: NameToTensorList = {}
        for name in {**held, **outputs}:
            tensors = held.get(name, []) + outputs.get(name, [])
            merged[name] = [torch.cat(tensors)] if len(tensors) > 1 else tensors
        return merged

    def drop(self, rid: int) -> None:
        for key in [key for key in self._held if key[0] == rid]:
            del self._held[key]
