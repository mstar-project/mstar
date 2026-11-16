"""Bind a DeepSeek V4.1 compressor's partial groups to engine-owned state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from mstar.engine.state import Array, Entry, TensorType, Value
from mstar.model.components.deepseek_v41.compressor import DeepSeekV41Compressor


def compressor_partial_value(
    name: str, *, node: str, source_layers: frozenset[int], ratio: int, head_dim: int,
) -> Array:
    """One partial group per request and source layer; ratio-one needs no Value."""
    if ratio < 2 or head_dim < 1 or not source_layers:
        raise ValueError("a compressor partial Value needs a ratio above one, a width and source layers")
    partial = TensorType((ratio, head_dim), dtype=torch.float32)
    return Value(name, nodes={node}, layers=source_layers, entry=Entry({"kv": partial, "score": partial}))


class DeepSeekV41CompressorSource(nn.Module):
    """A source layer whose partial group lives in a declared engine Value."""

    def __init__(self, layer_id: int, compressor: DeepSeekV41Compressor, state_name: str) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.compressor = compressor
        self.state_name = state_name
        self.state: Any = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        if self.compressor.ratio == 1:
            return
        self.state = handles[self.state_name]
        kv, score = self.compressor.new_state(1, self.compressor.wkv.weight.device)
        self.state.set_start({"kv": kv[0], "score": score[0]})

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        if self.compressor.ratio == 1:
            latent, _ = self.compressor(x, start_pos)
            return latent
        if self.state is None:
            raise RuntimeError("compressor source has no bound partial Value")
        if x.shape[0] != 1:
            raise ValueError("compressor source needs one request per step until positions are per row")
        fields = self.state.read(self.layer_id).fields
        latent, next_state = self.compressor(x, start_pos, (fields["kv"], fields["score"]))
        assert next_state is not None
        self.state.write(self.layer_id, kv=next_state[0], score=next_state[1])
        return latent


def load_compressor_weights(
    compressor: DeepSeekV41Compressor, layer_id: int, tensors: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Load one published source layer's norm and projection tensors exactly once."""
    prefix = f"layers.{layer_id}.attn.compressor."
    expected = dict(compressor.named_parameters())
    seen: set[str] = set()
    for name, tensor in tensors:
        if not name.startswith(prefix):
            raise ValueError(f"wrong DeepSeek compressor checkpoint key {name!r}")
        key = name.removeprefix(prefix)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate DeepSeek compressor weight {key!r}")
        target = expected[key]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(
                f"DeepSeek compressor {key} has shape {tuple(tensor.shape)} and dtype {tensor.dtype}, "
                f"expected {tuple(target.shape)} and {target.dtype}"
            )
        target.data.copy_(tensor)
        seen.add(key)
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"missing DeepSeek compressor weights: {', '.join(sorted(missing))}")
