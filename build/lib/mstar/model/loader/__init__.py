"""Generic weight loading infrastructure.

Two layers:
  * ``iterators`` — I/O. Streams ``(name, tensor)`` pairs from safetensors
    files / sharded HF directories without materializing the full
    state_dict.
  * ``base`` — per-model dispatch. ``load_weights_into`` walks the stream
    and calls each parameter's ``weight_loader`` (or a default copy).
    Models define their own ``load_weights(weights) -> set[str]`` method
    using this helper, declaring any name remapping and fused-shard
    routing they need.

Top-level driver: ``load_weights(model, source)`` picks the right
iterator and calls ``model.load_weights(...)``.
"""
from mstar.model.loader.base import (
    HF_DEFAULT_SKIP_FRAGMENTS,
    LLAMA_STACKED_PARAMS,
    StackedParamRule,
    default_weight_loader,
    load_hf_weights,
    load_weights,
    load_weights_into,
)
from mstar.model.loader.iterators import (
    iter_safetensors_file,
    iter_safetensors_shards,
)

__all__ = [
    "HF_DEFAULT_SKIP_FRAGMENTS",
    "LLAMA_STACKED_PARAMS",
    "StackedParamRule",
    "default_weight_loader",
    "load_hf_weights",
    "load_weights",
    "load_weights_into",
    "iter_safetensors_file",
    "iter_safetensors_shards",
]
