"""Checkpoint loading for MiniCPM-o's nodes.

Every loader streams only its node's prefix and raises if a parameter is left
unfilled: a silently half-loaded tower produces plausible output, not an
error.
"""
from __future__ import annotations

from torch import nn

from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
from mstar.model.loader.iterators import iter_safetensors_shards


def _load(module: nn.Module, weights_dir: str, device, prefixes: dict[str, str], stacked) -> None:
    """``prefixes`` maps a checkpoint prefix to the module path it fills."""
    def stream():
        for ckpt_prefix, module_prefix in prefixes.items():
            for name, tensor in iter_safetensors_shards(weights_dir, device=device, prefix=ckpt_prefix):
                yield module_prefix + name.removeprefix(ckpt_prefix), tensor

    loaded = load_hf_weights(module, stream(), stacked_params=stacked)
    missing = sorted(set(dict(module.named_parameters())) - loaded)
    if missing:
        raise RuntimeError(
            f"MiniCPM-o checkpoint left {len(missing)} parameter(s) of "
            f"{type(module).__name__} unloaded, e.g. {missing[:5]}"
        )


def load_vision_weights(model: nn.Module, weights_dir: str, device) -> None:
    """``vpm.*`` and ``resampler.*`` into a ``MiniCPMOVision``."""
    _load(model, weights_dir, device, {"vpm.": "vpm.", "resampler.": "resampler."}, WHISPER_STACKED_PARAMS)
    model.resampler.reset_buffers()


def load_audio_weights(model: nn.Module, weights_dir: str, device) -> None:
    """``apm.*`` and ``audio_projection_layer.*`` into a ``MiniCPMOAudio``."""
    # before loading: the checkpoint has no k_proj bias to write into its slice
    model.zero_missing_biases()
    _load(
        model, weights_dir, device,
        {"apm.": "apm.", "audio_projection_layer.": "audio_projection_layer."},
        WHISPER_STACKED_PARAMS,
    )


def load_llm_weights(model: nn.Module, weights_dir: str, device) -> None:
    """``llm.model.*`` and ``llm.lm_head`` into a ``Qwen3DenseLM`` (whose
    paths drop HF's ``model.``)."""
    def stream():
        for name, tensor in iter_safetensors_shards(weights_dir, device=device, prefix="llm."):
            yield name.removeprefix("llm.").removeprefix("model."), tensor

    model.load_weights(stream())

