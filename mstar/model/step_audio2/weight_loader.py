"""Checkpoint loading for Step-Audio2's audio tower and LLM.

Each loader streams only its own prefixes and raises if a parameter is left unfilled or a
checkpoint key names no parameter: a silently half-loaded tower produces plausible output,
not an error. Shapes are checked by the parameters' own weight loaders.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from mstar.model.loader import LLAMA_STACKED_PARAMS, WHISPER_STACKED_PARAMS, load_hf_weights
from mstar.model.loader.base import StackedParamRule, apply_stacked
from mstar.model.loader.iterators import iter_safetensors_shards

# A Whisper block's parameters in the checkpoint (``ResidualAttentionBlock``,
# ``modeling_step_audio_2.py:208-219``) -> ``AuTEncoderLayer``'s, whose q/k/v the
# Whisper stacked rules then fuse into ``qkv_proj``.
_BLOCK_RENAMES = {
    "attn_ln.": "self_attn_layer_norm.",
    "attn.query.": "self_attn.q_proj.",
    "attn.key.": "self_attn.k_proj.",
    "attn.value.": "self_attn.v_proj.",
    "attn.out.": "self_attn.out_proj.",
    "mlp_ln.": "final_layer_norm.",
    "mlp.0.": "fc1.",
    "mlp.2.": "fc2.",
}
_BLOCK = re.compile(r"^(encoder\.blocks\.\d+\.)(.*)$")
AUDIO_PREFIXES = ("encoder.", "adapter.")
LLM_PREFIXES = ("model.", "lm_head.")


def audio_param_name(key: str) -> str:
    """A checkpoint key under ``encoder.`` / ``adapter.`` -> its path in ``StepAudio2Audio``,
    before the stacked q/k/v rules. Raises on a block parameter it does not know."""
    match = _BLOCK.match(key)
    if match is None:
        return key
    head, rest = match.groups()
    for old, new in _BLOCK_RENAMES.items():
        if rest.startswith(old):
            return head + new + rest[len(old):]
    raise KeyError(f"unknown Whisper block parameter {key!r}")


def llm_param_name(key: str) -> str:
    """A checkpoint key under ``model.`` / ``lm_head.`` -> its ``Qwen3DenseLM`` path
    (which drops HF's ``model.``)."""
    return key.removeprefix("model.")


def _load(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked: list[StackedParamRule],
    rename: Callable[[str], str],
    what: str,
) -> set[str]:
    params = dict(module.named_parameters())
    unmatched: list[str] = []

    def checked() -> Iterator[tuple[str, torch.Tensor]]:
        for key, tensor in weights:
            if apply_stacked(rename(key), stacked)[0] not in params:
                unmatched.append(key)
            yield key, tensor

    loaded = load_hf_weights(module, checked(), stacked_params=stacked, name_remapper=rename)
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} checkpoint key(s) name no parameter of {what}, e.g. {unmatched[:5]}")
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(
            f"Step-Audio2 checkpoint left {len(missing)} parameter(s) of {what} unloaded, e.g. {missing[:5]}"
        )
    return loaded


def _stream(weights_dir: str, device, prefixes: tuple[str, ...]):
    for prefix in prefixes:
        yield from iter_safetensors_shards(weights_dir, device=device, prefix=prefix)


def load_audio_weights(
    model: nn.Module, weights_dir: str | None = None, device=None,
    weights: Iterable[tuple[str, torch.Tensor]] | None = None,
) -> set[str]:
    """``encoder.*`` and ``adapter.*`` into a ``StepAudio2Audio``, from ``weights_dir`` or
    a ``weights`` stream of checkpoint ``(key, tensor)`` pairs."""
    # before loading: the checkpoint has no key bias to write into its slice
    model.zero_missing_biases()
    stream = weights if weights is not None else _stream(weights_dir, device, AUDIO_PREFIXES)
    return _load(model, stream, WHISPER_STACKED_PARAMS, audio_param_name, "the audio tower")


def load_llm_weights(
    model: nn.Module, weights_dir: str | None = None, device=None,
    weights: Iterable[tuple[str, torch.Tensor]] | None = None,
) -> set[str]:
    """``model.*`` and ``lm_head.*`` into a ``Qwen3DenseLM``."""
    stream = weights if weights is not None else _stream(weights_dir, device, LLM_PREFIXES)
    return _load(model, stream, LLAMA_STACKED_PARAMS, llm_param_name, "the LLM")
