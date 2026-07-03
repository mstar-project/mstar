"""Per-parameter dispatch for weight loading.

Each model's ``load_weights(weights) -> set[str]`` method walks an
iterable of ``(checkpoint_key, tensor)`` and calls ``load_weights_into``
to do the actual work:

  1. Optional model-specific name remap (``name_remapper``).
  2. Optional skip predicate (``skip_predicate``) for keys to ignore
     entirely (e.g. ``rotary_emb.inv_freq`` buffers).
  3. Optional stacked-shard rules (``stacked_params``) that route
     checkpoint keys like ``q_proj`` → fused ``qkv_proj`` with a
     ``shard_id`` passed to the parameter's ``weight_loader``.
  4. Lookup in ``module.named_parameters()`` and dispatch to
     ``param.weight_loader(param, tensor, shard_id?)`` if present,
     else ``default_weight_loader``.

Models that don't need any of these (single-file, names already match,
no fused params) just call ``load_weights_into(module, weights)``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class StackedParamRule:
    """One rule for routing a per-shard checkpoint key into a fused
    parameter.

    Example for Llama-style attention::

        StackedParamRule(target_suffix=".qkv_proj",
                         source_suffix=".q_proj", shard_id="q")
        StackedParamRule(target_suffix=".qkv_proj",
                         source_suffix=".k_proj", shard_id="k")
        StackedParamRule(target_suffix=".qkv_proj",
                         source_suffix=".v_proj", shard_id="v")

    The rule matches when ``source_suffix in name``; the matched suffix
    is replaced with ``target_suffix`` to find the fused parameter, and
    ``shard_id`` is forwarded to that parameter's ``weight_loader``.
    """
    target_suffix: str
    source_suffix: str
    shard_id: str | int


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
) -> None:
    """Plain replicated copy. Used for any parameter that doesn't
    attach its own ``weight_loader`` attribute."""
    assert loaded_shard_id is None, (
        f"default_weight_loader doesn't take a shard id; got {loaded_shard_id!r}"
    )
    assert param.data.shape == loaded_weight.shape, (
        f"shape mismatch: param {tuple(param.data.shape)} vs "
        f"weight {tuple(loaded_weight.shape)}"
    )
    param.data.copy_(loaded_weight)


def _dispatch_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str | int | None = None,
) -> None:
    """Call the parameter's ``weight_loader`` if it has one, else the default."""
    loader = getattr(param, "weight_loader", None)
    if loader is None:
        default_weight_loader(param, loaded_weight, shard_id)
        return
    if shard_id is None:
        loader(param, loaded_weight)
    else:
        loader(param, loaded_weight, shard_id)


def _apply_stacked(
    name: str, stacked: list[StackedParamRule],
) -> tuple[str, str | int | None]:
    """Return ``(target_name, shard_id)`` after applying stacked rules,
    or ``(name, None)`` if no rule matched."""
    for rule in stacked:
        if rule.source_suffix in name:
            return name.replace(rule.source_suffix, rule.target_suffix), rule.shard_id
    return name, None


def load_weights_into(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params: list[StackedParamRule] | None = None,
    name_remapper: Callable[[str], str | None] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
) -> set[str]:
    """Walk a ``(name, tensor)`` stream and load each into the matching
    parameter of ``module``.

    Args:
        module: target ``nn.Module`` — ``named_parameters()`` defines
            the lookup table.
        weights: iterable of ``(checkpoint_key, tensor)``.
        stacked_params: rules for fused-shard routing
            (``q_proj`` → ``qkv_proj``, ``gate_proj`` → ``gate_up_proj``,
            etc.).
        name_remapper: optional ``(name) -> str | None`` for model-
            specific surgery (e.g. lerobot prefix stripping). Return
            ``None`` to drop the key entirely.
        skip_predicate: optional ``(name) -> bool``; returning ``True``
            skips the key. Applied before ``name_remapper``.

    Returns: set of parameter paths that received a tensor.
    """
    stacked = stacked_params or []
    params_dict = dict(module.named_parameters())
    loaded: set[str] = set()

    for name, tensor in weights:
        if skip_predicate is not None and skip_predicate(name):
            continue
        if name_remapper is not None:
            mapped = name_remapper(name)
            if mapped is None:
                continue
            name = mapped

        target, shard_id = _apply_stacked(name, stacked)
        if target not in params_dict:
            # Not an error: the checkpoint may carry extra keys (lm_head
            # ties, kv-scale, etc.). Caller can verify completeness via
            # the returned set.
            continue

        _dispatch_loader(params_dict[target], tensor, shard_id)
        loaded.add(target)

    return loaded


# Standard HF checkpoint keys that aren't loadable model parameters:
# precomputed rotary_emb buffers (every Llama-flavored checkpoint
# carries these) and similar non-parameter artifacts.
HF_DEFAULT_SKIP_FRAGMENTS: tuple[str, ...] = (
    "rotary_emb",
)

# Standard fused-projection routing used by Llama/Qwen/Mistral/Gemma/etc.:
# checkpoint stores ``q/k/v_proj`` and ``gate/up_proj`` separately; the
# model holds fused ``qkv_proj`` and ``gate_up_proj`` parameters with
# per-shard ``weight_loader`` methods.
LLAMA_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj",     ".q_proj",    "q"),
    StackedParamRule(".qkv_proj",     ".k_proj",    "k"),
    StackedParamRule(".qkv_proj",     ".v_proj",    "v"),
    StackedParamRule(".gate_up_proj", ".gate_proj", 0),
    StackedParamRule(".gate_up_proj", ".up_proj",   1),
]


def load_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params: list[StackedParamRule] | None = None,
    name_remapper: Callable[[str], str | None] | None = None,
    extra_skip_fragments: tuple[str, ...] = (),
) -> set[str]:
    """Convenience wrapper for HF-style checkpoint loading.

    Combines ``load_weights_into`` with the standard HF skip predicate
    (``HF_DEFAULT_SKIP_FRAGMENTS``). Use when the model's parameter
    paths line up with HF naming after the supplied stacked rules /
    ``name_remapper`` are applied.
    """
    skip_fragments = HF_DEFAULT_SKIP_FRAGMENTS + tuple(extra_skip_fragments)

    def skip(name: str) -> bool:
        return any(frag in name for frag in skip_fragments)

    return load_weights_into(
        module, weights,
        stacked_params=stacked_params,
        name_remapper=name_remapper,
        skip_predicate=skip,
    )


def load_weights(
    model: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
    **kwargs,
) -> set[str]:
    """Top-level driver.

    ``source`` is either a path to a single safetensors file or a
    directory containing an HF-style sharded checkpoint
    (``model.safetensors.index.json`` + shard files, or a single
    ``model.safetensors``).

    Calls ``model.load_weights(iter)`` with the resolved iterator. Extra
    kwargs are forwarded to ``model.load_weights``.
    """
    from mstar.model.loader.iterators import (
        iter_safetensors_file,
        iter_safetensors_shards,
    )

    source = Path(source)
    if source.is_file():
        weights = iter_safetensors_file(source, device=device)
    elif source.is_dir():
        weights = iter_safetensors_shards(source, device=device)
    else:
        raise FileNotFoundError(source)
    return model.load_weights(weights, **kwargs)
