"""Checkpoint access shared by the Chatterbox components.

The HF snapshot is a handful of independent safetensors files rather than one
sharded model: ``t3_cfg.safetensors`` (or ``t3_turbo_v1.safetensors``),
``s3gen.safetensors`` (or ``s3gen_meanflow.safetensors``, which also carries
the S3 tokenizer under ``tokenizer.*``) and ``ve.safetensors``. Each component
streams its own file through ``mstar.model.loader`` and then checks that every
parameter it owns was written and that nothing in the file went unclaimed, so
a renamed key fails at load time rather than as silence at serve time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_file

logger = logging.getLogger(__name__)

WeightStream = Iterable[tuple[str, torch.Tensor]]

# Buffer names a checkpoint carries and a module must receive (batch norm).
_LOADED_BUFFER_NAMES = frozenset({"running_mean", "running_var"})

# Buffers the reference modules register that are recomputed here, never loaded.
DEFAULT_SKIP_FRAGMENTS: tuple[str, ...] = (
    "num_batches_tracked",
    "tokenizer._mel_filters",
    "tokenizer.window",
)


def resolve_snapshot(model_path_hf: str, cache_dir: str | None = None) -> str:
    """Local directory holding the checkpoint: a path as given, else the HF
    cache snapshot (offline first, so a worker without egress still starts)."""
    if Path(model_path_hf).is_dir():
        return model_path_hf
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_path_hf, cache_dir=cache_dir, local_files_only=True
        )
    except Exception:
        return snapshot_download(repo_id=model_path_hf, cache_dir=cache_dir)


def iter_weights(
    path: str | Path,
    device: torch.device | str = "cpu",
    prefix: str | None = None,
    strip_prefix: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """``(name, tensor)`` from one safetensors file, optionally only the keys
    under ``prefix`` and with that prefix removed."""
    for name, tensor in iter_safetensors_file(path, device=device, prefix=prefix):
        if prefix is not None and strip_prefix:
            name = name[len(prefix):]
        yield name, tensor


def fold_weight_norm(weights: WeightStream) -> Iterator[tuple[str, torch.Tensor]]:
    """Replace ``torch.nn.utils.parametrizations.weight_norm`` pairs by the
    effective weight.

    The reference vocoder keeps ``<conv>.parametrizations.weight.original0``
    (the magnitude ``g``, shape ``[out, 1, 1]``) and ``original1`` (the
    direction ``v``). Inference only needs ``g * v / ||v||`` with the norm
    taken over every dim but the first, which is what ``torch._weight_norm``
    computes for ``dim=0``. Folding at load time removes the parametrization
    from the serving graph entirely.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    marker = ".parametrizations.weight.original"
    for name, tensor in weights:
        idx = name.find(marker)
        if idx < 0:
            yield name, tensor
            continue
        base = name[:idx]
        which = name[idx + len(marker):]
        parts = pending.setdefault(base, {})
        parts[which] = tensor
        if "0" in parts and "1" in parts:
            g, v = parts.pop("0"), parts.pop("1")
            del pending[base]
            yield f"{base}.weight", torch._weight_norm(v, g, dim=0)
    if pending:
        raise ValueError(
            f"weight-norm parametrizations without a partner: {sorted(pending)}"
        )


def load_component(
    module: nn.Module,
    weights: WeightStream,
    *,
    component: str,
    name_remapper: Callable[[str], str | None] | None = None,
    stacked_params: list[StackedParamRule] | None = None,
    skip_fragments: tuple[str, ...] = DEFAULT_SKIP_FRAGMENTS,
) -> set[str]:
    """Stream ``weights`` into ``module`` and insist on a complete, exact fit.

    Raises if a parameter of ``module`` received nothing, or if a checkpoint
    key mapped to no parameter (after ``name_remapper`` and the stacked-shard
    rules), unless it matches ``skip_fragments`` or the remapper dropped it by
    returning ``None``.
    """
    params = dict(module.named_parameters())
    # Persistent buffers (batch-norm running statistics) live in the
    # checkpoint too; ``load_weights_into`` only knows parameters, so they are
    # copied here and counted toward completeness like any parameter.
    buffers = {
        name: buf for name, buf in module.named_buffers()
        if name.split(".")[-1] in _LOADED_BUFFER_NAMES
    }
    unexpected: list[str] = []
    stacked = stacked_params or []

    def _target(name: str) -> str | None:
        if any(frag in name for frag in skip_fragments):
            return None
        mapped = name if name_remapper is None else name_remapper(name)
        if mapped is None:
            return None
        for rule in stacked:
            if rule.source_suffix in mapped:
                return mapped.replace(rule.source_suffix, rule.target_suffix)
        return mapped

    loaded_buffers: set[str] = set()

    def _audited() -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            target = _target(name)
            if target is not None and target in buffers:
                buffers[target].copy_(tensor.to(buffers[target].dtype))
                loaded_buffers.add(target)
                continue
            if target is not None and target not in params:
                unexpected.append(name)
                continue
            yield name, tensor

    loaded = load_weights_into(
        module,
        _audited(),
        stacked_params=stacked,
        name_remapper=name_remapper,
        skip_predicate=lambda name: any(frag in name for frag in skip_fragments),
    )
    missing = sorted((set(params) - loaded) | (set(buffers) - loaded_buffers))
    loaded = loaded | loaded_buffers
    if missing or unexpected:
        raise RuntimeError(
            f"{component}: checkpoint does not match the module. "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
            f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    logger.info("%s: loaded %d parameters", component, len(loaded))
    return loaded


def materialize(module: nn.Module, device: torch.device | str, dtype: torch.dtype | None) -> nn.Module:
    """Meta-built module -> real storage in ``dtype`` on ``device`` (cast on
    meta first so nothing is allocated twice)."""
    if dtype is not None:
        module = module.to(dtype)
    return module.to_empty(device=device)
