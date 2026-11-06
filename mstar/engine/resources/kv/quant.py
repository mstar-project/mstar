"""KV cache quantization: the config, its formats, and per-layer, per-head scales.

``KVConfig.dtype`` stays the unquantized dtype of the K/V entering the cache;
``KVConfig.quant`` says how the pages store them. Scales are fixed per layer and
KV head, so a page is plain bytes and moves like an unquantized one.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.kv.config import KVConfig

logger = logging.getLogger(__name__)

# format -> stored value dtype
KV_FORMATS: dict[str, torch.dtype] = {"fp8_e4m3": torch.float8_e4m3fn}
SCALE_KINDS = ("unit", "static")
# what the observer records, so the only rule a calibrated file can apply
_SCALE_RULES = ("amax",)
_SCALE_KEYS = ("k_scale", "v_scale")
MIN_SCALE = 1e-8


@dataclass(frozen=True)
class KVQuantConfig:
    # format key (see KV_FORMATS)
    format: str
    # "unit": scale 1, no calibration; "static": per layer and KV head, from scales_path
    scales: str = "unit"
    scale_rule: str = "amax"
    # static: safetensors with k_scale, v_scale [layers, kv_heads], unsharded
    scales_path: str | None = None

    def __post_init__(self):
        if self.format not in KV_FORMATS:
            raise ValueError(f"unknown KV quant format {self.format!r}; registered: {sorted(KV_FORMATS)}")
        if self.scales not in SCALE_KINDS:
            raise ValueError(f"unknown KV scales {self.scales!r}; accepted: {list(SCALE_KINDS)}")
        if self.scale_rule not in _SCALE_RULES:
            raise ValueError(
                f"KV scale_rule {self.scale_rule!r}: calibration records amax only; accepted: {list(_SCALE_RULES)}"
            )
        if self.scales == "static" and self.scales_path is None:
            raise ValueError("static KV scales need scales_path (see mstar.engine.resources.kv.calibrate)")
        if self.scales != "static" and self.scales_path is not None:
            raise ValueError(f"scales_path is read only for static scales, not {self.scales!r}")

    @property
    def value_dtype(self) -> torch.dtype:
        return KV_FORMATS[self.format]

    def digest(self) -> str:
        """Names the stored bytes: the config, and the scales file's contents."""
        parts = [repr(self)]
        if self.scales_path is not None:
            with open(self.scales_path, "rb") as f:
                parts.append(hashlib.file_digest(f, "sha256").hexdigest())
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


_YAML_KEYS = {f.name for f in fields(KVQuantConfig)}


def parse_yaml_kv_quant(resource_key: str, block: Mapping[str, Any] | KVQuantConfig | None) -> KVQuantConfig | None:
    """``resources.<key>.quant``; null removes the model's."""
    if block is None or isinstance(block, KVQuantConfig):
        return block
    if not isinstance(block, Mapping):
        raise ValueError(f"resources.{resource_key}.quant is a mapping or null, not {block!r}")
    unknown = sorted(block.keys() - _YAML_KEYS)
    if unknown:
        raise ValueError(
            f"resources.{resource_key}.quant: unknown key(s) {unknown}; accepted: {sorted(_YAML_KEYS)}"
        )
    if not block.get("format"):
        raise ValueError(f"resources.{resource_key}.quant: format is required")
    try:
        return KVQuantConfig(**block)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resources.{resource_key}.quant: {exc}") from exc


def local_heads(total: int, rank: int, world_size: int) -> slice:
    """The KV heads tensor-parallel rank ``rank`` holds, as ``KVConfig.shard`` and the QKV linear split them."""
    if world_size >= total:
        head = rank // (world_size // total)
        return slice(head, head + 1)
    per = total // world_size
    return slice(rank * per, (rank + 1) * per)


@dataclass(frozen=True)
class KVScales:
    """One rank's static scales, fp32 [layers, local kv heads]: stored = value / scale."""
    k: torch.Tensor
    v: torch.Tensor


@functools.cache
def load_kv_scales(
    quant: KVQuantConfig, num_layers: int, total_heads: int,
    rank: int, world_size: int, device: torch.device,
) -> KVScales | None:
    """The static scales for one rank; None for unit. Cached, so the KV and attention resources share them."""
    if quant.scales != "static":
        return None
    from safetensors.torch import load_file

    tensors = load_file(quant.scales_path)
    if sorted(tensors) != sorted(_SCALE_KEYS):
        raise ValueError(
            f"KV scales {quant.scales_path}: keys {sorted(tensors)}, expected {sorted(_SCALE_KEYS)}"
        )
    heads = local_heads(total_heads, rank, world_size)
    out = []
    for key in _SCALE_KEYS:
        t = tensors[key]
        if tuple(t.shape) != (num_layers, total_heads):
            raise ValueError(
                f"KV scales {quant.scales_path}: {key} is {tuple(t.shape)}, the cache has "
                f"{num_layers} layers and {total_heads} KV heads"
            )
        if not torch.isfinite(t).all() or (t <= 0).any():
            raise ValueError(f"KV scales {quant.scales_path}: {key} has non-positive or non-finite entries")
        out.append(t[:, heads].float().contiguous().to(device))
    return KVScales(*out)


def kv_scales_for(cfg: KVConfig, info: EngineResourceInfo) -> KVScales | None:
    """This rank's static scales for the cache ``cfg`` configures, or None (unquantized, or
    unit scales). The KV resource and every attention over it get the same object."""
    if cfg.quant is None or cfg.quant.scales != "static":
        return None
    group = info.joint_comm_group
    if group is not None and group.sp_group.world_size > 1:
        # Ulysses moves heads between SP ranks, so a rank's heads are not one TP slice
        raise ValueError("static KV scales are per TP slice of the KV heads; SP is not supported, use unit scales")
    rank, world_size = (group.tp_group.rank, group.tp_group.world_size) if group is not None else (0, 1)
    return load_kv_scales(cfg.quant, cfg.num_layers, cfg._unsharded_kv_heads, rank, world_size, info.device)


class LayerKVScales(NamedTuple):
    """One layer's static scales expanded per q head ([qo_heads, 1]), as attention folds them:
    K's into q, V's into the output."""
    k_qo: torch.Tensor
    v_qo: torch.Tensor


def layer_kv_scales(scales: KVScales | None, num_qo_heads: int) -> list[LayerKVScales] | None:
    if scales is None:
        return None
    group = num_qo_heads // scales.k.shape[1]
    return [
        LayerKVScales(k.repeat_interleave(group).unsqueeze(-1), v.repeat_interleave(group).unsqueeze(-1))
        for k, v in zip(scales.k, scales.v, strict=True)
    ]


def quantize_kv(x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """``x`` [tokens, heads, dim] stored in ``dtype``; ``scale`` [heads] or None for unit."""
    bound = torch.finfo(dtype).max
    xf = x.float()
    if scale is not None:
        xf = xf / scale[:, None]
    return xf.clamp(-bound, bound).to(dtype)


def dequantize_kv(x: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """Inverse of ``quantize_kv``, for ``x`` [..., heads, dim]."""
    xf = x.float()
    if scale is not None:
        xf = xf * scale[:, None]
    return xf.to(dtype)


def observe_dir() -> str | None:
    """``MSTAR_KV_OBSERVE``: a directory; every paged KV cache records the amax of the K/V it stores there."""
    return os.environ.get("MSTAR_KV_OBSERVE") or None


class KVObserver:
    """Per layer and local head amax of written K and V, saved for ``kv.calibrate``.

    Reset after capture sizing, whose dummy inputs are not activations. A replay's
    padding rows are still observed; they repeat real tokens, so they move amax little.
    """

    # at most one save per this many seconds
    SAVE_INTERVAL = 2.0

    def __init__(
        self, out_dir: str, resource_key: str, num_layers: int, num_heads: int,
        total_heads: int, rank: int, world_size: int, device: torch.device,
    ):
        self.path = os.path.join(out_dir, f"{resource_key}.rank{rank}.pt")
        self.amax = torch.zeros(2, num_layers, num_heads, dtype=torch.float32, device=device)
        self._meta = dict(resource_key=resource_key, total_heads=total_heads, rank=rank, world_size=world_size)
        self._lock = threading.Lock()
        self._last_save = 0.0
        os.makedirs(out_dir, exist_ok=True)
        logger.warning("KV %s: observing K/V amax into %s", resource_key, self.path)

    def observe(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape[0] == 0:
            return
        for i, x in enumerate((k, v)):
            amax = x.abs().amax(dim=(0, 2)).float()
            # fmax ignores NaN; an inf would end calibration, so drop it too
            amax = torch.where(torch.isfinite(amax), amax, 0.0)
            torch.fmax(self.amax[i, layer_idx], amax, out=self.amax[i, layer_idx])

    def reset(self) -> None:
        self.amax.zero_()

    def save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_save < self.SAVE_INTERVAL:
            return
        with self._lock:
            self._last_save = now
            tmp = f"{self.path}.tmp"
            torch.save({**self._meta, "amax": self.amax.cpu()}, tmp)
            os.replace(tmp, self.path)
