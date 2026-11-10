"""What a checkpoint stores, read from its safetensors headers and ``config.json``.

Each linear the checkpoint names (a *base*: ``model.layers.0.mlp.gate_proj``) is classified by the tensors it
carries and their dtypes, not by the config's ignore list: ``weight`` in e4m3 beside a 2-D ``weight_scale_inv``
is block fp8 wherever it appears, and ``weight`` in bf16 is bf16 even inside a quantized checkpoint
(``modules_to_not_convert``, ``ignore``, ``exclude_modules`` all come out the same). The formats vLLM and
SGLang read are recognised, so a checkpoint whose loader hasn't landed is refused by name at boot rather
than loaded as bytes:

=================================  ==========================================  ==============
tensors of a base                  checkpoints                                  stored format
=================================  ==========================================  ==============
weight e4m3, weight_scale_inv 2-D  DeepSeek/Qwen/GLM/MiniMax ``fp8`` (block)    fp8_block
weight e4m3, weight_scale [N(,1)]  compressed-tensors float-quantized channel   fp8_channel
weight e4m3, weight_scale [1]      compressed-tensors / modelopt FP8 tensor     fp8_tensor
weight e4m3, weight_scale_inv u8   ``mxfp8`` ([1, 32] e8m0, MiniMax-M3-MXFP8)   mxfp8
weight u8, weight_scale e4m3,      modelopt NVFP4                               nvfp4
weight_scale_2
weight_packed u8, weight_scale e4m3 compressed-tensors nvfp4-pack-quantized     nvfp4
weight_packed u8, weight_scale u8  compressed-tensors mxfp4-pack-quantized      mxfp4
weight_packed i32, weight_scale    compressed-tensors pack-quantized (int4)     w4a16
qweight, qzeros, scales            AutoAWQ / GPTQ                               w4a16
=================================  ==========================================  ==============

An int4 base also carries its ``params`` (group size, symmetry, the scales' dtype, GPTQ's zero-point offset), which
pick the format's variant the loader repacks it with; a GPTQ checkpoint with ``desc_act`` gets ``act_order`` on the
bases whose ``g_idx`` is not the trivial ``k // group`` (``scan_checkpoint`` reads those small tensors), and the
resolver refuses them.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from mstar.quantization.formats import BF16

# suffixes a base's tensors carry, longest first so `weight_scale_inv` isn't read as `weight`
SUFFIXES = (
    "weight_scale_inv", "weight_scale_2", "weight_global_scale", "input_global_scale", "weight_zero_point",
    "weight_packed", "weight_shape", "weight_scale", "input_scale", "weight", "qweight", "qzeros", "scales",
    "g_idx", "bias",
)
_FLOAT = {"BF16", "F16", "F32"}
UNKNOWN = "unknown"


class TensorInfo(NamedTuple):
    dtype: str  # safetensors dtype string: BF16, F8_E4M3, U8, I32, F32, ...
    shape: tuple[int, ...]


class StoredBase(NamedTuple):
    """How one checkpoint linear is stored: a format name (``formats.FORMATS`` or ``bf16``, ``unknown``) and the
    checkpoint layout it came in (``awq``, ``gptq``, ``compressed-tensors``, ``modelopt``, ``fp8``). ``params``
    are the facts a format's loader needs beyond those, as sorted ``(key, value)`` pairs (int4: ``group``,
    ``symmetric``, ``scale_dtype``, GPTQ's ``zero_offset``, ``act_order``)."""

    format: str
    layout: str = ""
    block: tuple[int, int] | None = None
    params: tuple[tuple[str, Any], ...] = ()

    def param(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    def with_params(self, **params: Any) -> StoredBase:
        return self._replace(params=tuple(sorted({**dict(self.params), **params}.items())))


def split_name(name: str) -> tuple[str, str]:
    """``(base, suffix)`` of a tensor name, suffix one of ``SUFFIXES`` (``""`` when none is)."""
    for suffix in SUFFIXES:
        if name.endswith("." + suffix):
            return name[: -len(suffix) - 1], suffix
    return name, ""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _int4_bits(config: Mapping[str, Any]) -> int:
    """The weight bits an int4-looking checkpoint declares (AutoAWQ ``bits`` / ``w_bit``, GPTQ ``bits``,
    compressed-tensors ``config_groups.*.weights.num_bits``, the largest of its groups); 4 when it says nothing."""
    bits = config.get("bits", config.get("w_bit"))
    if bits is None:
        groups = config.get("config_groups") or {}
        widths = [int(((g or {}).get("weights") or {}).get("num_bits", 4)) for g in groups.values()]
        bits = max(widths, default=4)
    return int(bits)


def _w4a16(tensors: Mapping[str, TensorInfo], method: str, config: Mapping[str, Any]) -> StoredBase:
    """An int4 base: AutoAWQ (``qweight [K, N/8]``, ``qzeros [K/g, N/8]``, ``scales [K/g, N]``), GPTQ
    (``qweight [K/8, N]``, ``qzeros``, ``scales``, ``g_idx [K]``) or compressed-tensors pack-quantized
    (``weight_packed [N, K/8]``, ``weight_scale [N, K/g]``, ``weight_zero_point [N/8, K/g]`` if asymmetric)."""
    if _int4_bits(config) != 4:
        return StoredBase(UNKNOWN, method)
    if "qweight" in tensors:
        layout = method if method in ("awq", "gptq") else ("gptq" if "g_idx" in tensors else "awq")
        qweight, scales = tensors["qweight"], tensors.get("scales")
        k = qweight.shape[0] * (8 if layout == "gptq" else 1)
        if layout == "gptq":
            symmetric = bool(config.get("sym", True))
            # GPTQ's v1 checkpoints store each zero point minus one; gptq_v2 (GPTQModel) stores it as is
            extra = {"zero_offset": 0 if config.get("checkpoint_format") == "gptq_v2" else 1}
        else:
            symmetric, extra = "qzeros" not in tensors, {}
        groups = scales.shape[0] if scales is not None and len(scales.shape) == 2 else 0
        scale_dtype = scales.dtype if scales is not None else ""
    else:
        packed, scale = tensors["weight_packed"], tensors.get("weight_scale")
        layout, k = "compressed-tensors", packed.shape[-1] * 8
        symmetric, extra = "weight_zero_point" not in tensors, {}
        groups = scale.shape[-1] if scale is not None and len(scale.shape) == 2 else 0
        scale_dtype = scale.dtype if scale is not None else ""
    if not groups or k % groups:
        return StoredBase(UNKNOWN, layout)
    group = k // groups
    return StoredBase("w4a16", layout, (1, group)).with_params(
        group=group, symmetric=symmetric, scale_dtype=scale_dtype, **extra)


def classify(
    tensors: Mapping[str, TensorInfo], method: str = "", block: tuple[int, int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> StoredBase:
    """The stored format of one base from its ``suffix -> TensorInfo``; ``method`` is the config's
    ``quant_method`` (it tells AWQ from GPTQ), ``block`` its ``weight_block_size``, ``config`` the whole
    ``quantization_config`` (an int4 checkpoint's bits, GPTQ's symmetry and zero-point convention)."""
    w = tensors.get("weight")
    config = config or {}
    if "qweight" in tensors:
        return _w4a16(tensors, method, config)
    if "weight_packed" in tensors:
        packed, scale = tensors["weight_packed"], tensors.get("weight_scale")
        if packed.dtype == "I32":
            return _w4a16(tensors, method or "compressed-tensors", config)
        if scale is not None and scale.dtype == "U8":
            return StoredBase("mxfp4", "compressed-tensors", (1, 32))
        if scale is not None and scale.dtype.startswith("F8"):
            return StoredBase("nvfp4", "compressed-tensors", (1, 16))
        return StoredBase(UNKNOWN, "compressed-tensors")
    if w is None:
        return StoredBase(UNKNOWN)
    if w.dtype in _FLOAT:
        return StoredBase(BF16)
    if w.dtype == "U8" and "weight_scale_2" in tensors:
        return StoredBase("nvfp4", "modelopt", (1, 16))
    if w.dtype == "F8_E4M3":
        inv = tensors.get("weight_scale_inv")
        if inv is not None and inv.dtype in ("U8", "F8_E8M0"):
            return StoredBase("mxfp8", "mxfp8", (1, 32))
        if inv is not None and len(inv.shape) == 2 and len(w.shape) == 2:
            n, k = w.shape
            # the config's block, else the usual 128 x 128 if the shapes fit it, else what they imply
            for candidate in ((block,) if block else ((128, 128), (_ceil_div(n, inv.shape[0]),
                                                                   _ceil_div(k, inv.shape[1])))):
                if inv.shape == (_ceil_div(n, candidate[0]), _ceil_div(k, candidate[1])):
                    return StoredBase("fp8_block", "fp8", tuple(candidate))
            return StoredBase(UNKNOWN, "fp8")
        scale = tensors.get("weight_scale")
        if scale is not None:
            numel = 1
            for d in scale.shape:
                numel *= d
            if numel == 1:
                return StoredBase("fp8_tensor", method or "compressed-tensors")
            if scale.shape[0] == w.shape[0] and numel == w.shape[0]:
                return StoredBase("fp8_channel", method or "compressed-tensors", (1, 0))
        return StoredBase(UNKNOWN, "fp8")
    return StoredBase(UNKNOWN)


def read_headers(checkpoint_dir: str | Path) -> dict[str, TensorInfo]:
    """Every tensor's dtype and shape from the safetensors headers under ``checkpoint_dir`` (no data read)."""
    root = Path(checkpoint_dir)
    files = sorted(root.glob("*.safetensors"))
    out: dict[str, TensorInfo] = {}
    for path in files:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            out[name] = TensorInfo(entry["dtype"], tuple(entry["shape"]))
    return out


def read_quant_config(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """``config.json``'s ``quantization_config`` (top level or under ``text_config``), else modelopt's
    ``hf_quant_config.json``; None for a checkpoint that declares neither."""
    root = Path(checkpoint_dir)
    config_path = root / "config.json"
    if config_path.is_file():
        raw = json.loads(config_path.read_text())
        quant = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config")
        if quant:
            return dict(quant)
    modelopt = root / "hf_quant_config.json"
    if modelopt.is_file():
        raw = json.loads(modelopt.read_text())
        return {"quant_method": "modelopt", **(raw.get("quantization") or raw)}
    return None


@dataclass
class CheckpointQuant:
    """A checkpoint's stored format per base (full checkpoint names)."""

    method: str = ""
    block: tuple[int, int] | None = None
    bases: dict[str, StoredBase] = field(default_factory=dict)
    # the config's ``quantization_config`` (or modelopt's ``hf_quant_config.json``), as read
    config: dict[str, Any] = field(default_factory=dict)
    # base -> its tensors' suffixes, for the loader's pairing
    suffixes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def quantized(self) -> bool:
        return any(stored.format != BF16 for stored in self.bases.values())

    def formats(self) -> set[str]:
        return {stored.format for stored in self.bases.values()}

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[str, TensorInfo], quant_config: Mapping[str, Any] | None = None,
    ) -> CheckpointQuant:
        method = str((quant_config or {}).get("quant_method", ""))
        raw_block = (quant_config or {}).get("weight_block_size")
        block = (int(raw_block[0]), int(raw_block[1])) if raw_block and method == "fp8" else None
        grouped: dict[str, dict[str, TensorInfo]] = defaultdict(dict)
        for name, tensor in tensors.items():
            base, suffix = split_name(name)
            if suffix:
                grouped[base][suffix] = tensor
        bases, suffixes = {}, {}
        for base, parts in grouped.items():
            # norms, biases-only, embeddings' `weight` with no linear around it still classify as bf16
            bases[base] = classify(parts, method, block, quant_config)
            suffixes[base] = tuple(sorted(parts))
        return cls(method=method, block=block, bases=bases, suffixes=suffixes, config=dict(quant_config or {}))

    def mark_act_order(self, g_idx: Mapping[str, Any]) -> None:
        """Flag the GPTQ bases whose ``g_idx`` (``base -> tensor``) isn't ``k // group``: act-order
        (``desc_act``) checkpoints permute K, which the canonical layout and its kernels don't."""
        import torch

        for base, idx in g_idx.items():
            stored = self.bases.get(base)
            if stored is None or stored.format != "w4a16":
                continue
            group = stored.param("group")
            trivial = torch.arange(idx.numel(), dtype=torch.int64) // group
            if not torch.equal(idx.reshape(-1).to(torch.int64), trivial):
                self.bases[base] = stored.with_params(act_order=True)


def _read_g_idx(checkpoint_dir: str | Path, names: set[str]) -> dict[str, Any]:
    """The named ``g_idx`` tensors (K int32 each, a few MB for a whole model), by base."""
    from safetensors import safe_open

    out = {}
    for path in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for name in names.intersection(f.keys()):
                out[split_name(name)[0]] = f.get_tensor(name)
    return out


def scan_checkpoint(checkpoint_dir: str | Path) -> CheckpointQuant:
    headers = read_headers(checkpoint_dir)
    ckpt = CheckpointQuant.from_tensors(headers, read_quant_config(checkpoint_dir))
    if ckpt.method == "gptq" and ckpt.config.get("desc_act"):
        ckpt.mark_act_order(_read_g_idx(checkpoint_dir, {n for n in headers if n.endswith(".g_idx")}))
    return ckpt
