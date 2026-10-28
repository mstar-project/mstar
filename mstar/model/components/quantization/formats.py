"""How a quantized weight is stored: one ``WeightFormat`` per checkpoint layout.

A format names the tensors one quantized ``(rows, cols)`` weight is stored as (its parts), the
dtype each part is held in, how many logical elements one stored element covers along each dim
(which is what TP slicing needs), and how the parts dequantize. Triton-free, so a model's config
and the CPU tests can import it.

* :class:`Fp8Block`: e4m3 with one scale per block (DeepSeek-V3's format: GLM-5.x with 128 x 128
  fp32 scales; DeepSeek-V4.1's dense linears with 32 x 32 power-of-two E8M0 scales).
* :class:`Int4Group`: compressed-tensors ``pack-quantized`` symmetric int4, one scale per group of
  columns (Kimi K2.7's routed experts).
* :class:`Fp4E2M1`: e2m1 values two to a byte, low nibble first, with an E8M0 scale per 32 columns
  (MXFP4: Kimi K3's routed experts as compressed-tensors ``mxfp4-pack-quantized``, DeepSeek-V4.1's
  routed experts as int8 + ``float8_e8m0fnu``).
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

logger = logging.getLogger(__name__)

_E8M0 = getattr(torch, "float8_e8m0fnu", None)
# torch dtypes a checkpoint stores a 1-byte part as; the module holds them all as uint8 bytes
_BYTE_DTYPES = tuple(
    d for d in (torch.uint8, torch.int8, FP8_DTYPE, torch.float8_e5m2, _E8M0) if d is not None)
_QUANTIZED_DTYPES = _BYTE_DTYPES + (torch.int32,)


class Part(NamedTuple):
    """One stored tensor of a quantized weight."""

    checkpoint: str  # the checkpoint name's suffix after the weight's base name
    container: str  # the module parameter's suffix
    dtype: torch.dtype  # what the module holds it as
    unit: tuple[int, int]  # logical (rows, cols) one stored element covers
    accepts: tuple[torch.dtype, ...]  # the dtypes a checkpoint may store it as


def as_container(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``tensor`` as the module's container dtype: 1-byte types reinterpret as bytes, not convert."""
    if tensor.dtype == dtype:
        return tensor
    if tensor.element_size() == 1 and torch.empty((), dtype=dtype).element_size() == 1:
        return tensor.view(dtype)
    return tensor.to(dtype)


def is_quantized_dtype(dtype: torch.dtype) -> bool:
    """A dtype only a quantized part is stored as in these checkpoints (never a bf16/fp32 weight)."""
    return dtype in _QUANTIZED_DTYPES


def e8m0_to_float(exponent: torch.Tensor) -> torch.Tensor:
    """E8M0 scales (power-of-two exponents biased by 127, held as bytes) as fp32: 2^(e - 127)."""
    e = as_container(exponent, torch.uint8).to(torch.int32)
    # the fp32 bits of 2^(e - 127), exactly: the exponent field, a subnormal for e = 0, NaN for 255
    bits = torch.where(e == 0, torch.full_like(e, 0x00400000), e << 23)
    return torch.where(e == 255, torch.full_like(e, 0x7FC00000), bits).view(torch.float32)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class WeightFormat:
    """A stored layout of one quantized ``(rows, cols)`` weight. Subclass per checkpoint format."""

    name: str = ""
    parts: tuple[Part, ...] = ()
    aux: tuple[str, ...] = ()  # checkpoint suffixes read for checks only, never loaded

    def shape(self, part: Part, rows: int, cols: int) -> tuple[int, int]:
        return _ceil_div(rows, part.unit[0]), _ceil_div(cols, part.unit[1])

    def part(self, checkpoint_suffix: str) -> Part | None:
        for p in self.parts:
            if p.checkpoint == checkpoint_suffix:
                return p
        return None

    def claims(self, suffix: str, tensor: torch.Tensor) -> bool:
        """Whether a checkpoint tensor ``<base>.<suffix>`` is one of this format's parts (or aux)."""
        if suffix in self.aux:
            return True
        p = self.part(suffix)
        return p is not None and tensor.dtype in p.accepts

    def check_shard(self, rows: int, cols: int, row_split: int | None = None,
                    col_split: int | None = None) -> None:
        """Refuse a shard that cuts a stored element: a dim split ``n`` ways (``row_split`` /
        ``col_split``; None leaves it whole, ragged tail allowed) must give each rank whole stored
        elements of every part."""
        for p in self.parts:
            for n, split, u, dim in ((rows, row_split, p.unit[0], "rows"),
                                     (cols, col_split, p.unit[1], "cols")):
                if split is not None and (n % split or (n // split) % u):
                    raise ValueError(
                        f"{self.name}: {n} {dim} split {split} ways is not a whole number of "
                        f"{p.container} elements ({u} {dim} each)")

    def dequantize(self, parts: dict[str, torch.Tensor], rows: int, cols: int,
                   out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """The dense ``(rows, cols)`` weight from ``parts`` keyed by container suffix."""
        raise NotImplementedError

    def fused_available(self, device: torch.device) -> bool:
        """Whether this format's fused MoE kernels run on ``device`` (none, by default)."""
        del device
        return False


class Fp8Block(WeightFormat):
    """e4m3 weights with one scale per ``block`` (rows, cols). ``scale`` names the checkpoint's
    scale tensor (DeepSeek-V3's HF ``weight_scale_inv``; DeepSeek-V4.1's ``scale``); ``e8m0`` scales
    are power-of-two bytes, held as such. Either way ``w = fp8 * scale``."""

    def __init__(self, block: tuple[int, int] = (128, 128), scale: str = "weight_scale_inv",
                 e8m0: bool = False) -> None:
        self.block = (int(block[0]), int(block[1]))
        self.scale_name = scale
        self.e8m0 = e8m0
        self.name = f"fp8 block {self.block[0]}x{self.block[1]}" + (" ue8m0" if e8m0 else "")
        scale_accepts = (torch.uint8,) + ((_E8M0,) if _E8M0 is not None else ()) if e8m0 else (
            torch.float32, torch.bfloat16)
        self.parts = (
            Part("weight", "fp8", torch.uint8, (1, 1), (FP8_DTYPE,)),
            Part(scale, "scale_inv", torch.uint8 if e8m0 else torch.float32, self.block,
                 scale_accepts),
        )

    def dequantize(self, parts, rows, cols, out_dtype=torch.bfloat16):
        scale = parts["scale_inv"]
        if self.e8m0:
            scale = e8m0_to_float(scale)
        weight = parts["fp8"]
        assert tuple(weight.shape) == (rows, cols), (tuple(weight.shape), (rows, cols))
        return dequantize_fp8_block_weight(weight, scale, block_size=self.block,
                                           out_dtype=out_dtype)

    def fused_available(self, device: torch.device) -> bool:
        return not self.e8m0 and fused_fp8_available(device, self.block)


# the weight scale block fused_experts_fp8 compiles for
_FUSED_FP8_BLOCK = (128, 128)


def fused_fp8_available(device: torch.device, block_size: tuple[int, int]) -> bool:
    """Whether fused_experts_fp8 can serve fp8 block-scale experts: CUDA, an importable kernel, and
    the scale block it compiles for."""
    if device.type != "cuda" or tuple(block_size) != _FUSED_FP8_BLOCK:
        return False
    try:
        from mstar.utils.fused_moe import fused_experts_fp8  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - 'auto' serves the reference loop, loudly
        logger.warning("fused fp8 MoE kernel unavailable (%r): the reference dispatch "
                       "serves eager, without CUDA graphs", exc)
        return False
    return True


class Int4Group(WeightFormat):
    """compressed-tensors ``pack-quantized`` int4, symmetric, one scale per ``group`` columns.

    Eight values per int32 along the columns, value ``8j + i`` in bits ``4i..4i+3`` of word ``j``,
    stored offset by 8 (unsigned); ``w = (q - 8) * scale``. ``weight_shape`` is read as a check.
    """

    aux = ("weight_shape",)

    def __init__(self, group: int = 32) -> None:
        self.group = int(group)
        self.name = f"int4 group {self.group}"
        self.parts = (
            Part("weight_packed", "packed", torch.int32, (1, 8), (torch.int32,)),
            Part("weight_scale", "scale", torch.bfloat16, (1, self.group),
                 (torch.bfloat16, torch.float16, torch.float32)),
        )

    def dequantize(self, parts, rows, cols, out_dtype=torch.bfloat16):
        q = unpack_int4(parts["packed"], cols)
        scale = parts["scale"].to(torch.float32).repeat_interleave(self.group, dim=1)[:, :cols]
        return (q.to(torch.float32) * scale).to(out_dtype)


def unpack_int4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """``(rows, ceil(cols / 8))`` int32 words to ``(rows, cols)`` int8 values in [-8, 7]."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    nibbles = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF  # (rows, words, 8)
    return (nibbles.reshape(packed.shape[0], -1)[:, :cols] - 8).to(torch.int8)


# e2m1 code -> value; bit 3 is the sign
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class Fp4E2M1(WeightFormat):
    """e2m1 values two per byte along the columns (element ``2j`` in the low nibble of byte ``j``)
    with one E8M0 scale per 32 columns: ``w = value * 2^(scale - 127)``. ``names`` are the
    checkpoint's (values, scales) suffixes: compressed-tensors' ``("weight_packed",
    "weight_scale")`` (uint8, uint8) or DeepSeek-V4.1's ``("weight", "scale")`` (int8, e8m0)."""

    group = 32

    def __init__(self, names: tuple[str, str] = ("weight_packed", "weight_scale")) -> None:
        self.name = "fp4 e2m1 / e8m0 group 32"
        e8m0 = (_E8M0,) if _E8M0 is not None else ()
        self.parts = (
            Part(names[0], "packed", torch.uint8, (1, 2), (torch.uint8, torch.int8)),
            Part(names[1], "scale", torch.uint8, (1, self.group), (torch.uint8,) + e8m0),
        )

    def dequantize(self, parts, rows, cols, out_dtype=torch.bfloat16):
        values = unpack_e2m1(parts["packed"], cols)
        scale = e8m0_to_float(parts["scale"]).repeat_interleave(self.group, dim=1)[:, :cols]
        return (values * scale).to(out_dtype)


def unpack_e2m1(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """``(rows, ceil(cols / 2))`` bytes to ``(rows, cols)`` fp32 e2m1 values, low nibble first."""
    b = as_container(packed, torch.uint8).to(torch.int64)
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(b.shape[0], -1)[:, :cols]
    return table[codes]


__all__ = [
    "E2M1_VALUES",
    "Fp4E2M1",
    "Fp8Block",
    "Int4Group",
    "Part",
    "WeightFormat",
    "as_container",
    "e8m0_to_float",
    "fused_fp8_available",
    "is_quantized_dtype",
    "unpack_e2m1",
    "unpack_int4",
]
