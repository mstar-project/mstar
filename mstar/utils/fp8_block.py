"""FP8 block-scale checkpoints: the ``quantization_config`` they declare, and their weights dequantized;
the MXFP8 and NVFP4 formats of cuBLASLt's block-scaled GEMMs, and their host-side quantizer.

A block-scale checkpoint (DeepSeek-V3's format, GLM-5.x's) stores each linear weight as e4m3 with one fp32
``weight_scale_inv`` per ``weight_block_size`` block. Triton-free, so a model's config can import it on a
CPU-only install; the activation quantizer that feeds the block-scale GEMMs is ``quant_fp8``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

_WEIGHT = ".weight"
_SCALE_INV = ".weight_scale_inv"

FP8_DTYPE = torch.float8_e4m3fn
E8M0_DTYPE = torch.float8_e8m0fnu
FP4_DTYPE = torch.float4_e2m1fn_x2


@dataclass(frozen=True)
class BlockFormat:
    """One matrix format of a block-scaled GEMM: ``block`` values along K share one ``scale_dtype`` scale, and
    ``packing`` values share one stored ``data_dtype`` element (NVFP4's K is half its value count)."""

    name: str
    data_dtype: torch.dtype
    scale_dtype: torch.dtype
    block: int
    packing: int
    data_max: float


MXFP8 = BlockFormat("mxfp8", FP8_DTYPE, E8M0_DTYPE, block=32, packing=1, data_max=448.0)
NVFP4 = BlockFormat("nvfp4", FP4_DTYPE, FP8_DTYPE, block=16, packing=2, data_max=6.0)
BLOCK_FORMATS = {fmt.name: fmt for fmt in (MXFP8, NVFP4)}

# The swizzled scale layout tiles blocks by this many along K and rows by 128.
BLOCKS_PER_TILE = 4
_ROWS_PER_TILE = 128
_E8M0_MAX_BIASED = 253  # 254 would make the data's multiply-back 2^-127, which fp32 can't hold as a normal
_E4M3_MAX_FINITE = 0x7E


class BlockQuant(NamedTuple):
    """One quantized matrix ``[rows, K // packing]``. ``scale`` is flat in the GEMM's swizzled layout;
    ``tensor_scale`` is NVFP4's second level, a ``[1]`` fp32 (None under MXFP8)."""

    data: torch.Tensor
    scale: torch.Tensor
    tensor_scale: torch.Tensor | None = None


def check_block_gemm(fmt: BlockFormat, k: int, n: int, dtype: torch.dtype) -> None:
    """Refuse what the block-scaled GEMM and the quantizer can't run: the quantizer writes whole scale tiles, so K
    is a multiple of ``BLOCKS_PER_TILE`` blocks; cuBLASLt wants N and the stored K to be multiples of 16."""
    if dtype != torch.bfloat16:
        raise ValueError(f"{fmt.name} replaces a bf16 GEMM; this linear's weight is {dtype}")
    if k % (fmt.block * BLOCKS_PER_TILE) or (k // fmt.packing) % 16:
        raise ValueError(f"{fmt.name} needs K a multiple of {fmt.block * BLOCKS_PER_TILE}; this linear's K is {k}")
    if n % 16:
        raise ValueError(f"{fmt.name} needs N a multiple of 16; this linear's N is {n}")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def swizzled_scale_elements(rows: int, k_blocks: int) -> int:
    """Length of the flat scales of ``[rows, k_blocks]``: both padded to whole tiles."""
    return (_ceil_div(rows, _ROWS_PER_TILE) * _ROWS_PER_TILE) * (_ceil_div(k_blocks, BLOCKS_PER_TILE) * BLOCKS_PER_TILE)


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """``[rows, k_blocks]`` -> the flat 32x4x4 layout of the cuBLAS block scaling factors: 128x4 tiles of 512
    elements, row ``r``, block ``j`` of a tile at ``(r % 32) * 16 + (r // 32) * 4 + j``."""
    rows, blocks = scales.shape
    row_tiles, col_tiles = _ceil_div(rows, _ROWS_PER_TILE), _ceil_div(blocks, BLOCKS_PER_TILE)
    padded = torch.zeros(
        row_tiles * _ROWS_PER_TILE, col_tiles * BLOCKS_PER_TILE, dtype=scales.dtype, device=scales.device,
    )
    padded[:rows, :blocks] = scales
    tiles = padded.view(row_tiles, _ROWS_PER_TILE, col_tiles, BLOCKS_PER_TILE).permute(0, 2, 1, 3)
    return tiles.reshape(-1, BLOCKS_PER_TILE, 32, BLOCKS_PER_TILE).transpose(1, 2).reshape(-1, 32, 16).flatten()


def unswizzle_block_scales(flat: torch.Tensor, rows: int, k_blocks: int) -> torch.Tensor:
    """The inverse of :func:`swizzle_block_scales`, padding dropped."""
    row_tiles, col_tiles = _ceil_div(rows, _ROWS_PER_TILE), _ceil_div(k_blocks, BLOCKS_PER_TILE)
    tiles = flat.view(-1, 32, 16).reshape(-1, 32, BLOCKS_PER_TILE, BLOCKS_PER_TILE).transpose(1, 2)
    padded = tiles.reshape(row_tiles, col_tiles, _ROWS_PER_TILE, BLOCKS_PER_TILE).permute(0, 2, 1, 3)
    return padded.reshape(row_tiles * _ROWS_PER_TILE, col_tiles * BLOCKS_PER_TILE)[:rows, :k_blocks]


def _fp4_codes(u: torch.Tensor) -> torch.Tensor:
    """fp32 to e2m1 codes, round to nearest even: ties go to 0, 1, 2 and 4."""
    a = u.abs()
    code = (
        (a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
        + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int() + (u < 0).int() * 8
    )
    return code.to(torch.uint8)


def _fp4_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    values = table[(codes & 7).long()]
    return torch.where(codes & 8 != 0, -values, values)


def quantize_block(
    x: torch.Tensor, fmt: BlockFormat, tensor_scale: torch.Tensor | float | None = None,
) -> BlockQuant:
    """Quantize a row-major ``[rows, K]`` matrix, one scale per ``fmt.block`` values along K per row: an MXFP8 scale
    is the power of two at or above ``amax / 448``, an NVFP4 scale the e4m3 at or above ``amax / 6`` in units of
    ``tensor_scale`` (default: the tensor's own ``amax / (6 * 448)``, which gives the largest block the top of
    e4m3's range). Rounding the scale up means no value clips. ``quant_fp8`` does this per call on the device."""
    rows, k = x.shape
    blocks = x.float().view(rows, k // fmt.block, fmt.block)
    amax = blocks.abs().amax(dim=-1)
    if fmt is MXFP8:
        mantissa, exponent = torch.frexp(amax / fmt.data_max)
        biased = (exponent - (mantissa == 0.5).to(exponent.dtype) + 127).clamp(1, _E8M0_MAX_BIASED)
        biased = torch.where(amax == 0, torch.ones_like(biased), biased)
        scale = torch.ldexp(torch.ones_like(amax), biased - 127)
        data = (blocks / scale[..., None]).view(rows, k).to(fmt.data_dtype)
        return BlockQuant(data, swizzle_block_scales(biased.to(torch.uint8)).view(fmt.scale_dtype))
    if tensor_scale is None:
        tensor_scale = (amax.max() / (fmt.data_max * 448.0)).clamp_min(torch.finfo(torch.float32).tiny)
    tensor_scale = torch.as_tensor(tensor_scale, dtype=torch.float32, device=x.device).reshape(1)
    want = amax / tensor_scale / fmt.data_max
    code = want.to(FP8_DTYPE).view(torch.uint8).int()
    code = (code + (code.to(torch.uint8).view(FP8_DTYPE).float() < want).int()).clamp(1, _E4M3_MAX_FINITE)
    code = code.to(torch.uint8)
    scale = code.view(FP8_DTYPE).float() * tensor_scale
    nibbles = _fp4_codes((blocks / scale[..., None]).view(rows, k)).view(rows, k // 2, 2)
    data = (nibbles[..., 0] | (nibbles[..., 1] << 4)).view(fmt.data_dtype)
    return BlockQuant(data, swizzle_block_scales(code).view(fmt.scale_dtype), tensor_scale)


def dequantize_block(q: BlockQuant, fmt: BlockFormat) -> torch.Tensor:
    """The fp32 ``[rows, K]`` a :class:`BlockQuant` stands for."""
    rows, stored = q.data.shape
    k = stored * fmt.packing
    scales = unswizzle_block_scales(q.scale, rows, k // fmt.block)
    if fmt is MXFP8:
        values = q.data.float()
        scale = scales.float()
    else:
        raw = q.data.view(torch.uint8)
        values = _fp4_values(torch.stack([raw & 15, raw >> 4], dim=-1)).view(rows, k)
        scale = scales.float() * q.tensor_scale
    return values * scale.repeat_interleave(fmt.block, dim=1)


@dataclass(frozen=True)
class Fp8BlockQuantConfig:
    """The fields of an HF fp8 ``quantization_config`` a block-scale checkpoint carries."""

    weight_block_size: tuple[int, int] = (128, 128)
    fmt: str = "e4m3"
    activation_scheme: str = "dynamic"
    ignore: tuple[str, ...] = field(default_factory=tuple)  # modules_to_not_convert

    @classmethod
    def from_hf_config_dict(cls, quant: dict | None) -> "Fp8BlockQuantConfig | None":
        if not quant or quant.get("quant_method") != "fp8":
            return None
        block = quant.get("weight_block_size") or [128, 128]
        return cls(
            weight_block_size=(int(block[0]), int(block[1])),
            fmt=str(quant.get("fmt", "e4m3")),
            activation_scheme=str(quant.get("activation_scheme", "dynamic")),
            ignore=tuple(quant.get("modules_to_not_convert", []) or []),
        )


def dequantize_fp8_block_weight(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one 2-D fp8 weight: ``w[i, j] * scale_inv[i//bo, j//bi]``."""
    if weight_fp8.dtype == torch.uint8:
        weight_fp8 = weight_fp8.view(FP8_DTYPE)
    out_f, in_f = weight_fp8.shape
    bo, bi = block_size
    expected = (-(-out_f // bo), -(-in_f // bi))  # ceil-div
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape)} does not match weight "
            f"{(out_f, in_f)} with block_size {block_size} (expected {expected})"
        )
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bo, dim=0)[:out_f]
    scale = scale.repeat_interleave(bi, dim=1)[:, :in_f]
    return (weight_fp8.to(torch.float32) * scale).to(out_dtype)


def dequant_fp8_block_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: Fp8BlockQuantConfig,
    out_dtype: torch.dtype = torch.bfloat16,
    keep_fp8: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pair fp8 ``.weight`` keys with ``.weight_scale_inv`` and dequantize."""
    buffers: dict[str, dict[str, torch.Tensor]] = {}

    def emit(base: str, slot: dict[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
        weight = dequantize_fp8_block_weight(
            slot[_WEIGHT], slot[_SCALE_INV],
            block_size=quant_config.weight_block_size, out_dtype=out_dtype,
        )
        del buffers[base]
        return base + _WEIGHT, weight

    for name, tensor in weights:
        if name.endswith(_SCALE_INV):
            base = name[: -len(_SCALE_INV)]
            if keep_fp8 is not None and keep_fp8(base):
                yield name, tensor
                continue
            slot = buffers.setdefault(base, {})
            slot[_SCALE_INV] = tensor
            if _WEIGHT in slot:
                yield emit(base, slot)
            continue

        if name.endswith(_WEIGHT) and tensor.dtype == FP8_DTYPE:
            base = name[: -len(_WEIGHT)]
            if keep_fp8 is not None and keep_fp8(base):
                yield name, tensor
                continue
            slot = buffers.setdefault(base, {})
            slot[_WEIGHT] = tensor
            if _SCALE_INV in slot:
                yield emit(base, slot)
            continue

        yield name, tensor  # bf16/fp32 passthrough (modules_to_not_convert)

    if buffers:
        missing = {b: sorted(slot) for b, slot in buffers.items()}
        raise ValueError(
            f"fp8 stream ended with unpaired quantized tensors (weight without "
            f"weight_scale_inv or vice versa): {missing}"
        )
