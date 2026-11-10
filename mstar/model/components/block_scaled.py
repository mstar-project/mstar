"""cuBLASLt's block-scaled linear formats, MXFP8 and NVFP4 (``linear_dtype: mxfp8 | nvfp4``).

``block`` values along K share one scale; the scales sit in the GEMM's 32x4x4 swizzled layout
(``utils.fp8_block``). Weights are quantized once at load by the host-side quantizer (``quantize``),
activations per call by a Triton kernel (``utils.quant_fp8``), and the product is ``torch._scaled_mm``.
Each format is one op, ``mstar::mxfp8_linear`` / ``mstar::nvfp4_linear``, with a fake implementation,
so a compiled forward holds one node per linear. Off a GPU with the GEMM (``Platform.has_block_scaled_gemm``)
the op runs its reference, the product of what the operands stand for in fp32; a deployment never does
(``quant_linear.select_linears`` keeps such a model bf16), tests do.

NVFP4's activation range. A block's e4m3 scale tops out at code 0x7E (448) in units of the linear's
input tensor scale ``s``, so with e2m1's 6 a 16-block clips above ``2688 * s``, and below ``0.094 * s``
(``6 * 2^-6``) its scale goes subnormal and loses precision. With ``s = 1``, over the 100 protocol
prompts' denoise steps, klein 4B's text feed-forward outputs reach 4080 (9B's 5056), Z-Image's
feed-forward outputs up to 340k, and everything else stays under 1800; the low end matters less (a
smaller ``s`` moved the NVFP4 GEMM's error by under 5% on any linear). So a model declares the inputs
measured above 2688 (``QuantizableLinears.input_amax``) and those linears take the smallest power of two
that holds it (``Nvfp4Format.input_scale``); the rest keep ``s = 1``. A power of two quantizes every block
inside both ranges the same way.

A modelopt NVFP4 checkpoint (``nvidia/*-NVFP4``) loads as it is stored: ``weight`` uint8 ``[N, K/2]``, ``weight_scale``
e4m3 ``[N, K/16]`` in plain row-major order (swizzled by ``after_load``, once this rank's rows and columns are cut,
since a swizzled tile can't be sharded), ``weight_scale_2`` (the fp32 tensor scale) and ``input_scale`` (the
activation's calibrated tensor scale, the role a declared ``input_amax`` plays at load). The two meet in the same
buffers, so a stored layer and one quantized at load run the same op.
"""

from __future__ import annotations

import math

import torch

from mstar.model.components.quant_linear import LinearFormat, Storage, register_format
from mstar.platforms import current_platform
from mstar.utils.fp8_block import (
    BLOCKS_PER_TILE,
    E8M0_DTYPE,
    FP4_DTYPE,
    FP8_DTYPE,
    BlockQuant,
    swizzle_block_scales,
    unswizzle_block_scales,
)

_E8M0_MAX_BIASED = 253  # 254 would make the data's multiply-back 2^-127, which fp32 can't hold as a normal
_E4M3_MAX_FINITE = 0x7E


class BlockScaledFormat(LinearFormat):
    """A block-scaled format: ``block`` values along K share one ``scale_dtype`` scale, and ``packing`` values share
    one stored ``data_dtype`` element (NVFP4's stored K is half its value count)."""

    data_dtype: torch.dtype
    scale_dtype: torch.dtype
    block: int
    packing: int
    data_max: float

    def unavailable_reason(self, device: torch.device) -> str | None:
        if current_platform.has_block_scaled_gemm(device):
            return None
        return (f"{current_platform.name} {current_platform.arch() or ''} has no block-scaled GEMM "
                "(cuBLASLt, compute capability 10.0 and up)")

    def check(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        """The quantizer writes whole scale tiles, so K is a multiple of ``BLOCKS_PER_TILE`` blocks; cuBLASLt wants
        N and the stored K to be multiples of 16."""
        if dtype != torch.bfloat16:
            raise ValueError(f"{self.name} replaces a bf16 GEMM; this linear's weight is {dtype}")
        k, tile = in_features, self.block * BLOCKS_PER_TILE
        if k % tile or (k // self.packing) % 16:
            raise ValueError(f"{self.name} needs K a multiple of {tile}; this linear's K is {k}")
        if out_features % 16:
            raise ValueError(f"{self.name} needs N a multiple of 16; this linear's N is {out_features}")

    def quantize(self, x: torch.Tensor, tensor_scale: torch.Tensor | float | None = None) -> BlockQuant:
        """Quantize a row-major ``[rows, K]`` matrix in plain torch, one scale per ``block`` values along K per row,
        rounded up so no value inside the scale's range clips."""
        raise NotImplementedError

    def dequantize(self, q: BlockQuant) -> torch.Tensor:
        """The fp32 ``[rows, K]`` a ``BlockQuant`` stands for."""
        raise NotImplementedError

    def _blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, k = x.shape
        blocks = x.float().view(rows, k // self.block, self.block)
        return blocks, blocks.abs().amax(dim=-1)

    def _scales(self, q: BlockQuant) -> torch.Tensor:
        rows, stored = q.data.shape
        return unswizzle_block_scales(q.scale, rows, stored * self.packing // self.block).float()

    def reference_linear(self, x: BlockQuant, weight: BlockQuant) -> torch.Tensor:
        return (self.dequantize(x) @ self.dequantize(weight).t()).to(torch.bfloat16)


class Mxfp8Format(BlockScaledFormat):
    """e4m3 values, one power-of-two (e8m0) scale per 32."""

    name = "mxfp8"
    buffers = ("weight", "weight_scale")
    data_dtype, scale_dtype, block, packing, data_max = FP8_DTYPE, E8M0_DTYPE, 32, 1, 448.0

    def quantize(self, x, tensor_scale=None) -> BlockQuant:
        """A block's scale is the power of two at or above ``amax / 448``; ``tensor_scale`` doesn't apply."""
        del tensor_scale
        rows, k = x.shape
        blocks, amax = self._blocks(x)
        mantissa, exponent = torch.frexp(amax / self.data_max)
        biased = (exponent - (mantissa == 0.5).to(exponent.dtype) + 127).clamp(1, _E8M0_MAX_BIASED)
        biased = torch.where(amax == 0, torch.ones_like(biased), biased)
        scale = torch.ldexp(torch.ones_like(amax), biased - 127)
        data = (blocks / scale[..., None]).view(rows, k).to(self.data_dtype)
        return BlockQuant(data, swizzle_block_scales(biased.to(torch.uint8)).view(self.scale_dtype))

    def dequantize(self, q: BlockQuant) -> torch.Tensor:
        return q.data.float() * self._scales(q).repeat_interleave(self.block, dim=1)

    def quantize_weight(self, weight, input_amax):
        q = self.quantize(weight)
        return {"weight": q.data, "weight_scale": q.scale}

    def linear(self, x, weight, weight_scale):
        return torch.ops.mstar.mxfp8_linear(x, weight, weight_scale)


_FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Nvfp4Format(BlockScaledFormat):
    """e2m1 values packed two to a byte, low nibble first; one e4m3 scale per 16 values, in units of an fp32 tensor
    scale (see the module docstring for an activation's)."""

    name = "nvfp4"
    buffers = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    data_dtype, scale_dtype, block, packing, data_max = FP4_DTYPE, FP8_DTYPE, 16, 2, 6.0
    # a 16-block's largest magnitude a unit tensor scale holds: e4m3's top code times e2m1's top
    UNIT_RANGE = 6.0 * 448.0

    def __init__(self, nearest: bool = False):
        # how an activation block's e4m3 scale rounds: up (nothing clips; the input scales this module places) or
        # to nearest, as modelopt calibrates a checkpoint's input_scale and FlashInfer's FP4 MoE quantizes
        self.nearest = nearest

    def for_stored(self, stored) -> Nvfp4Format:
        """A checkpoint's NVFP4 linears quantize their activations as its calibration assumed: to nearest."""
        if stored is None or stored.format != self.name:
            return self
        return NVFP4_STORED

    @classmethod
    def input_scale(cls, input_amax: float | None) -> float:
        """An input's tensor scale: the smallest power of two whose range holds ``input_amax``, never below one.
        A power of two quantizes every block that fit under one the same way."""
        if not input_amax or input_amax <= cls.UNIT_RANGE:
            return 1.0
        return 2.0 ** math.ceil(math.log2(input_amax / cls.UNIT_RANGE))

    @staticmethod
    def _codes(u: torch.Tensor) -> torch.Tensor:
        """fp32 to e2m1 codes, round to nearest even: ties go to 0, 1, 2 and 4."""
        a = u.abs()
        code = (
            (a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int() + (u < 0).int() * 8
        )
        return code.to(torch.uint8)

    def quantize(self, x, tensor_scale=None, nearest: bool = False) -> BlockQuant:
        """A block's scale is the e4m3 at or above ``amax / 6`` (``nearest``: nearest to it) in units of
        ``tensor_scale`` (default: the tensor's own ``amax / (6 * 448)``, which gives the largest block the top of
        e4m3's range)."""
        rows, k = x.shape
        blocks, amax = self._blocks(x)
        if tensor_scale is None:
            tensor_scale = (amax.max() / (self.data_max * 448.0)).clamp_min(torch.finfo(torch.float32).tiny)
        tensor_scale = torch.as_tensor(tensor_scale, dtype=torch.float32, device=x.device).reshape(1)
        want = amax / tensor_scale / self.data_max
        code = want.to(FP8_DTYPE).view(torch.uint8).int()
        if not nearest:
            code = code + (code.to(torch.uint8).view(FP8_DTYPE).float() < want).int()
        code = code.clamp(1, _E4M3_MAX_FINITE)
        code = code.to(torch.uint8)
        scale = code.view(FP8_DTYPE).float() * tensor_scale
        nibbles = self._codes((blocks / scale[..., None]).view(rows, k)).view(rows, k // 2, 2)
        data = (nibbles[..., 0] | (nibbles[..., 1] << 4)).view(self.data_dtype)
        return BlockQuant(data, swizzle_block_scales(code).view(self.scale_dtype), tensor_scale)

    def _dequantize(self, q: BlockQuant) -> torch.Tensor:
        rows, stored = q.data.shape
        raw = q.data.view(torch.uint8)
        codes = torch.stack([raw & 15, raw >> 4], dim=-1)
        table = torch.tensor(_FP4_TABLE, device=codes.device)
        values = table[(codes & 7).long()]
        values = torch.where(codes & 8 != 0, -values, values).view(rows, stored * 2)
        scale = self._scales(q) * q.tensor_scale
        return values * scale.repeat_interleave(self.block, dim=1)

    def quantize_weight(self, weight, input_amax):
        q = self.quantize(weight)
        scale = torch.full((1,), self.input_scale(input_amax), dtype=torch.float32, device=weight.device)
        return {"weight": q.data, "weight_scale": q.scale, "weight_scale_2": q.tensor_scale, "input_scale": scale}

    def linear(self, x, weight, weight_scale, weight_scale_2, input_scale):
        return torch.ops.mstar.nvfp4_linear(x, weight, weight_scale, weight_scale_2, input_scale, self.nearest)

    # -- a modelopt checkpoint, as stored ----------------------------------------------------------------
    def storage(self, n, k):
        return (Storage("weight", (n, k // 2), torch.uint8, FP4_DTYPE, (1, 2)),
                Storage("weight_scale", (n, k // self.block), torch.uint8, FP8_DTYPE, (1, self.block)),
                Storage("weight_scale_2", (1,), torch.int32, torch.float32, (None, None)),
                Storage("input_scale", (1,), torch.int32, torch.float32, (None, None)))

    def from_checkpoint(self, name, tensor):
        if name in ("weight_scale_2", "input_scale"):
            return tensor.to(torch.float32).reshape(-1).view(torch.int32)
        return tensor.view(torch.uint8) if tensor.element_size() == 1 and tensor.dtype != torch.uint8 else tensor

    def after_load(self, tensors):
        """The block scales swizzled for the GEMM, one tensor scale and one input scale for the layer: a fused
        layer's shards (q/k/v, gate/up) must agree on both (modelopt calibrates them together, e.g. nvidia/GLM-5.3-
        Flash-NVFP4's gate/up), since one GEMM epilogue applies one of each; a checkpoint where they differ is
        refused rather than served with the other shards' values."""
        out = dict(tensors)
        for name in ("weight_scale_2", "input_scale"):
            values = tensors[name].reshape(-1)
            if values.numel() > 1 and not bool((values == values[0]).all()):
                raise ValueError(f"nvfp4: this layer's checkpoint shards have different {name} "
                                 f"({values.tolist()}); one GEMM takes one. Serve the layer bf16 instead")
            out[name] = values[:1].clone()
        scale = tensors["weight_scale"]
        if scale.dim() == 2:
            out["weight_scale"] = swizzle_block_scales(scale.view(torch.uint8)).view(FP8_DTYPE)
        return out

    def dequantize(self, q):
        """The fp32 weight of a ``BlockQuant``, or of a layer's buffers by suffix (as stored or as swizzled)."""
        return self._dequantize(q if isinstance(q, BlockQuant) else self._as_blockquant(q))

    def reference_stored(self, x: torch.Tensor, buffers) -> torch.Tensor:
        """What the GEMM computes on a layer's buffers: ``x`` quantized in units of its ``input_scale``, times the
        dequantized weight (tests, the experts' reference)."""
        return self.reference_linear(self.quantize(x, _f32(buffers["input_scale"])[:1], self.nearest),
                                     self._as_blockquant(buffers))

    @staticmethod
    def _as_blockquant(buffers) -> BlockQuant:
        weight = buffers["weight"].view(torch.uint8)
        scale = buffers["weight_scale"].view(torch.uint8)
        if scale.dim() == 2:
            scale = swizzle_block_scales(scale)
        return BlockQuant(weight.view(FP4_DTYPE), scale.view(FP8_DTYPE), _f32(buffers["weight_scale_2"])[:1])


def _f32(t: torch.Tensor) -> torch.Tensor:
    """An fp32 scale held in an int32 container, or any float, as a flat fp32 tensor."""
    return (t.view(torch.float32) if t.dtype == torch.int32 else t.float()).reshape(-1)


MXFP8 = register_format(Mxfp8Format())
NVFP4 = register_format(Nvfp4Format())
# a checkpoint's NVFP4, as stored (``Nvfp4Format.for_stored``): its activations round to nearest
NVFP4_STORED = Nvfp4Format(nearest=True)


def _out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


# The ops' default implementation is the reference (CPU tensors, tests); CUDA tensors take the kernels.
@torch.library.custom_op("mstar::mxfp8_linear", mutates_args=())
def mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """``x [M, K]`` bf16 times an MXFP8 weight ``[N, K]``: ``[M, N]`` bf16."""
    return MXFP8.reference_linear(MXFP8.quantize(x), BlockQuant(weight, weight_scale))


@mxfp8_linear.register_fake
def _(x, weight, weight_scale):
    return _out(x, weight)


@mxfp8_linear.register_kernel("cuda")
def _(x, weight, weight_scale):
    from mstar.utils.quant_fp8 import quant_mxfp8

    q = quant_mxfp8(x)
    # ``weight`` is row-major [N, K], as nn.Linear keeps it: its transpose is the col-major [K, N] the GEMM reads
    return torch._scaled_mm(q.data, weight.t(), scale_a=q.scale, scale_b=weight_scale, out_dtype=torch.bfloat16)


@torch.library.custom_op("mstar::nvfp4_linear", mutates_args=())
def nvfp4_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor, nearest: bool = False,
) -> torch.Tensor:
    """``x [M, K]`` bf16 times an NVFP4 weight ``[N, K // 2]``, ``x`` quantized in units of ``input_scale`` (its
    block scales rounded up, or to ``nearest``)."""
    return NVFP4.reference_linear(
        NVFP4.quantize(x, input_scale, nearest), BlockQuant(weight, weight_scale, weight_scale_2),
    )


@nvfp4_linear.register_fake
def _(x, weight, weight_scale, weight_scale_2, input_scale, nearest=False):
    return _out(x, weight)


@nvfp4_linear.register_kernel("cuda")
def _(x, weight, weight_scale, weight_scale_2, input_scale, nearest=False):
    from mstar.utils.quant_fp8 import quant_nvfp4

    q = quant_nvfp4(x, input_scale, nearest)
    ops = torch.nn.functional
    recipe = [ops.ScalingType.BlockWise1x16.value, ops.ScalingType.TensorWise.value]
    swizzle = [ops.SwizzleType.SWIZZLE_32_4_4.value, ops.SwizzleType.NO_SWIZZLE.value]
    return torch._scaled_mm_v2(
        q.data, weight.t(), [q.scale, q.tensor_scale], recipe, swizzle,
        [weight_scale, weight_scale_2], recipe, swizzle, None, torch.bfloat16,
    )
