"""FP8 (e4m3) linear formats for Hopper and later: ``fp8_tensor``, ``fp8_channel`` and ``fp8_block``.

Each keeps the weight as e4m3 in a uint8 container and its fp32 scales in an int32 one (the engine casts a
loaded node's floating tensors to its dtype), under the checkpoint's own names, so a pre-quantized checkpoint
loads straight into them and a bf16 one is quantized at load into the same layout:

- ``fp8_tensor``: ``weight_scale [1]``, the activation scaled per tensor each call (compressed-tensors /
  modelopt FP8); ``torch._scaled_mm`` TensorWise.
- ``fp8_channel``: ``weight_scale [N, 1]``, the activation per token (compressed-tensors channel/token);
  ``torch._scaled_mm`` RowWise.
- ``fp8_block``: ``weight_scale_inv [N/128, K/128]`` (DeepSeek, Qwen, GLM, MiniMax FP8); the ``fp8_linear``
  kernel registry (``utils.dense_fp8``: Triton w8a16 for a few tokens, FlashInfer's or cuBLAS's block-scaled
  GEMM with the activation quantized per 128 columns above).

Off CUDA every format computes what its operands stand for in fp32 (tests); a deployment never runs that, the
resolver refuses a GPU without e4m3 tensor cores.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
import torch.nn.functional as F

from mstar.model.components.quant_linear import LinearFormat, Storage, register_format
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight, quantize_fp8_block_weight

_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_TINY = torch.finfo(torch.float32).tiny
_U8, _I32, _F32 = torch.uint8, torch.int32, torch.float32


def _to_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.view(_F32) if t.dtype == _I32 else t.float()


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    return t.view(FP8_DTYPE) if t.dtype == _U8 else t


class _Fp8Format(LinearFormat):
    scale_name = "weight_scale"

    def unavailable_reason(self, device: torch.device) -> str | None:
        from mstar.platforms import current_platform

        if device.type == "cuda" and current_platform.has_cuda_capability(8, 9, device):
            return None
        return f"{self.name} needs e4m3 tensor cores (CUDA compute capability 8.9 and up)"

    def check(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"{self.name} replaces a bf16 GEMM; this linear's weight is {dtype}")
        if out_features % 16 or in_features % 16:
            raise ValueError(
                f"{self.name} needs N and K multiples of 16; this linear is {out_features} x {in_features}")

    def from_checkpoint(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name == "weight":
            return tensor.view(_U8) if tensor.dtype == FP8_DTYPE else tensor
        if name == self.scale_name:
            return self._scale_shape(tensor.to(_F32)).contiguous().view(_I32)
        return tensor

    def _scale_shape(self, scale: torch.Tensor) -> torch.Tensor:
        return scale

    def dequantize(self, buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weight = _e4m3(buffers["weight"]).float()
        return weight * self._scale_shape(_f32(buffers[self.scale_name]))


class Fp8TensorFormat(_Fp8Format):
    name = "fp8_tensor"
    buffers = ("weight", "weight_scale")

    def _scale_shape(self, scale):
        return scale.reshape(1)

    def storage(self, n, k):
        return (Storage("weight", (n, k), _U8, FP8_DTYPE, (1, 1)),
                Storage("weight_scale", (1,), _I32, _F32, (None, None)))

    def quantize_weight(self, weight, input_amax):
        scale = (weight.float().abs().amax() / _FP8_MAX).clamp_min(_TINY).reshape(1)
        return {"weight": _to_e4m3(weight, scale).view(_U8), "weight_scale": scale.view(_I32)}

    def dequantize(self, buffers):
        return _e4m3(buffers["weight"]).float() * _f32(buffers["weight_scale"]).reshape(())

    def linear(self, x, weight, weight_scale):
        scale_b = weight_scale.view(_F32)
        scale_a = (x.float().abs().amax() / _FP8_MAX).clamp_min(_TINY).reshape(1)
        xq = _to_e4m3(x, scale_a)
        if not x.is_cuda:
            return ((xq.float() * scale_a) @ (_e4m3(weight).float() * scale_b).t()).to(x.dtype)
        return torch._scaled_mm(xq, _e4m3(weight).t(), scale_a=scale_a, scale_b=scale_b, out_dtype=x.dtype)


class Fp8ChannelFormat(_Fp8Format):
    name = "fp8_channel"
    buffers = ("weight", "weight_scale")

    def _scale_shape(self, scale):
        return scale.reshape(-1, 1)

    def storage(self, n, k):
        return (Storage("weight", (n, k), _U8, FP8_DTYPE, (1, 1)),
                Storage("weight_scale", (n, 1), _I32, _F32, (1, None)))

    def quantize_weight(self, weight, input_amax):
        scale = (weight.float().abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(_TINY)
        return {"weight": _to_e4m3(weight, scale).view(_U8), "weight_scale": scale.contiguous().view(_I32)}

    def linear(self, x, weight, weight_scale):
        scale_b = weight_scale.view(_F32)  # [N, 1]
        scale_a = (x.float().abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(_TINY)
        xq = _to_e4m3(x, scale_a)
        if not x.is_cuda:
            return ((xq.float() * scale_a) @ (_e4m3(weight).float() * scale_b).t()).to(x.dtype)
        # ``weight`` is row-major [N, K]: its transpose is the column-major [K, N] the GEMM reads
        return torch._scaled_mm(xq, _e4m3(weight).t(), scale_a=scale_a, scale_b=scale_b.t(), out_dtype=x.dtype)


class Fp8BlockFormat(_Fp8Format):
    """``block`` is the checkpoint's scale tile (128 x 128 for every released checkpoint; ``for_block`` makes
    another, which the kernels run as w8a16). ``reference`` runs the weight dequantized per call on any device:
    a model's numerics anchor (GLM-5.2's ``kernels: reference``)."""

    name = "fp8_block"
    buffers = ("weight", "weight_scale_inv")
    scale_name = "weight_scale_inv"
    _variants: ClassVar[dict] = {}

    def __init__(self, block: tuple[int, int] = (128, 128), reference: bool = False):
        self.block = tuple(block)
        self.reference = reference

    @classmethod
    def for_block(cls, block: tuple[int, int] | None, reference: bool = False) -> "Fp8BlockFormat":
        """The registered 128 x 128 format, or a variant for another block or the reference kernels."""
        key = (tuple(block or FP8_BLOCK.block), reference)
        if key == (FP8_BLOCK.block, False):
            return FP8_BLOCK
        if key not in cls._variants:
            cls._variants[key] = cls(*key)
        return cls._variants[key]

    def check(self, out_features, in_features, dtype):
        super().check(out_features, in_features, dtype)
        bn, bk = self.block
        if out_features % bn or in_features % bk:
            raise ValueError(
                f"fp8_block needs N and K multiples of {bn}; this linear is {out_features} x {in_features}")

    def storage(self, n, k):
        bn, bk = self.block
        return (Storage("weight", (n, k), _U8, FP8_DTYPE, (1, 1)),
                Storage("weight_scale_inv", (-(-n // bn), -(-k // bk)), _I32, _F32, (bn, bk)))

    def quantize_weight(self, weight, input_amax):
        q, scale_inv = quantize_fp8_block_weight(weight, self.block)
        return {"weight": q.view(_U8), "weight_scale_inv": scale_inv.contiguous().view(_I32)}

    def dequantize(self, buffers):
        return dequantize_fp8_block_weight(
            _e4m3(buffers["weight"]), _f32(buffers["weight_scale_inv"]), self.block, out_dtype=_F32,
        )

    def prepare_device(self, device):
        if device.type == "cuda":
            from mstar.utils import dense_fp8  # registers the op before any compile

            dense_fp8.reserve(device)

    has_glu = True

    def linear(self, x, weight, weight_scale_inv):
        return self._run(x, weight, weight_scale_inv, glu=False)

    def glu_linear(self, x, weight, weight_scale_inv):
        # dense_fp8's w8a16 kernel applies SwiGLU in its epilogue; its larger-M paths run the GEMM then one
        # act-and-mul, as a separate SwiGLU would
        return self._run(x, weight, weight_scale_inv, glu=True)

    def _run(self, x, weight, weight_scale_inv, glu: bool):
        scale = weight_scale_inv.view(_F32)
        if self.reference or not x.is_cuda:
            # the weight as a bf16 checkpoint's loader would hold it (dequantized to bf16), in x's dtype
            w = dequantize_fp8_block_weight(_e4m3(weight), scale, self.block).to(x.dtype)
            y = F.linear(x, w)
            if glu:
                gate, up = y.chunk(2, dim=-1)
                y = F.silu(gate) * up
            return y
        from mstar.utils import dense_fp8

        return dense_fp8.linear(x, weight, scale, self.block, glu)


FP8_TENSOR = register_format(Fp8TensorFormat())
FP8_CHANNEL = register_format(Fp8ChannelFormat())
FP8_BLOCK = register_format(Fp8BlockFormat())
