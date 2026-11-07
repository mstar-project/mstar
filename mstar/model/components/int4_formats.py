"""4-bit weight-only linear formats: ``w4a16`` (int4 in groups along K) and ``mxfp4`` (e2m1, an e8m0 scale per 32).

Both hold the weight in compressed-tensors' layout, the one every int4 checkpoint is repacked into as it streams in:

- ``w4a16``: ``weight_packed`` int32 ``[N, K/8]`` (eight 4-bit values per word along K, the first in the low bits,
  each ``q + 8`` for a signed ``q``), ``weight_scale`` ``[N, K/g]`` in the checkpoint's bf16 or fp16 (an int16
  container), and when asymmetric ``weight_zero_point`` int32 ``[N/8, K/g]`` (eight zero points per word along N,
  likewise offset by 8). ``W = (u - z) * s`` with ``z = 8`` when symmetric. Kimi K2.x's routed experts, RedHat's
  W4A16 and cyankiwi's Qwen3.5 checkpoints ship it; ``adapt_checkpoint`` repacks AutoAWQ (``qweight [K, N/8]`` in
  AWQ's interleaved nibble order, ``qzeros``, ``scales [K/g, N]``) and GPTQ (``qweight [K/8, N]``, ``qzeros`` minus
  one in v1 checkpoints, ``g_idx``) into it, bit for bit. A GPTQ act-order checkpoint (``desc_act``, a ``g_idx``
  other than ``k // g``) is refused at boot (``checkpoint.scan_checkpoint``), and again here should one get past.
- ``mxfp4``: ``weight_packed`` uint8 ``[N, K/2]`` (two e2m1 codes a byte, the first in the low nibble) and
  ``weight_scale`` uint8 ``[N, K/32]`` (e8m0: ``2 ** (s - 127)``), compressed-tensors ``mxfp4-pack-quantized``
  (Kimi K3/K3.1's routed experts).

An int32 word of ``weight_packed`` is four bytes, low nibble first, so both formats read as uint8 ``[N, K/2]``
with ``k``'s code in byte ``k // 2``, nibble ``k % 2``: one Triton dequant GEMM (``utils.w4a16``) runs both, the
weight dequantized in registers to the activation's dtype (bf16 tensor cores; Ampere and later). Off CUDA ``linear``
is the reference, the fp32 dequantized weight rounded to the activation's dtype times the activation.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch
import torch.nn.functional as F

from mstar.model.components.quant_linear import LinearFormat, Storage, register_format

_U8, _I16, _I32 = torch.uint8, torch.int16, torch.int32
E8M0_DTYPE = torch.float8_e8m0fnu
# AutoAWQ packs output channel ``8j + AWQ_ORDER[i]`` into nibble ``i`` of word ``j``
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_SCALE_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "": torch.bfloat16}
_SHIFTS = torch.arange(8, dtype=torch.int64) * 4


def unpack_int4(packed: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """int32 words -> their eight 4-bit values (uint8) along ``dim``, the low nibble first."""
    p = packed.movedim(dim, -1).to(torch.int64) & 0xFFFFFFFF
    values = (p.unsqueeze(-1) >> _SHIFTS.to(p.device)) & 0xF
    return values.reshape(*p.shape[:-1], p.shape[-1] * 8).to(_U8).movedim(-1, dim)


def pack_int4(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The inverse of ``unpack_int4``: 4-bit values (any integer dtype, ``[0, 15]``) eight to an int32 word."""
    v = values.movedim(dim, -1).to(torch.int64)
    if v.shape[-1] % 8:
        v = F.pad(v, (0, 8 - v.shape[-1] % 8))
    words = (v.reshape(*v.shape[:-1], -1, 8) << _SHIFTS.to(v.device)).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(_I32).movedim(-1, dim).contiguous()


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """AutoAWQ's int32 ``[R, C/8]`` (interleaved nibble order) -> uint8 ``[R, C]``."""
    nibbles = unpack_int4(packed).view(*packed.shape, 8)
    out = torch.empty_like(nibbles)
    out[..., list(AWQ_ORDER)] = nibbles
    return out.view(packed.shape[0], -1)


def pack_awq(values: torch.Tensor) -> torch.Tensor:
    """The inverse of ``unpack_awq`` (tests write AWQ checkpoints with it)."""
    v = values.view(values.shape[0], -1, 8)
    return pack_int4(v[..., list(AWQ_ORDER)].reshape(values.shape))


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 ``[..., K/2]`` -> e2m1 codes ``[..., K]``, the low nibble first."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(_U8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES, device=codes.device)
    values = table[(codes & 7).long()]
    return torch.where(codes & 8 != 0, -values, values)


def e2m1_codes(u: torch.Tensor) -> torch.Tensor:
    """fp32 to e2m1 codes, round to nearest even (ties to 0, 1, 2, 4), saturating at 6."""
    a = u.abs()
    code = (
        (a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
        + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int() + (u < 0).int() * 8
    )
    return code.to(_U8)


# The kernels' layout ("planes"): in each 128-column block (16 words), word i's nibble j holds column 16j + i, so the
# j-th nibble of every word is 16 contiguous columns, whose activations a kernel reads as one slice. Built from the
# compressed-tensors layout at load; the scales and zero points don't move (a 16-column slice sits in one group).
PLANE_BLOCK = 128


def to_planes(packed: torch.Tensor) -> torch.Tensor:
    """Compressed-tensors' int32 ``[N, K/8]`` (or an MXFP4 weight's uint8 ``[N, K/2]``) as the planes layout, int32
    ``[N, K/8]``; K a multiple of 128."""
    words = packed.view(_I32) if packed.dtype == _U8 else packed
    n, w = words.shape
    codes = unpack_int4(words).view(n, w // 16, 8, 16)  # [n, block, j, i]: column 16j + i
    return pack_int4(codes.transpose(-1, -2).reshape(n, w * 8))


def from_planes(planes: torch.Tensor) -> torch.Tensor:
    """The inverse of ``to_planes``: int32 ``[N, K/8]`` in compressed-tensors' order."""
    n, w = planes.shape
    codes = unpack_int4(planes).view(n, w // 16, 16, 8)  # [n, block, i, j]
    return pack_int4(codes.transpose(-1, -2).reshape(n, w * 8))


def _as_int16(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_I16) if t.dtype.is_floating_point else t


class W4A16Format(LinearFormat):
    """int4 weights in groups of ``group`` along K, bf16/fp16 scales, zero points when not ``symmetric``. The
    registered instance is the on-the-fly variant (symmetric, group 128, bf16 scales); ``for_stored`` gives a
    checkpoint's (its group, symmetry, scale dtype and the layout ``adapt_checkpoint`` repacks from)."""

    name = "w4a16"

    def __init__(self, group: int = 128, symmetric: bool = True, scale_dtype: torch.dtype = torch.bfloat16,
                 layout: str = "compressed-tensors", zero_offset: int = 0):
        self.group, self.symmetric, self.scale_dtype = group, symmetric, scale_dtype
        self.layout, self.zero_offset = layout, zero_offset
        # Built at load, one 4-bit copy of the weight: marlin_* (the vendored Marlin GEMM's, ``mstar.utils.marlin``)
        # on CUDA where it builds and takes the shape; else weight_planes (the checkpoint's weight_packed in the
        # Triton kernels' layout, ``to_planes``) with its scales and zero points, plus tinygemm_* (torch's int4 GEMM,
        # 1-3 tokens) on CUDA. What a layer doesn't keep is an empty placeholder.
        self.buffers = (("weight_planes", "weight_scale") + (() if symmetric else ("weight_zero_point",))
                        + ("tinygemm_weight", "tinygemm_scale_zero", "marlin_weight", "marlin_scale")
                        + (() if symmetric else ("marlin_zero",)))

    def __repr__(self) -> str:
        return (f"w4a16(g{self.group}, {'sym' if self.symmetric else 'asym'}, {self.scale_dtype}, "
                f"{self.layout})")

    def for_stored(self, stored) -> W4A16Format:
        if stored is None or stored.format != self.name:
            return self
        return _w4a16_variant(
            stored.param("group", 128), stored.param("symmetric", True),
            _SCALE_DTYPES[stored.param("scale_dtype", "BF16")], stored.layout or "compressed-tensors",
            stored.param("zero_offset", 0),
        )

    # -- geometry, capability ---------------------------------------------------------------------------
    def check(self, out_features, in_features, dtype):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"w4a16 replaces a bf16 GEMM; this linear's weight is {dtype}")
        if in_features % self.group or in_features % PLANE_BLOCK:
            raise ValueError(f"w4a16 runs K in blocks of {PLANE_BLOCK} and groups of {self.group}; this linear's K "
                             f"is {in_features}")
        if not self.symmetric and out_features % 8:
            raise ValueError(f"asymmetric w4a16 packs zero points 8 rows a word; this linear's N is {out_features}")

    def unavailable_reason(self, device):
        return _triton_reason(device, self.name)

    # -- storage and the checkpoint ---------------------------------------------------------------------
    def storage(self, n, k):
        g = self.group
        out = [Storage("weight_packed", (n, k // 8), _I32, _I32, (1, 8)),
               Storage("weight_scale", (n, k // g), _I16, self.scale_dtype, (1, g))]
        if not self.symmetric:
            out.append(Storage("weight_zero_point", (-(-n // 8), k // g), _I32, _I32, (8, g)))
        return tuple(out)

    def from_checkpoint(self, name, tensor):
        if name == "weight_scale":
            return _as_int16(tensor.to(self.scale_dtype))
        return tensor

    def adapt_checkpoint(self, suffix, tensor):
        """AutoAWQ / GPTQ tensors -> the compressed-tensors layout, under its names; ``weight_shape`` (ct's
        unpadded shape, which a layout without padding doesn't need) and a trivial ``g_idx`` are dropped."""
        if suffix == "weight_shape":
            return None
        if self.layout == "awq":
            if suffix == "qweight":
                return "weight_packed", pack_int4(unpack_awq(tensor).t())
            if suffix == "qzeros":
                return "weight_zero_point", pack_int4(unpack_awq(tensor).t(), dim=0)
            if suffix == "scales":
                return "weight_scale", tensor.t().contiguous()
        elif self.layout == "gptq":
            if suffix == "qweight":
                return "weight_packed", pack_int4(unpack_int4(tensor, dim=0).t())
            if suffix == "qzeros":
                zeros = (unpack_int4(tensor).to(torch.int16) + self.zero_offset) & 0xF
                if self.symmetric:
                    if not bool((zeros == 8).all()):
                        raise ValueError("a symmetric GPTQ checkpoint (sym: true) holds zero points other than 8")
                    return None
                return "weight_zero_point", pack_int4(zeros.t(), dim=0)
            if suffix == "scales":
                return "weight_scale", tensor.t().contiguous()
            if suffix == "g_idx":
                trivial = torch.arange(tensor.numel(), dtype=torch.int64, device=tensor.device) // self.group
                if not torch.equal(tensor.reshape(-1).to(torch.int64), trivial):
                    raise ValueError("GPTQ act-order (desc_act) checkpoints permute K; this build serves int4 groups "
                                     "in K order only")
                return None
        return suffix, tensor

    def dequantize(self, buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "weight_packed" not in buffers and not buffers["weight_planes"].numel():
            return self._dequantize_marlin(buffers)
        packed = buffers["weight_packed"] if "weight_packed" in buffers else from_planes(buffers["weight_planes"])
        u = unpack_int4(packed).float()
        n, k = u.shape
        scale = buffers["weight_scale"]
        scale = (scale.view(self.scale_dtype) if scale.dtype == _I16 else scale).float()
        if self.symmetric:
            zero = torch.full_like(scale, 8.0)
        else:
            zero = unpack_int4(buffers["weight_zero_point"], dim=0)[:n].float()
        return ((u.view(n, -1, self.group) - zero[..., None]) * scale[..., None]).view(n, k)

    def quantize_weight(self, weight, input_amax):
        return self.after_load(self.quantize_storage(weight))

    def quantize_storage(self, weight):
        """Symmetric round-to-nearest in groups (compressed-tensors' and GPTQ's scale ``amax / 7.5``), as a
        checkpoint stores it."""
        n, k = weight.shape
        groups = weight.float().view(n, k // self.group, self.group)
        scale = (groups.abs().amax(-1) / 7.5).clamp_min(torch.finfo(torch.float32).tiny).to(self.scale_dtype)
        q = torch.round(groups / scale.float()[..., None]).clamp(-8, 7) + 8
        out = {"weight_packed": pack_int4(q.view(n, k).to(_U8)), "weight_scale": scale}
        if not self.symmetric:
            out["weight_zero_point"] = pack_int4(torch.full((n, k // self.group), 8, dtype=_U8,
                                                            device=weight.device), dim=0)
        return out

    def _dequantize_marlin(self, buffers) -> torch.Tensor:
        from mstar.utils import marlin

        kind = KIND_INT4_SYM if self.symmetric else KIND_INT4_ZP
        held = {name: buffers[name] for name in buffers if name.startswith("marlin_")}
        if held["marlin_scale"].dtype == _I16:  # the module's container: Marlin's scales are bf16
            held["marlin_scale"] = held["marlin_scale"].view(torch.bfloat16)
        codes, scale, *zero = marlin.unprepare(kind, held)
        n, k = codes.shape
        zero = zero[0].float()[..., None] if zero else 8.0
        return ((codes.float().view(n, -1, self.group) - zero) * scale.float()[..., None]).view(n, k)

    def after_load(self, tensors):
        """One 4-bit copy: Marlin's where it builds (the checkpoint layout's placeholders empty), else the Triton
        kernels' layout (``weight_planes``) and tinygemm's."""
        if "weight_planes" in tensors:
            return dict(tensors)
        marlin = self._marlin(tensors)
        if marlin["marlin_weight"].numel():
            dev = marlin["marlin_weight"].device
            out = {name: torch.empty(0, dtype=t.dtype, device=dev) for name, t in tensors.items()
                   if name != "weight_packed"}
            out["weight_planes"] = torch.empty(0, dtype=_I32, device=dev)
            return {**out, **self._tinygemm(tensors, skip=True), **marlin}
        out = {name: t for name, t in tensors.items() if name != "weight_packed"}
        out["weight_planes"] = to_planes(tensors["weight_packed"])
        return {**out, **self._tinygemm(tensors), **marlin}

    def _marlin(self, tensors) -> dict[str, torch.Tensor]:
        packed = tensors["weight_packed"]
        n, k = packed.shape[0], packed.shape[1] * 8
        kind = KIND_INT4_SYM if self.symmetric else KIND_INT4_ZP
        names = ("marlin_weight", "marlin_scale") + (() if self.symmetric else ("marlin_zero",))
        if packed.is_cuda:
            from mstar.utils import marlin

            if marlin.shape_reason(kind, self.group, n, k) is None and marlin.available(kind, self.group,
                                                                                         packed.device):
                scale = tensors["weight_scale"]
                scale = scale.view(self.scale_dtype) if scale.dtype == _I16 else scale
                zero = None if self.symmetric else unpack_int4(tensors["weight_zero_point"].view(_I32), dim=0)[:n]
                return marlin.prepare(kind, unpack_int4(packed.view(_I32)), scale, zero)
        return {name: torch.empty(0, dtype=_I32, device=packed.device) for name in names}

    def _tinygemm(self, tensors, skip: bool = False) -> dict[str, torch.Tensor]:
        """torch's int4 GEMM's copy of the weight (``aten._convert_weight_to_int4pack``: uint8 ``[N, K/2]`` with the
        even column in the high nibble; scales and zeros ``[K/g, N, 2]`` bf16 for ``(q - 8) * scale + zero``, so a
        zero point z is ``(8 - z) * scale``), on CUDA for groups of 32-256; empty placeholders elsewhere. It doubles
        the layer's 4-bit bytes for tinygemm's 1.6-1.9x over bf16 at 1-2 tokens on an H200 (job 1479), where the
        Triton GEMV reached ~1x."""
        packed = tensors["weight_packed"]
        n, k = packed.shape[0], packed.shape[1] * 8
        usable = (not skip and packed.is_cuda and self.group in (32, 64, 128, 256) and k % 128 == 0 and n % 8 == 0
                  and hasattr(torch.ops.aten, "_convert_weight_to_int4pack"))
        if not usable:
            return {"tinygemm_weight": torch.empty(0, dtype=_I32, device=packed.device),
                    "tinygemm_scale_zero": torch.empty(0, dtype=torch.bfloat16, device=packed.device)}
        codes = unpack_int4(packed.view(_I32))
        u8 = ((codes[:, 0::2] << 4) | codes[:, 1::2]).contiguous()
        scale = tensors["weight_scale"]
        scale = (scale.view(self.scale_dtype) if scale.dtype == _I16 else scale).float()
        if self.symmetric:
            zero = torch.zeros_like(scale)
        else:
            z = unpack_int4(tensors["weight_zero_point"].view(_I32), dim=0)[:n].float()
            zero = (8.0 - z) * scale
        scale_zero = torch.stack([scale.t(), zero.t()], dim=-1).to(torch.bfloat16).contiguous()
        return {"tinygemm_weight": torch.ops.aten._convert_weight_to_int4pack(u8, 8),
                "tinygemm_scale_zero": scale_zero}

    # -- the GEMM ------------------------------------------------------------------------------------------
    def prepare_device(self, device):
        if device.type == "cuda":
            from mstar.utils import w4a16  # noqa: F401  (registers the CUDA kernel before any compile)

    def linear(self, x, weight_planes, weight_scale, *rest):
        if self.symmetric:
            (tg_weight, tg_scale_zero, m_weight, m_scale), zero_point, m_zero = rest, None, None
        else:
            zero_point, tg_weight, tg_scale_zero, m_weight, m_scale, m_zero = rest
        scale = weight_scale.view(self.scale_dtype) if weight_scale.dtype == _I16 else weight_scale
        if tg_scale_zero.dtype == _I16:
            tg_scale_zero = tg_scale_zero.view(torch.bfloat16)
        if m_scale.dtype == _I16:
            m_scale = m_scale.view(torch.bfloat16)
        kind = KIND_INT4_SYM if zero_point is None else KIND_INT4_ZP
        return torch.ops.mstar.w4a16_linear(x, weight_planes.view(_U8), scale, zero_point, self.group, kind,
                                            tg_weight, tg_scale_zero, m_weight, m_scale, m_zero)


@lru_cache(maxsize=None)
def _w4a16_variant(group, symmetric, scale_dtype, layout, zero_offset) -> W4A16Format:
    return W4A16Format(group, symmetric, scale_dtype, layout, zero_offset)


class Mxfp4Format(LinearFormat):
    """e2m1 codes two a byte, one e8m0 scale per 32 along K (OCP MX)."""

    name = "mxfp4"
    # weight_planes: the checkpoint's weight_packed in the Triton kernels' layout (``to_planes``); marlin_*: the
    # vendored Marlin GEMM's copy where it builds (CUDA); both built at load
    buffers = ("weight_planes", "weight_scale", "marlin_weight", "marlin_scale")
    group = 32

    def check(self, out_features, in_features, dtype):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"mxfp4 replaces a bf16 GEMM; this linear's weight is {dtype}")
        if in_features % PLANE_BLOCK:
            raise ValueError(f"mxfp4 runs K in blocks of {PLANE_BLOCK}; this linear's K is {in_features}")

    def unavailable_reason(self, device):
        return _triton_reason(device, self.name)

    def storage(self, n, k):
        return (Storage("weight_packed", (n, k // 2), _U8, _U8, (1, 2)),
                Storage("weight_scale", (n, k // 32), _U8, E8M0_DTYPE, (1, 32)))

    def from_checkpoint(self, name, tensor):
        return tensor.view(_U8) if tensor.dtype != _U8 and tensor.element_size() == 1 else tensor

    def dequantize(self, buffers):
        scale = buffers.get("weight_scale")
        if "weight_packed" in buffers:
            codes = unpack_fp4(buffers["weight_packed"].view(_U8))
        elif buffers["weight_planes"].numel():
            codes = unpack_int4(from_planes(buffers["weight_planes"].view(_I32)))
        else:
            from mstar.utils import marlin

            codes, scale = marlin.unprepare(KIND_E2M1, {"marlin_weight": buffers["marlin_weight"],
                                                        "marlin_scale": buffers["marlin_scale"]})
        n, k = codes.shape
        scale = torch.exp2(scale.view(_U8).float() - 127.0)
        return (e2m1_values(codes).view(n, -1, 32) * scale[..., None]).view(n, k)

    def quantize_weight(self, weight, input_amax):
        return self.after_load(self.quantize_storage(weight))

    def after_load(self, tensors):
        if "weight_planes" in tensors:
            return dict(tensors)
        packed = tensors["weight_packed"].view(_U8)
        out = {name: t for name, t in tensors.items() if name != "weight_packed"}
        out["weight_planes"] = to_planes(packed)
        n, k = packed.shape[0], packed.shape[1] * 2
        empty = torch.empty(0, dtype=_I32, device=packed.device)
        out.update(marlin_weight=empty, marlin_scale=empty)
        if packed.is_cuda:
            from mstar.utils import marlin

            if marlin.shape_reason(KIND_E2M1, 32, n, k) is None and marlin.available(KIND_E2M1, 32, packed.device):
                # one copy: Marlin's, the planes layout and its scales left empty
                out.update(marlin.prepare(KIND_E2M1, unpack_fp4(packed), tensors["weight_scale"].view(_U8)))
                out.update(weight_planes=empty, weight_scale=torch.empty(0, dtype=_U8, device=packed.device))
        return out

    def quantize_storage(self, weight):
        """A group's scale is the smallest power of two at or above ``amax / 6``, so nothing clips."""
        n, k = weight.shape
        groups = weight.float().view(n, k // 32, 32)
        amax = groups.abs().amax(-1)
        mantissa, exponent = torch.frexp(amax / 6.0)
        biased = (exponent - (mantissa == 0.5).to(exponent.dtype) + 127).clamp(1, 254)
        biased = torch.where(amax == 0, torch.full_like(biased, 127), biased)
        scale = torch.ldexp(torch.ones_like(amax), biased - 127)
        codes = e2m1_codes((groups / scale[..., None]).view(n, k))
        return {"weight_packed": pack_fp4(codes), "weight_scale": biased.to(_U8)}

    def prepare_device(self, device):
        if device.type == "cuda":
            from mstar.utils import w4a16  # noqa: F401

    def linear(self, x, weight_planes, weight_scale, marlin_weight, marlin_scale):
        return torch.ops.mstar.w4a16_linear(x, weight_planes.view(_U8), weight_scale.view(_U8), None, 32,
                                            KIND_E2M1, None, None, marlin_weight, marlin_scale.view(_U8), None)


def _triton_reason(device: torch.device, name: str) -> str | None:
    from mstar.platforms import current_platform

    if device.type != "cuda" or not current_platform.has_cuda_capability(8, 0, device):
        return f"{name} needs bf16 tensor cores (CUDA compute capability 8.0 and up)"
    try:
        import triton  # noqa: F401
    except ImportError:
        return f"{name}'s GEMM is a Triton kernel, and triton is not installed"
    return None


# the weight decodes of ``mstar::w4a16_linear`` (and the grouped experts kernel)
KIND_INT4_SYM, KIND_INT4_ZP, KIND_E2M1 = 0, 1, 2


def dequantize_kind(packed_u8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int,
                    kind: int, planes: bool = True) -> torch.Tensor:
    """fp32 ``[N, K]`` from the op's operands (``packed_u8 [N, K/2]``, in the planes layout unless ``planes`` is
    False, as the experts hold it): the reference the kernels match."""
    codes = unpack_int4(from_planes(packed_u8.view(_I32))) if planes else unpack_fp4(packed_u8)
    n, k = codes.shape
    if kind == KIND_E2M1:
        values, scale = e2m1_values(codes), torch.exp2(scale.view(_U8).float() - 127.0)
    else:
        zero = 8.0 if zero_point is None else unpack_int4(zero_point, dim=0)[:n].float()[..., None]
        values = codes.float().view(n, -1, group) - zero
        scale = scale.float()
    return (values.view(n, -1, group) * scale[..., None]).view(n, k)


@torch.library.custom_op("mstar::w4a16_linear", mutates_args=())
def w4a16_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int,
    kind: int, tinygemm_weight: torch.Tensor | None = None, tinygemm_scale_zero: torch.Tensor | None = None,
    marlin_weight: torch.Tensor | None = None, marlin_scale: torch.Tensor | None = None,
    marlin_zero: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x [M, K]`` times a 4-bit weight (uint8 ``[N, K/2]`` in the planes layout, its group scales, zero points for
    ``KIND_INT4_ZP``; optionally tinygemm's or Marlin's copy of it) dequantized to ``x``'s dtype: ``[M, N]``.
    Off CUDA, the reference."""
    w = dequantize_kind(weight, scale, zero_point, group, kind).to(x.dtype)
    return F.linear(x, w)


@w4a16_linear.register_fake
def _(x, weight, scale, zero_point, group, kind, tinygemm_weight=None, tinygemm_scale_zero=None, marlin_weight=None,
      marlin_scale=None, marlin_zero=None):
    return x.new_empty((x.shape[0], weight.shape[0]))


W4A16 = register_format(W4A16Format())
MXFP4 = register_format(Mxfp4Format())
