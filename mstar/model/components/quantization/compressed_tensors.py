"""Reader for the compressed-tensors checkpoint format (neuralmagic / vLLM).

Model-agnostic: this is a *serialization format*, not one model's quirk. Kimi-K2.7
is simply the first mstar model whose checkpoint ships in it.

Packed values are stored low-order-first in int32 containers. Symmetric INT4 is
offset-binary, so dequant subtracts 8 to match vLLM's ``uint4b8`` layout.

:class:`CompressedTensorsQuantConfig` describes how a checkpoint was *written*;
:meth:`CompressedTensorsQuantConfig.moe_quant_data` translates that into the
:class:`~mstar.utils.quantization.QuantizationData` a specific kernel call needs.
:attr:`~CompressedTensorsQuantConfig.quant_type` is the only place ``num_bits`` is
mapped to a supported scheme; :meth:`~CompressedTensorsQuantConfig.ensure_kernel_support`
is how callers reject a width no kernel implements.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch

from mstar.utils.quantization import QuantizationData, QuantizationType, W4A16Data

_PACKED = ".weight_packed"
_SCALE = ".weight_scale"
_ZERO_POINT = ".weight_zero_point"
_SHAPE = ".weight_shape"
_QUANT_SUFFIXES = (_PACKED, _SCALE, _ZERO_POINT, _SHAPE)


@dataclass(frozen=True)
class CompressedTensorsQuantConfig:
    """Subset of the compressed-tensors config used by Kimi-K2.7."""

    num_bits: int = 4
    group_size: int = 32
    symmetric: bool = True
    strategy: str = "group"  # "group" | "channel"
    quant_format: str = "pack-quantized"
    quant_method: str = "compressed-tensors"
    ignore: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pack_factor(self) -> int:
        return 32 // self.num_bits

    @property
    def quant_type(self) -> QuantizationType | None:
        """The scheme mstar's kernels implement for this checkpoint, else ``None``.

        The single ``num_bits`` -> scheme decision. ``None`` means "no kernel for
        this width"; call :meth:`ensure_kernel_support` to turn that into an error.
        """
        return QuantizationType.W4A16 if self.num_bits == 4 else None

    def ensure_kernel_support(self) -> QuantizationType:
        """Return this checkpoint's scheme, raising if mstar has no kernel for it.

        Call before allocating packed parameters or selecting a backend. The
        packed-expert kernels hardcode 4-bit nibble extraction, so an INT8
        checkpoint would otherwise allocate self-consistent shapes, load without
        complaint, and return wrong numbers — this turns that into a load-time
        error naming the width the checkpoint declared.
        """
        quant_type = self.quant_type
        if quant_type is None:
            raise ValueError(
                f"compressed-tensors checkpoint declares num_bits={self.num_bits}, which "
                "mstar does not implement (supported: 4-bit / W4A16). The fused-MoE and "
                "Marlin kernels are INT4-only."
            )
        return quant_type

    def moe_quant_data(
        self,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w1_zp: torch.Tensor | None = None,
        w2_zp: torch.Tensor | None = None,
    ) -> QuantizationData:
        """Companion data for one stacked routed-expert GEMM under this config.

        Call per dispatch rather than caching on the module: the scales are
        ``nn.Parameter`` s that ``Module._apply`` rebuilds on ``.to(device)``, so
        binding them at call time keeps the returned object from ever holding a
        stale tensor.
        """
        quant_type = self.ensure_kernel_support()
        if quant_type is QuantizationType.W4A16:
            return W4A16Data(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                group_size=self.group_size,
                w1_zp=None if self.symmetric else w1_zp,
                w2_zp=None if self.symmetric else w2_zp,
            )
        raise ValueError(f"No routed-expert quantization data for {quant_type}")

    @classmethod
    def from_hf_config_dict(
        cls, quant: dict | None
    ) -> "CompressedTensorsQuantConfig | None":
        if not quant:
            return None
        groups = quant.get("config_groups") or {}
        weights: dict = {}
        if groups:
            first = next(iter(groups.values()))
            weights = (first or {}).get("weights") or {}
        strategy = weights.get("strategy", "group")
        group_size = weights.get("group_size")
        if group_size is None:
            group_size = -1 if strategy == "channel" else 32
        return cls(
            num_bits=int(weights.get("num_bits", 4)),
            group_size=int(group_size),
            symmetric=bool(weights.get("symmetric", True)),
            strategy=str(strategy),
            quant_format=str(quant.get("format", "pack-quantized")),
            quant_method=str(quant.get("quant_method", "compressed-tensors")),
            ignore=tuple(quant.get("ignore", []) or []),
        )

def pack_int32(values_unsigned: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack unsigned values along the last axis into low-order-first int32."""
    pack_factor = 32 // num_bits
    *lead, n = values_unsigned.shape
    if n % pack_factor != 0:
        raise ValueError(f"last dim {n} not divisible by pack_factor {pack_factor}")
    q = values_unsigned.to(torch.int64).reshape(*lead, n // pack_factor, pack_factor)
    shifts = torch.arange(pack_factor, device=q.device, dtype=torch.int64) * num_bits
    packed = (q << shifts).sum(dim=-1)
    return (packed & 0xFFFFFFFF).to(torch.int32)


def unpack_int32(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_int32`; reads int32 as an unsigned bit pattern."""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    p = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(pack_factor, device=p.device, dtype=torch.int64) * num_bits
    unpacked = (p.unsqueeze(-1) >> shifts) & mask  # (..., last, pack_factor)
    *lead, m, _ = unpacked.shape
    return unpacked.reshape(*lead, m * pack_factor)


def dequantize_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    symmetric: bool,
    zero_point: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one packed compressed-tensors weight."""
    nibbles = unpack_int32(packed, num_bits).to(torch.float32)  # (out, in) unsigned
    out_f, in_f = nibbles.shape
    gs = in_f if group_size in (-1, None) else group_size
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {gs}")

    if symmetric:
        nibbles -= float(1 << (num_bits - 1))  # offset-binary: nibble - bias
    else:
        if zero_point is None:
            raise ValueError("asymmetric quantization requires a zero_point")
        zp = zero_point.to(torch.float32)
        if zp.shape[-1] != in_f:  # per-group -> broadcast to per-column
            zp = zp.repeat_interleave(gs, dim=-1)
        nibbles -= zp

    s = scale.to(torch.float32)
    if s.shape[-1] != in_f:  # per-group -> broadcast to per-column
        s = s.repeat_interleave(gs, dim=-1)
    return (nibbles * s).to(out_dtype)
def dequant_compressed_tensors_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: CompressedTensorsQuantConfig,
    out_dtype: torch.dtype = torch.bfloat16,
    keep_packed: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert quantized sub-key streams to bf16 weights unless ``keep_packed`` matches."""
    buffers: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in weights:
        suffix = next((s for s in _QUANT_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            yield name, tensor
            continue

        base = name[: -len(suffix)]
        if keep_packed is not None and keep_packed(base):
            yield name, tensor
            continue

        slot = buffers.setdefault(base, {})
        slot[suffix] = tensor

        have_core = _PACKED in slot and _SCALE in slot
        have_zp = quant_config.symmetric or _ZERO_POINT in slot
        if have_core and have_zp:
            weight = dequantize_weight(
                slot[_PACKED],
                slot[_SCALE],
                num_bits=quant_config.num_bits,
                group_size=quant_config.group_size,
                symmetric=quant_config.symmetric,
                zero_point=slot.get(_ZERO_POINT),
                out_dtype=out_dtype,
            )
            del buffers[base]
            yield base + ".weight", weight

    if buffers:
        missing = {b: sorted(slot) for b, slot in buffers.items()}
        raise ValueError(
            f"compressed-tensors stream ended with incomplete quantized tensors "
            f"(missing weight_packed and/or weight_scale): {missing}"
        )
