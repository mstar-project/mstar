"""Compressed-tensors INT4 (W4A16) quantization for Kimi-K2.7 weights.

On-disk format (compressed-tensors ``pack-quantized``), per quantized Linear
weight of logical shape ``(out, in)``:

  * ``.weight_packed`` — int32 ``(out, in // pack_factor)``; ``pack_factor =
    32 // num_bits`` (8 for INT4) values packed low-order-first along the input axis.
  * ``.weight_scale`` — bf16 ``(out, in // group_size)``, one scale per (row, group).
  * ``.weight_zero_point`` — asymmetric only.
  * ``.weight_shape`` — original ``(out, in)``; optional, used only to validate.

Symmetric INT4 is stored offset-binary: the packed nibble is ``(signed value + 8)``,
so dequant subtracts 8 (matches vLLM's ``uint4b8``); asymmetric subtracts the zero
point. To flip a checkpoint to plain two's-complement, change the ``bias`` line in
:func:`dequantize_weight`. Layout authority: vLLM
``compressed_tensors/schemes/compressed_tensors_wNa16.py``.

``dequant_compressed_tensors_stream`` dequantizes a checkpoint stream to bf16 on
load; ``keep_packed`` leaves selected weights packed for in-kernel dequant.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch

# Suffixes a compressed-tensors checkpoint attaches to each quantized tensor.
_PACKED = ".weight_packed"
_SCALE = ".weight_scale"
_ZERO_POINT = ".weight_zero_point"
_SHAPE = ".weight_shape"
_QUANT_SUFFIXES = (_PACKED, _SCALE, _ZERO_POINT, _SHAPE)


@dataclass(frozen=True)
class CompressedTensorsQuantConfig:
    """The subset of a compressed-tensors ``quantization_config`` this port reads.

    Kimi-K2.7 ships a single ``config_groups`` entry (``weights`` only, W4A16), so
    the whole checkpoint shares one ``num_bits`` / ``group_size`` / ``symmetric``.
    """

    num_bits: int = 4
    group_size: int = 32  # -1 => channelwise (one group spans the full input dim)
    symmetric: bool = True
    strategy: str = "group"  # "group" | "channel"
    quant_format: str = "pack-quantized"
    quant_method: str = "compressed-tensors"
    ignore: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pack_factor(self) -> int:
        """Number of ``num_bits`` values packed into one int32."""
        return 32 // self.num_bits

    @classmethod
    def from_hf_config_dict(
        cls, quant: dict | None
    ) -> "CompressedTensorsQuantConfig | None":
        """Build from a checkpoint ``config.json``'s ``quantization_config`` block.

        Returns ``None`` when there is no quantization block (a plain bf16
        checkpoint). Reads the first ``config_groups`` entry's ``weights`` spec —
        Kimi uses exactly one group.
        """
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


# ---------------------------------------------------------------------------
# Bit packing — exact inverses. Packing is along the last (input) axis, which is
# the input axis of a checkpoint ``(out, in)`` Linear weight (what gets quantized).
# ---------------------------------------------------------------------------

def pack_int32(values_unsigned: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack ``pack_factor`` unsigned ``num_bits`` values (last axis) into int32.

    ``values_unsigned`` holds integers in ``[0, 2**num_bits)``; the last axis must
    be divisible by ``pack_factor = 32 // num_bits``. Values are combined
    low-order-first (element ``j`` occupies bits ``[num_bits*j, num_bits*(j+1))``),
    matching compressed-tensors. Returns int32 of shape
    ``(..., last // pack_factor)``.
    """
    pack_factor = 32 // num_bits
    *lead, n = values_unsigned.shape
    if n % pack_factor != 0:
        raise ValueError(f"last dim {n} not divisible by pack_factor {pack_factor}")
    q = values_unsigned.to(torch.int64).reshape(*lead, n // pack_factor, pack_factor)
    shifts = torch.arange(pack_factor, device=q.device, dtype=torch.int64) * num_bits
    packed = (q << shifts).sum(dim=-1)
    # Wrap the 32-bit pattern into a signed int32 container (matches on-disk dtype).
    return (packed & 0xFFFFFFFF).to(torch.int32)


def unpack_int32(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_int32`: expand int32 to unsigned ``num_bits`` nibbles.

    Returns an int64 tensor of shape ``(..., last * pack_factor)`` with values in
    ``[0, 2**num_bits)``. The int32 is read as an unsigned 32-bit pattern, so the
    top nibble is recovered correctly regardless of the container's sign bit.
    """
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
    """Dequantize one compressed-tensors weight to ``out_dtype``.

    Args:
        packed: ``(out, in // pack_factor)`` int32 packed weight.
        scale: ``(out, in // group_size)`` per-(row, group) scale.
        num_bits: bit width (4 for Kimi INT4).
        group_size: group granularity along the input axis; ``-1`` => channelwise.
        symmetric: symmetric offset-binary (subtract ``2**(num_bits-1)``) vs.
            asymmetric (subtract ``zero_point``).
        zero_point: ``(out, in // group_size)`` per-group zero point (asymmetric).
        out_dtype: result dtype (bf16 to feed the existing fused-expert GEMM).

    Returns:
        ``(out, in)`` dequantized weight, in ``out_dtype``.
    """
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

# ---------------------------------------------------------------------------
# Streaming dequant-on-load — the generator wired into load_kimi_hf_weights.
# ---------------------------------------------------------------------------

def dequant_compressed_tensors_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: CompressedTensorsQuantConfig,
    out_dtype: torch.dtype = torch.bfloat16,
    keep_packed: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Wrap a checkpoint ``(name, tensor)`` stream, dequantizing on the fly.

    For every quantized tensor the checkpoint carries ``<base>.weight_packed`` +
    ``<base>.weight_scale`` (+ ``<base>.weight_zero_point`` for asymmetric); this
    buffers those components per ``<base>`` and, once complete, yields a single
    ``(<base>.weight, bf16 tensor)`` — exactly the key a native-bf16 checkpoint
    would carry — then drops the quant sub-keys. Any key that is not a
    compressed-tensors component (norms, the router ``gate``, ``embed_tokens``,
    ``lm_head``, or a weight the checkpoint left in bf16) passes straight through.

    Buffering is bounded to the in-flight incomplete tensors: a ``<base>`` is
    emitted and freed the moment its required components have all been seen,
    independent of the iterator's key order.

    ``keep_packed``: when it returns True for a ``<base>``, that base's
    compressed-tensors sub-keys are passed through RAW (no buffering, no dequant)
    so a downstream packed-expert loader can route them to int32 params. Kimi
    passes a predicate matching the routed experts (kept packed for in-kernel
    dequant) while every other quantized weight — MLA, dense FFN, shared expert —
    still dequantizes here. Because kept bases never enter ``buffers``, the
    end-of-stream completeness check is unaffected.
    """
    buffers: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in weights:
        suffix = next((s for s in _QUANT_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            yield name, tensor  # not a quant component — pass through untouched
            continue

        base = name[: -len(suffix)]
        if keep_packed is not None and keep_packed(base):
            # Leave this base packed — hand the raw sub-key downstream.
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
