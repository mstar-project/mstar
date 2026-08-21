"""FP8 block-scale helpers for the GLM-5.2 checkpoint.

The checkpoint stores most weights as float8_e4m3fn with per-[128, 128]-block
fp32 scales under ``<base>.weight_scale_inv`` (DeepSeek-V3 layout: dequant is
``fp8 * scale_inv``). ``modules_to_not_convert`` names the bf16 remainder
(embeddings, norms, router gates, lm_head), which shows up in the stream as
plain ``.weight`` keys with no scale sibling.

Load-time policy (see ``weight_loader.py``): everything dequantizes to bf16
except routed experts, which must stay FP8-resident — bf16 experts alone are
~181 GB/rank at TP8, over the H200's 141 GB. Resident expert bytes live in
uint8 containers (ints dodge the module-wide autocast the same way Kimi's
packed int32 weights do) and are re-viewed as e4m3 at dispatch time.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch

_WEIGHT = ".weight"
_SCALE_INV = ".weight_scale_inv"

FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class Fp8BlockQuantConfig:
    """Subset of the HF fp8 quantization_config used by GLM-5.2."""

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
    """Dequantize one 2-D fp8 weight: ``w[i, j] * scale_inv[i//bo, j//bi]``.

    ``weight_fp8`` may be e4m3 or its uint8 byte view. Scales broadcast per
    block and are cropped to the weight shape, so ragged tail blocks are fine.
    """
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
    """Pair fp8 ``.weight`` keys with ``.weight_scale_inv`` and dequantize.

    Unlike compressed-tensors, quantized and bf16 weights share the plain
    ``.weight`` suffix — an fp8 dtype is what marks a tensor as needing its
    scale sibling. Bases matching ``keep_fp8`` pass both keys through raw
    (the FP8-resident expert path). Order-independent; raises at stream end
    if any fp8 weight never met its scale (or vice versa).
    """
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
