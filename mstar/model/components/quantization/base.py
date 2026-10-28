"""A quantized checkpoint in a model: which format each role is stored in, and which stay quantized.

The checkpoint's ``quantization_config`` picks the method (:func:`quant_method_for`). The model maps
each weight's checkpoint name to a role (``ROLES``), streams the checkpoint through
:meth:`QuantMethod.checkpoint_stream`, and takes its routed experts from
:meth:`QuantMethod.routed_experts`, so one model serves every format a method exists for, and a new
format is a :class:`~.formats.WeightFormat`, not a model change.

``kernels`` is the one switch, the same for every model:

* ``auto``: the fused kernels where the format has them on this device, else (loudly) ``reference``.
* ``fused``: the fused kernels or a load-time error, for a deployment that must capture.
* ``reference``: each weight dequantized per call and multiplied in the activation dtype. Eager
  only; the anchor parity tests compare against.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.quantization.experts import (
    QUANT_KERNELS,
    ExpertMath,
    QuantizedExperts,
    expert_param_rules,
    expert_read_slice,
)
from mstar.model.components.quantization.formats import (
    Fp4E2M1,
    Fp8Block,
    Int4Group,
    WeightFormat,
    is_quantized_dtype,
)
from mstar.model.loader.base import StackedParamRule

# "linear": attention projections and dense MLPs; "routed_experts"; "shared_expert"
ROLES = ("linear", "routed_experts", "shared_expert")


@dataclass(frozen=True)
class QuantMethod:
    """``formats``: the stored format per role (a role absent is plain bf16/fp32 in the checkpoint).
    ``resident``: the roles held quantized in memory; the rest dequantize to bf16 at load."""

    formats: Mapping[str, WeightFormat]
    resident: frozenset[str] = field(default_factory=lambda: frozenset({"routed_experts"}))
    kernels: str = "auto"

    def __post_init__(self) -> None:
        if self.kernels not in QUANT_KERNELS:
            raise ValueError(f"quant kernels {self.kernels!r} is not one of {QUANT_KERNELS}")
        unknown = (set(self.formats) | set(self.resident)) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown quantization roles {sorted(unknown)}; roles are {ROLES}")

    def format_for(self, role: str) -> WeightFormat | None:
        return self.formats.get(role)

    def keeps_quantized(self, role: str) -> bool:
        return role in self.resident and role in self.formats

    def checkpoint_stream(
        self, weights: Iterable[tuple[str, torch.Tensor]], role_of: Callable[[str], str],
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """The checkpoint's tensors as the model loads them. ``role_of(base)`` names the role of a
        quantized weight ``<base>.<part>``. A resident role's parts pass through as stored (aux
        parts such as ``weight_shape`` dropped); any other quantized weight is collected and
        yielded once as a dequantized ``<base>.weight``; plain tensors pass through. A quantized
        tensor no role's format claims is refused rather than loaded as garbage."""
        suffixes = sorted({p.checkpoint for f in self.formats.values() for p in f.parts}
                          | {a for f in self.formats.values() for a in f.aux}, key=len,
                          reverse=True)
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            base = suffix = fmt = None
            for s in suffixes:
                if name.endswith("." + s):
                    base, suffix = name[: -len(s) - 1], s
                    fmt = self.formats.get(role_of(base))
                    break
            if fmt is None or not fmt.claims(suffix, tensor):
                if is_quantized_dtype(tensor.dtype):
                    raise ValueError(
                        f"{name} is stored {tensor.dtype}, but no quantized format of this "
                        f"checkpoint claims it (formats by role: "
                        f"{ {r: f.name for r, f in self.formats.items()} })")
                yield name, tensor
                continue
            if self.keeps_quantized(role_of(base)):
                if suffix not in fmt.aux:
                    yield name, tensor
                continue
            slot = pending.setdefault(base, {})
            slot[suffix] = tensor
            if all(p.checkpoint in slot for p in fmt.parts) and all(a in slot for a in fmt.aux):
                del pending[base]
                yield base + ".weight", _dequantize_checkpoint(fmt, slot, out_dtype)
        if pending:
            missing = {b: sorted(slot) for b, slot in pending.items()}
            raise ValueError(f"checkpoint ended with incomplete quantized weights: {missing}")

    def routed_experts(
        self, num_experts: int, intermediate_size: int, hidden_size: int,
        comm_group: CommGroup | None = None, math: ExpertMath = ExpertMath(),
    ) -> QuantizedExperts:
        """The routed experts in their stored format; ``forward(x, topk_weights, topk_ids)``
        returns this rank's unreduced partial."""
        fmt = self._resident_format("routed_experts")
        return QuantizedExperts(fmt, num_experts, intermediate_size, hidden_size, comm_group,
                                kernels=self.kernels, math=math)

    def expert_rules(self, num_experts: int,
                     source: Callable[[str, int], str]) -> list[StackedParamRule]:
        """Loader rules stacking each expert's checkpoint tensors into :meth:`routed_experts`'
        containers; see :func:`~.experts.expert_param_rules`."""
        return expert_param_rules(self._resident_format("routed_experts"), num_experts, source)

    def expert_read_slice(self, checkpoint_suffix: str, proj: str, intermediate_size: int,
                          tp_rank: int, tp_size: int) -> tuple[int, int, int] | None:
        """This rank's slice of one resident expert tensor for a sliced checkpoint read."""
        return expert_read_slice(self._resident_format("routed_experts"), checkpoint_suffix, proj,
                                 intermediate_size, tp_rank, tp_size)

    def _resident_format(self, role: str) -> WeightFormat:
        if not self.keeps_quantized(role):
            raise ValueError(f"{role} is not held quantized by this method (resident: "
                             f"{sorted(self.resident)}, formats: {sorted(self.formats)})")
        return self.formats[role]


def _dequantize_checkpoint(fmt: WeightFormat, slot: dict[str, torch.Tensor],
                           out_dtype: torch.dtype) -> torch.Tensor:
    parts = {p.container: slot[p.checkpoint] for p in fmt.parts}
    first = fmt.parts[0]
    rows = parts[first.container].shape[0] * first.unit[0]
    cols = parts[first.container].shape[1] * first.unit[1]
    if "weight_shape" in slot:  # the true shape, where packing padded the columns
        rows, cols = (int(v) for v in slot["weight_shape"].tolist())
    return fmt.dequantize(parts, rows, cols, out_dtype)


def quant_method_for(quantization_config: Mapping | None, kernels: str = "auto",
                     resident: Iterable[str] = ("routed_experts",)) -> QuantMethod | None:
    """The method for a checkpoint's HF ``quantization_config`` dict; None for an unquantized
    checkpoint. A format no method serves is refused rather than misread.

    * ``fp8`` with ``weight_block_size``: every role fp8 block-scale (DeepSeek-V3's HF format,
      GLM-5.x). ``scale_fmt: ue8m0`` stores power-of-two scales as ``<base>.scale``, and
      ``expert_dtype: fp4`` stores the routed experts as e2m1 / e8m0 (DeepSeek-V4.1).
    * ``compressed-tensors`` with one ``group`` weight scheme over ``Linear``: ``pack-quantized``
      symmetric int4 (Kimi K2.7) or ``mxfp4-pack-quantized`` (Kimi K3). Its ``ignore`` list keeps
      the other linears out of the checkpoint's quantization, so they load as stored.
    """
    if not quantization_config:
        return None
    method = quantization_config.get("quant_method")
    resident = frozenset(resident)
    if method == "fp8":
        block = tuple(quantization_config.get("weight_block_size") or (128, 128))
        e8m0 = quantization_config.get("scale_fmt") == "ue8m0"
        dense = Fp8Block(block, scale="scale" if e8m0 else "weight_scale_inv", e8m0=e8m0)
        expert_dtype = quantization_config.get("expert_dtype")
        if expert_dtype in (None, "fp8"):
            experts = dense
        elif expert_dtype == "fp4":
            experts = Fp4E2M1(names=("weight", "scale"))
        else:
            raise ValueError(f"fp8 checkpoint expert_dtype {expert_dtype!r} has no format")
        formats = {"linear": dense, "shared_expert": dense, "routed_experts": experts}
        return QuantMethod(formats, resident, kernels)
    if method == "compressed-tensors":
        fmt = _compressed_tensors_format(quantization_config)
        return QuantMethod({role: fmt for role in ROLES}, resident, kernels)
    raise ValueError(f"quantization_config quant_method {method!r} has no quantization method; "
                     "served: fp8, compressed-tensors (int4 pack-quantized, mxfp4)")


def _compressed_tensors_format(config: Mapping) -> WeightFormat:
    groups = config.get("config_groups") or {}
    if len(groups) != 1:
        raise ValueError(f"compressed-tensors with {len(groups)} config groups is not served")
    (group,) = groups.values()
    w = group.get("weights") or {}
    fmt = group.get("format") or config.get("format")
    if group.get("input_activations") is not None:
        raise ValueError("compressed-tensors activation quantization is not served")
    if w.get("strategy") != "group" or not w.get("symmetric", False) or w.get("num_bits") != 4:
        raise ValueError(f"compressed-tensors weights {w} are not a served scheme "
                         "(4-bit symmetric group)")
    if fmt == "pack-quantized" and w.get("type") == "int":
        return Int4Group(int(w["group_size"]))
    if fmt == "mxfp4-pack-quantized" and w.get("type") == "float" and int(w["group_size"]) == 32:
        return Fp4E2M1(names=("weight_packed", "weight_scale"))
    raise ValueError(f"compressed-tensors format {fmt!r} with {w.get('type')} weights is not "
                     "served")


__all__ = ["ROLES", "QuantMethod", "quant_method_for"]
