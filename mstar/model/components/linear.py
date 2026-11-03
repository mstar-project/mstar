"""Non-distributed fused linear projections, and linears on block-scaled low precision.

``FusedColumnLinear`` concatenates several output shards into a single
weight (and optional bias) so a transformer's QKV or gate/up projections
run as one GEMM instead of several. Each shard is loaded independently
via a ``weight_loader`` so HF checkpoints that keep the shards as
separate tensors (``q``/``k``/``v``, ``gate``/``up``) load straight into
the fused parameter.

This mirrors the role of the TP-aware ``QKVParallelLinear`` /
``MergedColumnParallelLinear`` in ``model.components.distributed`` but
without any tensor-parallel sharding — use it for models that fuse
projections but don't need TP.

``BlockScaledLinear`` runs a linear through cuBLASLt's block-scaled GEMM: weights quantized once
at load, the activation per call. A model opts in by declaring which of its linears may run that way
(``LowPrecisionLinears``, applied by ``apply_linear_precision``). The output differs from the bf16
GEMM's by the format's rounding, so it is opt-in, checked against a calibrated quality bar.
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.lora.config import LoraTarget
from mstar.model.components.lora.capable import LoraCapable, LoraOutputAdd
from mstar.platforms import current_platform
from mstar.platforms.block_gemm import block_quant, block_scaled_gemm
from mstar.utils.fp8_block import BLOCK_FORMATS, BlockFormat, BlockQuant, check_block_gemm, quantize_block

logger = logging.getLogger(__name__)

LINEAR_PRECISIONS = ("bf16", *BLOCK_FORMATS)


class FusedColumnLinear(nn.Module, LoraCapable):
    """Linear whose output is the concatenation of several shards along
    dim 0, fused into a single weight (and optional bias).

    Args:
        input_size: input feature dim.
        shard_sizes: maps a shard id to its output size. Shards are laid
            out along dim 0 in iteration order, so the order of this dict
            defines the layout (and the split used at forward time).
        bias: whether to include a (fused) bias.

    The fused ``weight`` / ``bias`` carry a ``weight_loader(param, tensor,
    shard_id)`` method that copies one checkpoint shard into its slice of
    the fused parameter, dispatched by ``shard_id`` (a key of
    ``shard_sizes``). Wire the per-shard checkpoint keys to it with the
    loader's stacked-param rules.
    """

    # bound: the delta whole on every rank, like the fused linear
    lora_twin = LoraOutputAdd

    def __init__(
        self,
        input_size: int,
        shard_sizes: dict[str | int, int],
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._shard_sizes = dict(shard_sizes)
        self._shard_offsets: dict[str | int, int] = {}
        offset = 0
        for shard_id, size in self._shard_sizes.items():
            self._shard_offsets[shard_id] = offset
            offset += size
        self.output_size = offset
        self.weight = nn.Parameter(torch.empty(offset, input_size, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(offset, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # ``to_empty`` / ``.to(...)`` reallocate the Parameters and drop the
        # attached ``weight_loader``; re-attach to the new objects.
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None = None,
    ) -> None:
        assert loaded_shard_id in self._shard_sizes, (
            f"FusedColumnLinear got unknown shard id {loaded_shard_id!r}; "
            f"expected one of {list(self._shard_sizes)}"
        )
        offset = self._shard_offsets[loaded_shard_id]
        size = self._shard_sizes[loaded_shard_id]
        dst = param.data.narrow(0, offset, size)
        assert dst.shape == loaded_weight.shape, (
            f"shard {loaded_shard_id!r} shape mismatch: dst {tuple(dst.shape)} "
            f"vs weight {tuple(loaded_weight.shape)}"
        )
        dst.copy_(loaded_weight)

    def shard_slice(self, shard_id: str | int) -> tuple[int, int]:
        """``(offset, size)`` of one shard's rows in the fused parameter, for callers that
        update a shard in place (e.g. merging a LoRA delta into ``q`` only)."""
        if shard_id not in self._shard_sizes:
            raise KeyError(f"unknown shard id {shard_id!r}; expected one of {list(self._shard_sizes)}")
        return self._shard_offsets[shard_id], self._shard_sizes[shard_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def lora_target(self) -> LoraTarget:
        # a slice per shard, named by the shard ids the stacked rules route by
        return LoraTarget(
            in_features=self.weight.shape[1],
            out_slices=tuple(self._shard_sizes.values()),
            parallel="none",
            slice_names=tuple(self._shard_sizes),
        )


class BlockScaledLinear(nn.Module):
    """A bias-free linear whose weight is block-scaled low precision. Input of any rank, ``[..., K]`` contiguous;
    output bf16, as the linear's was. Replaces the module that held the bf16 weight, so the weight's bytes go."""

    def __init__(self, weight: torch.Tensor, fmt: BlockFormat):
        super().__init__()
        self.fmt = fmt
        self.out_features, self.in_features = weight.shape
        check_block_gemm(fmt, self.in_features, self.out_features, weight.dtype)
        quantized = quantize_block(weight, fmt)
        self.register_buffer("weight_data", quantized.data)
        self.register_buffer("weight_scale", quantized.scale)
        self.register_buffer("weight_tensor_scale", quantized.tensor_scale)
        # an activation's NVFP4 block scales are in units of one (``quant_block``)
        unit = None if quantized.tensor_scale is None else torch.ones_like(quantized.tensor_scale)
        self.register_buffer("x_tensor_scale", unit)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A graph break: the quantizer is one anyway, and its resume frame, shared by every instance at every batch
        # size, would run into Dynamo's recompile limit. Capture records the same kernels either way.
        lead = x.shape[:-1]
        weight = BlockQuant(self.weight_data, self.weight_scale, self.weight_tensor_scale)
        quantized = block_quant(x.reshape(-1, self.in_features), self.fmt)._replace(tensor_scale=self.x_tensor_scale)
        return block_scaled_gemm(quantized, weight, self.fmt).reshape(*lead, self.out_features)


@dataclass(frozen=True)
class LowPrecisionLinears:
    """A model's declaration over its linears' names (``fnmatch`` patterns): the ones that may run low precision, and
    the ones that stay bf16. Every linear has to be in one or the other, so a new layer can't change format
    silently; ``keep_bf16`` wins where both match."""

    candidates: tuple[str, ...]
    keep_bf16: tuple[str, ...]

    def selected(self, name: str, keep: tuple[str, ...] = ()) -> bool:
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in (*self.keep_bf16, *keep)):
            return False
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.candidates):
            return True
        raise ValueError(f"linear {name!r} is neither a low-precision candidate nor kept bf16")


def apply_linear_precision(
    module: nn.Module, precision: str, declaration: LowPrecisionLinears, keep: tuple[str, ...] = (),
) -> list[str]:
    """Move the declared linears of ``module``, weights loaded, to ``precision``; ``keep`` adds patterns to the
    declaration's bf16 list for one deployment. Returns the names moved. ``bf16``, or a GPU without the block-scaled
    GEMM, changes nothing."""
    if precision not in LINEAR_PRECISIONS:
        raise ValueError(f"linear_precision must be one of {LINEAR_PRECISIONS}, got {precision!r}")
    if precision == "bf16":
        return []
    fmt = BLOCK_FORMATS[precision]
    linears = [(name, m) for name, m in module.named_modules() if isinstance(m, (nn.Linear, FusedColumnLinear))]
    selected = [(name, m) for name, m in linears if declaration.selected(name, keep)]
    if not selected:
        return []
    if not current_platform.has_block_scaled_gemm(selected[0][1].weight.device):
        logger.warning("linear_precision %s needs a block-scaled GEMM, not on this GPU; staying on bf16", precision)
        return []
    block_quant.resolve()
    block_scaled_gemm.resolve()
    for name, linear in selected:
        if linear.bias is not None:
            raise ValueError(f"{precision} replaces bias-free linears; {name!r} has a bias")
        parent_name, _, leaf = name.rpartition(".")
        module.get_submodule(parent_name).register_module(leaf, BlockScaledLinear(linear.weight.data, fmt))
    logger.info("linear_precision %s: %d of %d linears", precision, len(selected), len(linears))
    return [name for name, _ in selected]
