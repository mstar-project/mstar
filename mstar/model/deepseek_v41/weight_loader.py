"""DeepSeek-V4.1 checkpoint -> module parameters.

The checkpoint names its tensors as its reference does (``layers.N.attn.wq_b.weight`` with a
``.scale`` beside every quantized weight). Dense linears are E4M3 with a power-of-two scale per
32 x 32 block and are dequantized to the model's dtype, which holds them exactly. Routed
experts are packed E2M1 with a scale per row and 32 inputs: kept as those bytes, upcast to E4M3
blocks without loss (refused where a block's row scales are too far apart), or dequantized, as
``moe_experts`` says. The Engram tables stay fp8. Every tensor is sliced to this TP rank's
shard here. Tensors of parts not served yet (the DSpark layers, the vision tower, the image
router bias) are skipped by name; anything else unrecognized raises.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

import torch

from mstar.model.deepseek_v41.components.linear import RoundedLinear
from mstar.model.deepseek_v41.components.moe import FP4_GROUP
from mstar.utils.fp8_block import E8M0_DTYPE, FP8_DTYPE
from mstar.utils.mx_round import dequant_block_fp8, dequant_row_fp4, unpack_e2m1

# not served yet: DSpark (``mtp.*``), the vision tower and aligner, the image delimiters and
# the router bias for image tokens
_SKIPPED = re.compile(r"^(mtp\.|vision\.|aligner\.|image_(start|end|newline)$)|\.gate\.bias_vl$")
_EXPERT = re.compile(r"^(layers\.\d+\.ffn)\.(experts\.(\d+)|shared_experts)\.w([123])$")
# E4M3's largest value over E2M1's: an upcast block may scale a row up by 2**6 at most
_MAX_OFFSET_BITS = 6


def skipped(name: str) -> bool:
    return _SKIPPED.search(name) is not None


def upcast_fp4_block(packed: torch.Tensor, scale: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A packed E2M1 ``[out, in // 2]`` weight with a power-of-two scale per row and ``block``
    inputs, as E4M3 ``[out, in]`` with one fp32 scale per ``block`` x ``block`` block. Exact:
    each block takes its largest row scale over ``2 ** 6`` and the rows shift up to meet it.
    Raises where a row's scale is so far below its block's largest that E4M3 cannot hold it."""
    values = unpack_e2m1(packed)
    out_dim, in_dim = values.shape
    rows = scale.float().view(out_dim // block, block, in_dim // block)
    block_scale = rows.amax(dim=1) / 2 ** _MAX_OFFSET_BITS
    offset = (rows / block_scale.unsqueeze(1)).repeat_interleave(block, dim=-1).view(out_dim, in_dim)
    upcast = (values * offset).to(FP8_DTYPE)
    # (a layout check on meta tensors has no values to compare)
    if values.device.type != "meta" and not torch.equal(upcast.float(), values * offset):
        raise ValueError("an fp4 expert block's row scales are too far apart to upcast to fp8 "
                         "exactly; serve it with moe_experts: fp4 or dense")
    return upcast, block_scale


class _Stored(NamedTuple):
    """One tensor an expert matrix is held as: the parameter's suffix after ``gate_up_proj`` /
    ``down_proj``, and how many of the matrix's rows and columns one element of it covers."""
    suffix: str
    value: torch.Tensor
    rows: int = 1
    cols: int = 1


class _Loader:
    def __init__(self, lm) -> None:
        self.lm, self.config = lm, lm.config
        self.group = lm.comm_group
        self.modules = dict(lm.named_modules())
        self.params = dict(lm.named_parameters())
        self.loaded: set[str] = set()
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        # stacked expert params fill part by part
        self.parts: dict[str, int] = {}

    def _set(self, name: str, value: torch.Tensor) -> None:
        param = self.params[name]
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint gives {tuple(value.shape)}, the module holds {tuple(param.shape)}")
        param.data.copy_(value.to(param.dtype))
        self.loaded.add(name)

    def _rank_rows(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        size = value.shape[dim] // self.group.world_size
        return value.narrow(dim, self.group.rank * size, size)

    def _pair(self, stem: str, kind: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """A quantized weight and its scale arrive as two tensors, in either order."""
        got = self.pending.setdefault(stem, {})
        got[kind] = tensor
        return self.pending.pop(stem) if len(got) == 2 else None

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if skipped(name):
            return
        stem, _, leaf = name.rpartition(".")
        if leaf == "scale" or (leaf == "weight" and tensor.dtype in (FP8_DTYPE, torch.int8, torch.uint8)):
            pair = self._pair(stem, leaf, tensor)
            if pair is not None:
                self._quantized(stem, pair["weight"], pair["scale"])
            return
        self._plain(name, tensor)

    def _plain(self, name: str, tensor: torch.Tensor) -> None:
        if name == "embed.weight":
            return self._set("model.embed.weight", self._rank_rows(tensor))
        if name == "head.weight":
            return self._set("lm_head.weight", self._rank_rows(tensor))
        name = "model." + name
        hc = re.sub(r"\.hc_(attn|ffn)_(fn|base|scale)$", r".hc_\1.\2", name)
        if hc != name:
            return self._set(hc, tensor)
        if name.endswith(".attn.attn_sink"):
            return self._set(name, self._rank_rows(tensor))
        if name.endswith(".ffn.gate.bias"):
            return self._set(name.removesuffix("bias") + "e_score_correction_bias", tensor)
        module = self.modules.get(name.removesuffix(".weight"))
        if isinstance(module, RoundedLinear):
            return self._set(name, module.local(tensor))
        if name not in self.params:
            raise KeyError(f"checkpoint tensor {name.removeprefix('model.')!r} has no parameter in this model")
        self._set(name, tensor)

    def _quantized(self, stem: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
        expert = _EXPERT.match(stem)
        if expert is not None:
            return self._expert(expert, weight, scale)
        if stem.endswith(".engram.embed"):
            return self._engram_table("model." + stem.removesuffix(".embed"), weight, scale)
        module = self.modules.get("model." + stem)
        if not isinstance(module, RoundedLinear):
            raise KeyError(f"checkpoint tensor {stem!r} has no quantized linear in this model")
        dense = dequant_block_fp8(weight.view(FP8_DTYPE), scale.view(E8M0_DTYPE), module.weight.dtype)
        self._set(f"model.{stem}.weight", module.local(dense))

    def _engram_table(self, name: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
        engram = self.modules[name]
        weight, scale = weight.view(torch.uint8), scale.view(torch.uint8)
        rows = slice(engram.row_start, engram.row_start + engram.rows)
        for leaf, value, pad in (("table", weight[rows], 0), ("table_scale", scale[rows], None)):
            param = self.params[f"{name}.{leaf}"]
            if pad is None:
                # a padding row past the table's end scales by one: E8M0's byte for 2 ** 0
                pad = int(torch.ones(1).to(E8M0_DTYPE).view(torch.uint8))
            param.data.fill_(pad)
            param.data[: value.shape[0]].copy_(value)
            self.loaded.add(f"{name}.{leaf}")

    def _expert(self, match: re.Match, weight: torch.Tensor, scale: torch.Tensor) -> None:
        ffn, _, expert, which = match.groups()
        moe = self.modules["model." + ffn]
        routed = expert is not None
        inter, rank = moe.inter, self.group.rank
        if routed:  # packed E2M1, a scale per row and 32 inputs
            packed, row_scale = weight.view(torch.uint8), scale.view(E8M0_DTYPE)
            if moe.layout == "fp4":
                stored = [_Stored("_packed", packed, cols=2),
                          _Stored("_scale", scale.view(torch.uint8), cols=FP4_GROUP)]
            elif moe.layout == "fp8":
                value, block_scale = upcast_fp4_block(packed, row_scale, moe.block)
                stored = [_Stored("_fp8", value.view(torch.uint8)),
                          _Stored("_scale_inv", block_scale, rows=moe.block, cols=moe.block)]
            else:
                stored = [_Stored("", dequant_row_fp4(packed, row_scale, moe.experts.down_proj.dtype))]
            holder, index = moe.experts, int(expert)
        else:  # the shared expert is a dense fp8 linear
            value = dequant_block_fp8(weight.view(FP8_DTYPE), scale.view(E8M0_DTYPE), moe.shared.down_proj.dtype)
            stored, holder, index = [_Stored("", value)], moe.shared, None
        proj = "down_proj" if which == "2" else "gate_up_proj"
        prefix = f"model.{ffn}.{'experts' if routed else 'shared'}.{proj}"
        for suffix, value, rows, cols in stored:
            param = getattr(holder, proj + suffix)
            dest = param.data if index is None else param.data[index]
            if which == "2":  # down: this rank's columns
                n = inter // cols
                value = value[:, rank * n:(rank + 1) * n]
            else:  # gate (w1) then up (w3): this rank's rows, into that half of the stack
                n, half = inter // rows, int(which == "3")
                value, dest = value[rank * n:(rank + 1) * n], dest[half * n:(half + 1) * n]
            if tuple(dest.shape) != tuple(value.shape):
                raise ValueError(f"{prefix}{suffix}: checkpoint gives {tuple(value.shape)}, "
                                 f"the module holds {tuple(dest.shape)}")
            dest.copy_(value.to(param.dtype))
            # two halves (gate, up) of every expert's gate_up, one down per expert
            want = (1 if which == "2" else 2) * (moe.num_experts if routed else 1)
            self.parts[prefix + suffix] = self.parts.get(prefix + suffix, 0) + 1
            if self.parts[prefix + suffix] == want:
                self.loaded.add(prefix + suffix)


def load_deepseek_v41_weights(lm, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Fill ``lm`` (a ``DeepseekV41ForCausalLM``) from ``(name, tensor)`` pairs in the
    checkpoint's names. Returns the parameters filled, for ``check_loaded``."""
    loader = _Loader(lm)
    for name, tensor in weights:
        loader.add(name, tensor)
    if loader.pending:
        raise KeyError(f"quantized tensors without their weight or scale: {sorted(loader.pending)[:4]}")
    return loader.loaded
