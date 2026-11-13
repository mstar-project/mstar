"""DeepSeek-V4.1 checkpoint -> module parameters.

The checkpoint names its tensors as its reference does (``layers.N.attn.wq_b.weight`` with a
``.scale`` beside every quantized weight). Dense linears are E4M3 with a power-of-two scale per
32 x 32 block and are dequantized to the model's dtype, which holds them exactly. Routed
experts are packed E2M1 with a scale per row and 32 inputs: kept as those bytes, upcast to E4M3
blocks without loss (refused where a block's row scales are too far apart), or dequantized, as
``moe_experts`` says. The Engram tables stay fp8, and with ``engram_tables: host`` stay in the
checkpoint's files for the engine to map. Tensors of parts not served yet (the DSpark layers,
the vision tower, the image router bias) are skipped by name; anything else unrecognized raises.

A TP rank holds a slice of most tensors (``slice_of``). ``rank_slices`` hands those slices to
the shard reader, so a rank reads its rows of a 98 GB Engram table and not the table; fed whole
tensors, the loader takes the same slices itself.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

import torch

from mstar.model.deepseek_v41.components.linear import COLUMN, RoundedLinear
from mstar.model.deepseek_v41.components.moe import FP4_GROUP
from mstar.model.loader.iterators import SliceSpec, TensorSlice
from mstar.utils.fp8_block import E8M0_DTYPE, FP8_DTYPE
from mstar.utils.mx_round import dequant_block_fp8, dequant_row_fp4, unpack_e2m1

# not served yet: DSpark (``mtp.*``), the vision tower and aligner, the image delimiters and
# the router bias for image tokens
_SKIPPED = re.compile(r"^(mtp\.|vision\.|aligner\.|image_(start|end|newline)$)|\.gate\.bias_vl$")
_EXPERT = re.compile(r"^(layers\.\d+\.ffn)\.(experts\.(\d+)|shared_experts)\.w([123])$")
# E4M3's largest value over E2M1's: an upcast block may scale a row up by 2**6 at most
_MAX_OFFSET_BITS = 6
# the slice of a tensor no parameter of this rank holds
_NOTHING = TensorSlice(0, 0, 0)


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
    ``down_proj``, and its value."""
    suffix: str
    value: torch.Tensor


class _Loader:
    def __init__(self, lm, sliced: bool = False) -> None:
        self.lm, self.config = lm, lm.config
        self.group = lm.comm_group
        self.modules = dict(lm.named_modules())
        self.params = dict(lm.named_parameters())
        # whether the tensors arrive as ``slice_of`` cuts them (read through ``rank_slices``)
        self.sliced = sliced
        self.loaded: set[str] = set()
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        # stacked expert params fill part by part
        self.parts: dict[str, int] = {}

    # -- the part of each checkpoint tensor this rank holds ----------------------------------

    def slice_of(self, name: str) -> TensorSlice | None:
        """The part of checkpoint tensor ``name`` this rank holds, as ``(dim, start, stop)`` in
        the tensor's own elements: None is all of it, and a tensor nothing here holds is empty."""
        if skipped(name):
            return _NOTHING
        stem, _, leaf = name.rpartition(".")
        if stem.endswith(".engram.embed"):
            engram = self.modules["model." + stem.removesuffix(".embed")]
            if engram.host_table is not None:
                return _NOTHING  # the engine maps it from the checkpoint's file
            return self._mine(0, engram.rows)
        expert = _EXPERT.match(stem)
        if expert is not None:
            return self._expert_slice(expert, leaf)
        if name in ("embed.weight", "head.weight"):
            param = "model.embed.weight" if name == "embed.weight" else "lm_head.weight"
            return self._mine(0, self.params[param].shape[0])
        if name.endswith(".attn.attn_sink"):
            return self._mine(0, self.params["model." + name].shape[0])
        module = self.modules.get("model." + stem)
        if isinstance(module, RoundedLinear) and module.shard is not None:
            dim, size = (0, module.out_features) if module.shard == COLUMN else (1, module.in_features)
            # a stored scale covers ``block`` rows and columns: a slice that splits one is
            # left to ``RoundedLinear.local`` once the whole weight is dequantized
            if size % module.block == 0:
                return self._mine(dim, size, module.block if leaf == "scale" else 1)
        return None

    def _mine(self, dim: int, size: int, unit: int = 1) -> TensorSlice | None:
        """This rank's ``size`` of the ranks' equal shares along ``dim``, in elements that each
        cover ``unit`` of them."""
        if self.group.world_size == 1:
            return None
        start = self.group.rank * size
        return TensorSlice(dim, start // unit, (start + size) // unit)

    def _expert_slice(self, match: re.Match, leaf: str) -> TensorSlice | None:
        ffn, _, expert, which = match.groups()
        moe = self.modules["model." + ffn]
        if expert is not None and moe.expert_parallel:  # whole experts, the ones it holds
            return None if 0 <= int(expert) - moe.expert_start < moe.local_experts else _NOTHING
        # a slice of the intermediate dim: gate and up rows, down columns. A packed routed
        # expert stores two columns a byte and a scale per row and FP4_GROUP columns; the
        # shared expert a scale per ``block`` rows and columns.
        if expert is not None:
            rows, cols = 1, (FP4_GROUP if leaf == "scale" else 2)
        else:
            rows = cols = moe.block if leaf == "scale" else 1
        return self._mine(1, moe.inter, cols) if which == "2" else self._mine(0, moe.inter, rows)

    # -- filling the parameters --------------------------------------------------------------

    def _set(self, name: str, value: torch.Tensor) -> None:
        param = self.params[name]
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint gives {tuple(value.shape)}, the module holds {tuple(param.shape)}")
        param.data.copy_(value.to(param.dtype))
        self.loaded.add(name)

    def _pair(self, stem: str, kind: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """A quantized weight and its scale arrive as two tensors, in either order."""
        got = self.pending.setdefault(stem, {})
        got[kind] = tensor
        return self.pending.pop(stem) if len(got) == 2 else None

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if skipped(name):
            return
        if not self.sliced:
            cut = self.slice_of(name)
            if cut is not None:
                stop = min(cut.stop, tensor.shape[cut.dim])
                tensor = tensor.narrow(cut.dim, min(cut.start, stop), max(stop - cut.start, 0))
        stem, _, leaf = name.rpartition(".")
        if leaf == "scale" or (leaf == "weight" and tensor.dtype in (FP8_DTYPE, torch.int8, torch.uint8)):
            pair = self._pair(stem, leaf, tensor)
            if pair is not None:
                self._quantized(stem, pair["weight"], pair["scale"])
            return
        self._plain(name, tensor)

    def _plain(self, name: str, tensor: torch.Tensor) -> None:
        if name == "embed.weight":
            return self._set("model.embed.weight", tensor)
        if name == "head.weight":
            return self._set("lm_head.weight", tensor)
        checkpoint_name, name = name, "model." + name
        hc = re.sub(r"\.hc_(attn|ffn)_(fn|base|scale)$", r".hc_\1.\2", name)
        if hc != name:
            return self._set(hc, tensor)
        if name.endswith(".ffn.gate.bias"):
            return self._set(name.removesuffix("bias") + "e_score_correction_bias", tensor)
        module = self.modules.get(name.removesuffix(".weight"))
        if isinstance(module, RoundedLinear):
            return self._set(name, self._local(module, checkpoint_name, tensor))
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
        self._set(f"model.{stem}.weight", self._local(module, stem + ".weight", dense))

    def _local(self, module: RoundedLinear, name: str, weight: torch.Tensor) -> torch.Tensor:
        """This rank's part of a linear's weight: as it arrived, or cut from the whole weight
        where ``slice_of`` left it whole."""
        return weight if self.slice_of(name) is not None else module.local(weight)

    def _engram_table(self, name: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
        engram = self.modules[name]
        if engram.host_table is not None:
            return  # the engine maps it from the checkpoint's file: no parameter holds it
        # the last rank's rows end with the table, short of its parameter
        tables = (("table", weight.view(torch.uint8), 0), ("table_scale", scale.view(torch.uint8), None))
        for leaf, value, pad in tables:
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
        if routed:  # packed E2M1, a scale per row and 32 inputs
            index = int(expert) - moe.expert_start
            if not 0 <= index < moe.local_experts:
                return  # another rank's expert
            packed, row_scale = weight.view(torch.uint8), scale.view(E8M0_DTYPE)
            if moe.layout == "fp4":
                stored = [_Stored("_packed", packed), _Stored("_scale", scale.view(torch.uint8))]
            elif moe.layout == "fp8":
                value, block_scale = upcast_fp4_block(packed, row_scale, moe.block)
                stored = [_Stored("_fp8", value.view(torch.uint8)), _Stored("_scale_inv", block_scale)]
            else:
                stored = [_Stored("", dequant_row_fp4(packed, row_scale, moe.experts.down_proj.dtype))]
            holder = moe.experts
        else:  # the shared expert is a dense fp8 linear
            value = dequant_block_fp8(weight.view(FP8_DTYPE), scale.view(E8M0_DTYPE), moe.shared.down_proj.dtype)
            stored, holder, index = [_Stored("", value)], moe.shared, None
        proj = "down_proj" if which == "2" else "gate_up_proj"
        prefix = f"model.{ffn}.{'experts' if routed else 'shared'}.{proj}"
        for suffix, value in stored:
            param = getattr(holder, proj + suffix)
            dest = param.data if index is None else param.data[index]
            if which != "2":  # gate (w1) then up (w3): each a half of the stack
                n, half = dest.shape[0] // 2, int(which == "3")
                dest = dest[half * n:(half + 1) * n]
            if tuple(dest.shape) != tuple(value.shape):
                raise ValueError(f"{prefix}{suffix}: checkpoint gives {tuple(value.shape)}, "
                                 f"the module holds {tuple(dest.shape)}")
            dest.copy_(value.to(param.dtype))
            # two halves (gate, up) of every expert's gate_up, one down per expert
            want = (1 if which == "2" else 2) * (moe.local_experts if routed else 1)
            self.parts[prefix + suffix] = self.parts.get(prefix + suffix, 0) + 1
            if self.parts[prefix + suffix] == want:
                self.loaded.add(prefix + suffix)


def rank_slices(lm) -> SliceSpec:
    """For the shard reader (``slice_spec``): the part of each checkpoint tensor ``lm``'s rank
    holds, so that is all it reads. Load what it yields with ``sliced=True``."""
    return _Loader(lm).slice_of


def load_deepseek_v41_weights(lm, weights: Iterable[tuple[str, torch.Tensor]], sliced: bool = False) -> set[str]:
    """Fill ``lm`` (a ``DeepseekV41ForCausalLM``) from ``(name, tensor)`` pairs in the
    checkpoint's names: whole tensors, or with ``sliced`` what ``rank_slices`` cut for this
    rank. Returns the parameters filled, for ``check_loaded``."""
    loader = _Loader(lm, sliced)
    for name, tensor in weights:
        loader.add(name, tensor)
    if loader.pending:
        raise KeyError(f"quantized tensors without their weight or scale: {sorted(loader.pending)[:4]}")
    return loader.loaded
