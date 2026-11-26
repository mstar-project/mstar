"""DeepSeek-V4.1 checkpoint weight loader with .weight/.scale pair buffering and refusal checks."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import torch

from mstar.model.components.deepseek_v41.backbone import DeepSeekV41TextBackbone
from mstar.model.components.deepseek_v41.quant import RoundedLinear, dequant_block_fp8
from mstar.utils.fp8_block import E8M0_DTYPE, FP8_DTYPE

logger = logging.getLogger(__name__)

_SKIPPED_PREFIXES = ("vision.", "aligner.", "mtp.")
_SKIPPED_EXACT = frozenset({"image_start", "image_end", "image_newline"})

_GATE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
_EXPERT_PROJ = re.compile(r"^layers\.(\d+)\.ffn\.(experts\.\d+|shared_experts)\.(w[123])$")
_ENGRAM_WKV = re.compile(r"^layers\.(\d+)\.engram\.wkv$")
_ENGRAM_EMBED = re.compile(r"^layers\.(\d+)\.engram\.embed$")
_ENGRAM_QK = re.compile(r"^layers\.(\d+)\.engram\.(q_weight|k_weight)$")
_HC_PARAM = re.compile(r"^layers\.(\d+)\.hc_(attn|ffn)_(base|fn|scale)$")
_NORM_PARAM = re.compile(r"^layers\.(\d+)\.(attn_norm|ffn_norm)\.weight$")
_ATTN_QUANT = re.compile(r"^layers\.(\d+)\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)$")
_ATTN_PARAM = re.compile(
    r"^layers\.(\d+)\.attn\.(attn_sink|q_norm\.weight|kv_norm\.weight|"
    r"compressor\.(?:norm\.weight|wkv\.weight|wgate\.weight)|"
    r"indexer\.(?:k_norm\.weight|wk\.weight|weights_proj\.weight))$"
)


def is_skipped_checkpoint_key(name: str) -> bool:
    """Return True for Phase-2/3/4 tensors (vision, aligner, image sentinels, mtp)."""
    clean = name.removeprefix("model.")
    return clean in _SKIPPED_EXACT or clean.startswith(_SKIPPED_PREFIXES)


def _copy_checked(param: torch.nn.Parameter, src: torch.Tensor, name: str) -> None:
    if tuple(param.shape) != tuple(src.shape):
        raise ValueError(f"weight {name!r} has shape {tuple(src.shape)}, expected {tuple(param.shape)}")
    param.data.copy_(src.to(device=param.device, dtype=param.dtype))


def load_deepseek_v41_weights(
    backbone: DeepSeekV41TextBackbone,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> set[str]:
    """Load root-level HF DeepSeek-V4.1 weights into ``backbone``.

    Buffers ``.weight``/``.scale`` pairs, routes MoE projections and Engram pairs through
    their dedicated validators, dequantizes FP8 attention linears, skips out-of-scope
    ``vision.*``/``aligner.*``/``image_*``/``mtp.*`` tensors, and refuses missing/unpaired
    or duplicate weights when ``strict=True``.
    """
    loaded: set[str] = set()
    pending: dict[str, dict[str, torch.Tensor]] = {}
    num_layers = len(backbone.layers)

    def mark(param_name: str) -> None:
        if param_name in loaded:
            raise ValueError(f"duplicate DeepSeek-V4.1 weight {param_name!r}")
        loaded.add(param_name)

    def pair_ready(stem: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
        if m := _EXPERT_PROJ.fullmatch(stem):
            layer_id, owner, proj = int(m[1]), m[2], m[3]
            if not 0 <= layer_id < num_layers:
                raise ValueError(f"layer index out of range in {stem!r}")
            moe = backbone.layers[layer_id].ffn
            moe.load_projection_pair(stem, weight, scale)
            if owner.startswith("experts."):
                suffixes = (
                    ("down_proj_fp8", "down_proj_scale_inv")
                    if proj == "w2"
                    else ("gate_up_proj_fp8", "gate_up_proj_scale_inv")
                )
                for suffix in suffixes:
                    loaded.add(f"layers.{layer_id}.ffn.experts.{suffix}")
            else:
                suffix = "down_proj.weight" if proj == "w2" else "gate_up_proj.weight"
                loaded.add(f"layers.{layer_id}.ffn.shared_expert.{suffix}")
            return

        if m := _ENGRAM_WKV.fullmatch(stem):
            lid = m[1]
            if lid not in backbone.engrams:
                raise ValueError(f"unexpected Engram layer in {stem!r}")
            engram = backbone.engrams[lid]
            engram.load_wkv_pair(weight, scale)
            mark(f"engrams.{lid}.wkv.weight")
            mark(f"engrams.{lid}.wkv.weight_scale_inv")
            return

        if m := _ENGRAM_EMBED.fullmatch(stem):
            lid = m[1]
            if lid not in backbone.engrams:
                raise ValueError(f"unexpected Engram layer in {stem!r}")
            embed = backbone.engrams[lid].embed
            embed.weight.weight_loader(embed.weight, weight)
            embed.scale.weight_loader(embed.scale, scale)
            mark(f"engrams.{lid}.embed.weight")
            mark(f"engrams.{lid}.embed.scale")
            return

        if m := _ATTN_QUANT.fullmatch(stem):
            layer_id, attr = int(m[1]), m[2]
            if not 0 <= layer_id < num_layers:
                raise ValueError(f"layer index out of range in {stem!r}")
            attn = backbone.layers[layer_id].attn
            target_mod: RoundedLinear | None = attn
            for part in attr.split("."):
                target_mod = getattr(target_mod, part, None)
            if not isinstance(target_mod, RoundedLinear):
                raise ValueError(f"layer {layer_id} has no attention linear {attr!r}")
            dense = dequant_block_fp8(weight.view(FP8_DTYPE), scale.view(E8M0_DTYPE), target_mod.weight.dtype)
            local = target_mod.local(dense)
            _copy_checked(target_mod.weight, local, f"{stem}.weight")
            mark(f"layers.{layer_id}.attn.{attr}.weight")
            return

        raise ValueError(f"unrecognized quantized DeepSeek-V4.1 tensor stem {stem!r}")

    for raw_name, tensor in weights:
        if is_skipped_checkpoint_key(raw_name):
            logger.debug("Skipping Phase-2/3/4 DeepSeek-V4.1 tensor %s", raw_name)
            continue
        name = raw_name.removeprefix("model.")

        if name.endswith((".weight", ".scale")):
            stem, kind = name.rsplit(".", 1)
            is_quant_stem = bool(
                _EXPERT_PROJ.fullmatch(stem)
                or _ENGRAM_WKV.fullmatch(stem)
                or _ENGRAM_EMBED.fullmatch(stem)
                or _ATTN_QUANT.fullmatch(stem)
            )
            if is_quant_stem:
                entry = pending.setdefault(stem, {})
                if kind in entry:
                    raise ValueError(f"duplicate DeepSeek-V4.1 tensor {name!r}")
                entry[kind] = tensor
                if len(entry) == 2:
                    pair = pending.pop(stem)
                    pair_ready(stem, pair["weight"], pair["scale"])
                continue

        if name == "embed.weight":
            if hasattr(backbone.embed.weight, "weight_loader"):
                backbone.embed.weight.weight_loader(backbone.embed.weight, tensor.to(backbone.embed.weight.dtype))
            else:
                _copy_checked(backbone.embed.weight, tensor, name)
            mark("embed.weight")
        elif name == "head.weight":
            if hasattr(backbone.head.weight, "weight_loader"):
                backbone.head.weight.weight_loader(backbone.head.weight, tensor.to(backbone.head.weight.dtype))
            else:
                _copy_checked(backbone.head.weight, tensor, name)
            mark("head.weight")
        elif name == "norm.weight":
            _copy_checked(backbone.norm.weight, tensor, name)
            mark("norm.weight")
        elif m := _GATE.fullmatch(name):
            layer_id, field = int(m[1]), m[2]
            moe = backbone.layers[layer_id].ffn
            if field == "bias_vl" and moe.gate.bias_vl is None:
                logger.debug("Skipping bias_vl for text-only MoE layer %d", layer_id)
                continue
            moe.load_gate(name, tensor)
            param_name = "e_score_correction_bias" if field == "bias" else field
            mark(f"layers.{layer_id}.ffn.gate.{param_name}")
        elif m := _ENGRAM_QK.fullmatch(name):
            lid, field = m[1], m[2]
            if lid not in backbone.engrams:
                raise ValueError(f"unexpected Engram tensor {name!r}")
            _copy_checked(getattr(backbone.engrams[lid], field), tensor, name)
            mark(f"engrams.{lid}.{field}")
        elif m := _HC_PARAM.fullmatch(name):
            layer_id, site, field = int(m[1]), m[2], m[3]
            hc_mod = getattr(backbone.layers[layer_id], f"{site}_hc")
            _copy_checked(getattr(hc_mod, field), tensor, name)
            mark(f"layers.{layer_id}.{site}_hc.{field}")
        elif m := _NORM_PARAM.fullmatch(name):
            layer_id, norm_name = int(m[1]), m[2]
            _copy_checked(getattr(backbone.layers[layer_id], norm_name).weight, tensor, name)
            mark(f"layers.{layer_id}.{norm_name}.weight")
        elif m := _ATTN_PARAM.fullmatch(name):
            layer_id, subpath = int(m[1]), m[2]
            attn = backbone.layers[layer_id].attn
            if subpath == "attn_sink":
                rank, n_heads = attn.comm_group.rank, attn.num_heads
                local = tensor.narrow(0, rank * n_heads, n_heads) if attn.comm_group.world_size > 1 else tensor
                _copy_checked(attn.attn_sink, local, name)
                mark(f"layers.{layer_id}.attn.attn_sink")
            elif subpath == "indexer.weights_proj.weight":
                if attn.indexer is None:
                    raise ValueError(f"layer {layer_id} has no indexer for {name!r}")
                local = attn.indexer.weights_proj.local(tensor)
                _copy_checked(attn.indexer.weights_proj.weight, local, name)
                mark(f"layers.{layer_id}.attn.indexer.weights_proj.weight")
            else:
                obj: object = attn
                for part in subpath.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        raise ValueError(f"layer {layer_id} has no attribute {subpath!r} for {name!r}")
                assert isinstance(obj, torch.nn.Parameter)
                _copy_checked(obj, tensor, name)
                mark(f"layers.{layer_id}.attn.{subpath}")
        else:
            raise ValueError(f"unrecognized DeepSeek-V4.1 checkpoint tensor {raw_name!r}")

    if pending:
        raise ValueError(f"unpaired quantized DeepSeek-V4.1 tensors (missing weight or scale): {sorted(pending)[:4]}")

    if strict:
        expected = {name for name, _ in backbone.named_parameters()}
        missing = sorted(expected - loaded)
        if missing:
            raise KeyError(f"DeepSeek-V4.1 checkpoint is missing {len(missing)} parameters, e.g. {missing[:4]}")

    target_device = torch.device(device) if device is not None else backbone.embed.weight.device
    backbone.process_weights_after_loading(target_device)
    return loaded
