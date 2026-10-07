"""HF GLM-5.3-Flash checkpoint loading for the glm5_next module tree."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.quantization import dequant_fp8_block_stream
from mstar.model.loader.base import StackedParamRule

if TYPE_CHECKING:
    from mstar.model.glm5_next.quantization import Fp8BlockQuantConfig

logger = logging.getLogger(__name__)

_TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

# Keep the expert index attached while remapping both fp8 and bf16 sub-keys.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale_inv)$"
)

_EXPERT_BASE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")

# Flat layer-level mHC params -> per-site submodules.
_HC_RE = re.compile(r"\.hc_(attn|ffn)_(fn|base|scale)$")

# KDA ForgetGate params live FLAT under self_attn. in the checkpoint but
# the module (mirroring the HF class structure) nests them under
# forget_gate. — the four suffixes exist only on KDA layers, so the
# rewrite is unambiguous (kda spec: "map names accordingly").
_KDA_FORGET_RE = re.compile(
    r"\.self_attn\.(f_a_proj\.weight|f_b_proj\.weight|dt_bias|A_log)$"
)

# KDA tensors head-sharded across TP (Glm5NextKdaAttention): the read plan
# slices these so each rank reads only its head block. Row-sharded by
# heads x head_dim, by heads, or column-sharded (o_proj's contraction dim).
_KDA_ROW_QKV_RE = re.compile(
    r"\.self_attn\.(q_proj|k_proj|v_proj|f_b_proj|g_b_proj|q_conv1d|k_conv1d|v_conv1d)\.weight$"
)
_KDA_ROW_QKV_FLAT_RE = re.compile(r"\.self_attn\.dt_bias$")
_KDA_ROW_HEADS_RE = re.compile(r"\.self_attn\.(A_log|b_proj\.weight)$")
_KDA_COL_RE = re.compile(r"\.self_attn\.o_proj\.weight$")

# Mirrors glm52.components.mtp.remap_mtp_key / MTP_GLUE_PREFIXES (same four
# DeepSeek-V3 names). Defined locally because that module drags in the
# engine stack (cache_manager -> zmq) and this loader must stay importable
# on any machine — the same reason the KDA/mHC math files must.
MTP_GLUE_PREFIXES = ("enorm", "hnorm", "eh_proj", "shared_head")


def _remap_mtp_sub_key(sub_key: str) -> str:
    """``model.language_model.layers.45.<sub_key>`` -> MTP state-dict key."""
    if sub_key.startswith(MTP_GLUE_PREFIXES):
        return sub_key
    return f"transformer_layer.{sub_key}"


def _is_routed_expert_base(base: str) -> bool:
    return _EXPERT_BASE_RE.search(base) is not None


def glm5_next_name_remapper(name: str) -> str | None:
    if name.startswith(_TEXT_PREFIX):
        name = "model." + name[len(_TEXT_PREFIX):]
    name = _HC_RE.sub(r".\1_hc.\2", name)
    name = _KDA_FORGET_RE.sub(r".self_attn.forget_gate.\1", name)
    name = name.replace(".shared_experts.", ".shared_expert.")
    m = _EXPERT_RE.match(name)
    if m:
        prefix, expert_idx, proj, suffix = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.{suffix}"
    return name


def _make_glm5_next_name_remapper(num_hidden_layers: int, load_mtp: bool):
    """Trunk remapping, plus layer-45 routing onto the ``mtp.`` submodule:
    strip the layer prefix, place glue keys direct and the rest under
    ``transformer_layer.``, then the trunk naming conventions — the
    expert/shared-expert rewrites are prefix-agnostic, so the fused
    stacked-param rules apply to the MTP MoE unchanged.
    """
    if not load_mtp:
        return glm5_next_name_remapper

    def remap(name: str) -> str | None:
        m = _LAYER_RE.match(name)
        if m and int(m.group(1)) >= num_hidden_layers:
            return glm5_next_name_remapper(
                "mtp." + _remap_mtp_sub_key(name[m.end():])
            )
        return glm5_next_name_remapper(name)

    return remap


def skip_vision_and_mtp_keys(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_hidden_layers: int,
    load_mtp: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Drop vision-tower keys — and, unless ``load_mtp``, MTP-layer keys —
    before any dequant buffering, counting every drop.
    """
    skipped_vision = 0
    skipped_mtp = 0
    for name, tensor in weights:
        if name.startswith(VISION_PREFIX):
            skipped_vision += 1
            continue
        m = _LAYER_RE.match(name)
        if not load_mtp and m and int(m.group(1)) >= num_hidden_layers:
            skipped_mtp += 1
            continue
        yield name, tensor
    if skipped_vision:
        logger.info(
            "GLM-5.3 load: skipped %d vision-tower keys (glm5_next_text is "
            "text-only).", skipped_vision,
        )
    if skipped_mtp:
        logger.info("GLM-5.3 load: skipped %d MTP-layer keys.", skipped_mtp)


def build_glm5_next_stacked_params(
    n_routed_experts: int, fp8_experts: bool = False,
) -> list[StackedParamRule]:
    rules: list[StackedParamRule] = []
    for i in range(n_routed_experts):
        if fp8_experts:
            # scale_inv rules MUST precede weight rules: matching is
            # first-win substring, and ".weight" is a prefix of
            # ".weight_scale_inv".
            for proj, sid in (("gate_proj", f"gate:{i}"), ("up_proj", f"up:{i}")):
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_scale_inv",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight_scale_inv",
                    shard_id=sid,
                ))
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_fp8",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight",
                    shard_id=sid,
                ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_scale_inv",
                source_suffix=f".experts.down_proj.__expert{i}__.weight_scale_inv",
                shard_id=f"down:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_fp8",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
        else:
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.gate_proj.__expert{i}__.weight",
                shard_id=f"gate:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.up_proj.__expert{i}__.weight",
                shard_id=f"up:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
    # KDA fused depthwise conv: three [qkv_dim, 1, kernel] checkpoint convs
    # row-concat into one [3*qkv_dim, 1, kernel] parameter in (q, k, v)
    rules.append(StackedParamRule(".conv1d.weight", ".q_conv1d.weight", "q"))
    rules.append(StackedParamRule(".conv1d.weight", ".k_conv1d.weight", "k"))
    rules.append(StackedParamRule(".conv1d.weight", ".v_conv1d.weight", "v"))
    # Dense/shared gate-up rules must follow expert rules because matching
    # is first-win.
    rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
    rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
    return rules


# Parameter-name suffixes the checkpoint stores in fp32 and that must stay
# fp32 after ``model.to(autocast_dtype)`` narrows every float param to bf16.
FP32_PARAM_SUFFIXES = (
    "e_score_correction_bias",  # DeepSeek-V3 router selection bias (glm52 set)
    "_scale_inv",               # fp8 block scales (glm52 set)
    ".dt_bias",                 # KDA, HF _keep_in_fp32_modules_strict
    ".A_log",                   # KDA per-head decay, same
    "conv1d.weight",            # KDA short conv, same
    "_hc.base",                 # mHC per-site bias (attn_hc/ffn_hc): the
    "_hc.scale",                # checkpoint ships these fp32 and the Sinkhorn
                                # reads them at fp32 — matches ``fn``'s fp32 cache
)


def restore_fp32_params(module: nn.Module) -> None:
    """Re-widen params the checkpoint stores fp32 before loading into them."""
    for name, param in module.named_parameters():
        if name.endswith(FP32_PARAM_SUFFIXES) and param.dtype != torch.float32:
            param.data = param.data.float()


def build_glm5_next_read_plan(
    checkpoint_keys: Iterable[str],
    config: Glm5NextModelConfig,
    tp_rank: int,
    tp_size: int,
    load_mtp: bool = False,
) -> tuple[set[str], dict[str, tuple[int, int, int]]]:
    """Keys-to-read + per-key slice specs for the TP fast read path."""
    fp8_experts = config.quantization_config is not None and config.moe_fp8_resident
    shard_inter = config.moe_intermediate_size // tp_size
    keys: set[str] = set()
    specs: dict[str, tuple[int, int, int]] = {}
    kda_layers = set(config.kda_layer_indices)
    heads_per_rank = config.linear_num_heads // tp_size
    qkv_per_rank = heads_per_rank * config.linear_head_dim
    if fp8_experts:
        bo, bi = config.quantization_config.weight_block_size
        assert shard_inter % bo == 0 and shard_inter % bi == 0, (
            f"per-rank intermediate {shard_inter} must be a multiple of the "
            f"scale block ({bo}, {bi}) for sliced reads"
        )

    for key in checkpoint_keys:
        if key.startswith(VISION_PREFIX):
            continue  # text-only package: never read, never transfer
        m_layer = _LAYER_RE.match(key)
        layer = int(m_layer.group(1)) if m_layer else None
        if (
            layer is not None
            and layer >= config.num_hidden_layers
            and not load_mtp
        ):
            continue  # MTP module off: never read, never transfer
        keys.add(key)

        if tp_size > 1 and layer in kda_layers:
            if _KDA_ROW_QKV_RE.search(key) or _KDA_ROW_QKV_FLAT_RE.search(key):
                specs[key] = (0, tp_rank * qkv_per_rank, (tp_rank + 1) * qkv_per_rank)
            elif _KDA_ROW_HEADS_RE.search(key):
                specs[key] = (0, tp_rank * heads_per_rank, (tp_rank + 1) * heads_per_rank)
            elif _KDA_COL_RE.search(key):
                specs[key] = (1, tp_rank * qkv_per_rank, (tp_rank + 1) * qkv_per_rank)

        if not fp8_experts:
            continue
        m = _EXPERT_RE.match(key)
        if not m:
            continue
        _, _, proj, suffix = m.groups()
        is_scale = suffix == "weight_scale_inv"
        if proj in ("gate_proj", "up_proj"):
            unit = bo if is_scale else 1
            rows = shard_inter // unit
            specs[key] = (0, tp_rank * rows, (tp_rank + 1) * rows)
        else:  # down_proj: contraction dim is sharded -> column slice
            unit = bi if is_scale else 1
            cols = shard_inter // unit
            specs[key] = (1, tp_rank * cols, (tp_rank + 1) * cols)

    return keys, specs


def load_glm5_next_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    quant_config: Fp8BlockQuantConfig | None = None,
    fp8_experts: bool = False,
    num_hidden_layers: int = 45,
    load_mtp: bool = False,
) -> set[str]:
    from mstar.model.loader import load_hf_weights

    weights = skip_vision_and_mtp_keys(
        weights, num_hidden_layers, load_mtp=load_mtp,
    )
    if quant_config is not None:
        keep = _is_routed_expert_base if fp8_experts else None
        weights = dequant_fp8_block_stream(weights, quant_config, keep_fp8=keep)
    elif fp8_experts:
        raise ValueError("fp8_experts=True requires a quant_config")

    restore_fp32_params(module)
    return load_hf_weights(
        module,
        weights,
        stacked_params=build_glm5_next_stacked_params(
            n_routed_experts, fp8_experts=fp8_experts,
        ),
        name_remapper=_make_glm5_next_name_remapper(num_hidden_layers, load_mtp),
    )


def load_weights(
    module: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)


# ---------------------------------------------------------------------------
# Index cross-check: a name-level dry run of the exact load pipeline.

_SCALE_SUFFIX = ".weight_scale_inv"


def resolve_index_names(
    index_names: Iterable[str],
    config: Glm5NextModelConfig,
    load_mtp: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Classify every checkpoint name through the real pipeline stages."""
    # The real dispatcher's rule matcher — not a reimplementation.
    from mstar.model.loader.base import _apply_stacked

    names = list(index_names)
    name_set = set(names)
    fp8_experts = (
        config.quantization_config is not None and config.moe_fp8_resident
    )
    scale_bases = {
        n[: -len(_SCALE_SUFFIX)] for n in names if n.endswith(_SCALE_SUFFIX)
    }
    if scale_bases and config.quantization_config is None:
        raise ValueError(
            "index has weight_scale_inv tensors but the config carries no "
            "quantization_config — the fp8 stream would never pair them"
        )
    remapper = _make_glm5_next_name_remapper(config.num_hidden_layers, load_mtp)
    rules = build_glm5_next_stacked_params(
        config.n_routed_experts, fp8_experts=fp8_experts,
    )

    out: dict[str, list[str]] = {
        "skip_vision": [], "skip_mtp": [],
        "loaded": [], "absorbed": [], "unmapped": [],
    }
    for raw in names:
        # Stage 1: skip_vision_and_mtp_keys, name-level.
        if raw.startswith(VISION_PREFIX):
            out["skip_vision"].append(raw)
            continue
        m = _LAYER_RE.match(raw)
        if not load_mtp and m and int(m.group(1)) >= config.num_hidden_layers:
            out["skip_mtp"].append(raw)
            continue

        # Stage 2: dequant_fp8_block_stream, name-level. A non-resident
        # scale is consumed into its .weight sibling (one emitted tensor
        # per pair); resident routed-expert pairs pass through raw.
        effective = raw
        absorbed = False
        if raw.endswith(_SCALE_SUFFIX):
            base = raw[: -len(_SCALE_SUFFIX)]
            if not (fp8_experts and _is_routed_expert_base(base)):
                if base + ".weight" not in name_set:
                    out["unmapped"].append(f"{raw} (orphan weight_scale_inv)")
                    continue
                effective = base + ".weight"
                absorbed = True

        # Stages 3-4: remapper, then first-win stacked-rule dispatch.
        mapped = remapper(effective)
        if mapped is None:
            out["unmapped"].append(f"{raw} (remapper dropped)")
            continue
        target, _shard = _apply_stacked(mapped, rules)
        out["absorbed" if absorbed else "loaded"].append(f"{raw} -> {target}")

    return {kind: tuple(v) for kind, v in out.items()}


def _norm_and_hc_targets(prefix: str, with_hc: bool) -> dict[str, int]:
    targets = {
        f"{prefix}input_layernorm.weight": 1,
        f"{prefix}post_attention_layernorm.weight": 1,
    }
    if with_hc:
        for site in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                targets[f"{prefix}{site}.{p}"] = 1
    return targets


def _kda_attention_targets(prefix: str) -> dict[str, int]:
    p = f"{prefix}self_attn."
    targets = {
        f"{p}{name}": 1
        for name in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight",
            "b_proj.weight", "forget_gate.f_a_proj.weight",
            "forget_gate.f_b_proj.weight", "forget_gate.dt_bias",
            "forget_gate.A_log", "g_a_proj.weight", "g_b_proj.weight",
            "o_norm.weight", "o_proj.weight",
        )
    }
    targets[f"{p}conv1d.weight"] = 3  # q/k/v row-concat shards
    return targets


def _mla_attention_targets(prefix: str) -> dict[str, int]:
    # The indexer targets are unconditional: Glm5NextMLAAttention always
    # builds its indexer, so the expected tree always feeds it (an
    # indexer-less load would be 84 silently-uninitialized parameters).
    p = f"{prefix}self_attn."
    names = [
        "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
        "kv_b_proj.weight", "o_proj.weight",
        "indexer.wq_b.weight", "indexer.wk.weight",
        "indexer.weights_proj.weight", "indexer.k_norm.weight",
        "indexer.k_norm.bias", "indexer.index_kpool_compress_ape",
        "indexer.index_kpool_compress_gate",
    ]
    return {f"{p}{name}": 1 for name in names}


def _mlp_targets(prefix: str, dense: bool, n_experts: int,
                 fp8_experts: bool) -> dict[str, int]:
    p = f"{prefix}mlp."
    if dense:
        return {f"{p}gate_up_proj.weight": 2, f"{p}down_proj.weight": 1}
    targets = {
        f"{p}gate.weight": 1,
        f"{p}gate.e_score_correction_bias": 1,
        f"{p}shared_expert.gate_up_proj.weight": 2,
        f"{p}shared_expert.down_proj.weight": 1,
    }
    if fp8_experts:
        targets[f"{p}experts.gate_up_proj_fp8"] = 2 * n_experts
        targets[f"{p}experts.gate_up_proj_scale_inv"] = 2 * n_experts
        targets[f"{p}experts.down_proj_fp8"] = n_experts
        targets[f"{p}experts.down_proj_scale_inv"] = n_experts
    else:
        targets[f"{p}experts.gate_up_proj"] = 2 * n_experts
        targets[f"{p}experts.down_proj"] = n_experts
    return targets


def expected_parameter_paths(
    config: Glm5NextModelConfig,
    load_mtp: bool = True,
) -> dict[str, int]:
    """The module tree the schedule implies: parameter path -> the number
    of checkpoint tensors that must land in it (stacked shards count each;
    dequant-absorbed scale siblings do not)."""
    fp8_experts = (
        config.quantization_config is not None and config.moe_fp8_resident
    )
    expected: dict[str, int] = {
        "model.embed_tokens.weight": 1,
        "model.norm.weight": 1,
        "lm_head.weight": 1,
    }
    for idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{idx}."
        expected.update(_norm_and_hc_targets(prefix, with_hc=config.mhc))
        if config.is_full_attention_layer(idx):
            expected.update(_mla_attention_targets(prefix))
        else:
            expected.update(_kda_attention_targets(prefix))
        expected.update(_mlp_targets(
            prefix, dense=config.is_dense_mlp_layer(idx),
            n_experts=config.n_routed_experts, fp8_experts=fp8_experts,
        ))
    if load_mtp and config.num_nextn_predict_layers > 0:
        for glue in ("enorm.weight", "hnorm.weight", "eh_proj.weight",
                     "shared_head.norm.weight"):
            expected[f"mtp.{glue}"] = 1
        # One plain-residual (no mHC) full-attention MoE decoder layer.
        prefix = "mtp.transformer_layer."
        expected.update(_norm_and_hc_targets(prefix, with_hc=False))
        expected.update(_mla_attention_targets(prefix))
        expected.update(_mlp_targets(
            prefix, dense=False,
            n_experts=config.n_routed_experts, fp8_experts=fp8_experts,
        ))
    return expected


def cross_check_index(
    index_path: str | Path,
    config: Glm5NextModelConfig | None = None,
    load_mtp: bool = True,
    max_examples: int = 10,
) -> bool:
    """Dry-run the load pipeline over the safetensors index and diff it
    against the expected module tree. Prints a report; returns pass/fail.
    """
    import json
    from collections import Counter

    index_path = Path(index_path)
    if index_path.is_dir():
        index_path = index_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    index_names = list(index["weight_map"])

    if config is None:
        with open(index_path.parent / "config.json") as f:
            config = Glm5NextModelConfig.from_hf_config(json.load(f))

    resolved = resolve_index_names(index_names, config, load_mtp=load_mtp)
    loaded_counts: Counter[str] = Counter(
        entry.rsplit(" -> ", 1)[1] for entry in resolved["loaded"]
    )
    absorbed_targets = {
        entry.rsplit(" -> ", 1)[1] for entry in resolved["absorbed"]
    }
    expected = expected_parameter_paths(config, load_mtp=load_mtp)

    unexpected = sorted(set(loaded_counts) - set(expected))
    missing = sorted(set(expected) - set(loaded_counts))
    stray_absorbed = sorted(absorbed_targets - set(expected))
    count_mismatches = sorted(
        f"{target}: got {loaded_counts[target]} shards, want {want}"
        for target, want in expected.items()
        if target in loaded_counts and loaded_counts[target] != want
    )
    non_vision_skips = list(resolved["skip_mtp"])

    n_vision = len(resolved["skip_vision"])
    n_mtp_keys = sum(
        1 for n in index_names
        if (m := _LAYER_RE.match(n)) and int(m.group(1)) >= config.num_hidden_layers
    )
    n_trunk = len(index_names) - n_vision - n_mtp_keys - 1  # - lm_head

    def show(label: str, items: list[str]) -> None:
        print(f"  {label}: {len(items)}")
        for item in items[:max_examples]:
            print(f"    {item}")
        if len(items) > max_examples:
            print(f"    ... and {len(items) - max_examples} more")

    print(f"glm5_next weight-map cross-check: {index_path}")
    print(
        f"  config: {config.num_hidden_layers} layers "
        f"({len(config.kda_layer_indices)} KDA / "
        f"{len(config.full_attn_layer_indices)} MLA+DSA), "
        f"{config.n_routed_experts}+{config.n_shared_experts} experts, "
        f"fp8={config.quantization_config is not None} "
        f"(experts resident={config.moe_fp8_resident}), "
        f"load_mtp={load_mtp}"
    )
    print(
        f"  index: {len(index_names)} tensors = {n_trunk} text trunk "
        f"+ {n_mtp_keys} MTP layer-{config.num_hidden_layers} "
        f"+ {n_vision} vision + 1 lm_head"
    )
    print(
        f"  mapped: {len(resolved['loaded'])} tensors -> "
        f"{len(loaded_counts)} parameters "
        f"(+ {len(resolved['absorbed'])} fp8 scales dequant-absorbed into "
        f"their .weight)"
    )
    skip_note = (
        "OK (vision-only)" if not non_vision_skips
        else f"NOT vision-only ({len(non_vision_skips)} flag-skipped)"
    )
    print(
        f"  skipped: {n_vision + len(non_vision_skips)} "
        f"(vision {n_vision}, mtp {len(resolved['skip_mtp'])}) — "
        f"skip list {skip_note}"
    )
    show("unmapped checkpoint keys", list(resolved["unmapped"]))
    show("unexpected targets (mapped but not in the module tree)", unexpected)
    show("missing targets (module tree got no tensor)", missing)
    show("absorbed scales with no expected .weight target", stray_absorbed)
    show("shard-count mismatches", count_mismatches)

    ok = not (
        resolved["unmapped"] or unexpected or missing
        or stray_absorbed or count_mismatches
    )
    print("PASS" if ok else "FAIL")
    return ok


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-check the glm5_next weight mapping against a "
        "GLM-5.3-Flash model.safetensors.index.json",
    )
    parser.add_argument(
        "checkpoint",
        help="checkpoint directory (or the index file itself); config.json "
        "must sit next to the index",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="account layer 45 as skipped (mtp_num_draft_tokens == 0 load)",
    )
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    ok = cross_check_index(
        args.checkpoint,
        load_mtp=not args.no_mtp,
        max_examples=args.examples,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
