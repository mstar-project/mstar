"""Weight loader for the Ling-2.0 thinker.

Maps the released ``inclusionAI/Ming-flash-omni-2.0`` checkpoint's key
namespace into :class:`mminf.model.ming_omni_flash.components.model.LingMoeModel`'s
``state_dict`` and runs the per-expert fusion that packs 256 separate
``gate_proj`` / ``up_proj`` / ``down_proj`` weights into the dense
``experts.gate_up_proj`` and ``experts.down_proj`` tensors that mminf's
fused-MoE kernel expects.

Step-3c scope: thinker only, no KV cache, no engine glue. The submodule
wrapping that exposes this to ``mminf-serve`` is step 3d.

## Key mapping

The released checkpoint stores the LLM weights under ``model.model.*``
(the outer ``model.`` is the multimodal wrapper, the inner ``model.``
is HF's convention for ``BailingMoeV2ForCausalLM.model``). Translation
to my :class:`LingMoeModel` state dict::

    model.lm_head.weight                                 →  lm_head.weight
    model.model.word_embeddings.weight                   →  embed_tokens.weight
    model.model.norm.weight                              →  norm.weight
    model.model.layers.{i}.input_layernorm.weight        →  layers.{i}.input_layernorm.weight
    model.model.layers.{i}.post_attention_layernorm.w    →  layers.{i}.post_attention_layernorm.weight
    model.model.layers.{i}.attention.query_key_value.w   →  layers.{i}.self_attn.qkv_proj.weight
    model.model.layers.{i}.attention.dense.weight        →  layers.{i}.self_attn.dense.weight
    model.model.layers.{i}.attention.q_norm.weight       →  layers.{i}.self_attn.q_norm.weight
    model.model.layers.{i}.attention.k_norm.weight       →  layers.{i}.self_attn.k_norm.weight
    # dense layer 0 (first_k_dense_replace=1)
    model.model.layers.0.mlp.{gate,up,down}_proj.w       →  layers.0.mlp.{gate,up,down}_proj.weight
    # MoE layers 1..31 (router weights nest through LingMoeRouter's inner nn.Linear)
    model.model.layers.{i}.mlp.{gate,image_gate,audio_gate}.weight        →  layers.{i}.mlp.{...}.gate.weight
    model.model.layers.{i}.mlp.{gate,image_gate,audio_gate}.expert_bias   →  layers.{i}.mlp.{...}.expert_bias
    model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight  ─┐
    model.model.layers.{i}.mlp.experts.{j}.up_proj.weight    ─┴─→  layers.{i}.mlp.experts.gate_up_proj
    model.model.layers.{i}.mlp.experts.{j}.down_proj.weight                →  layers.{i}.mlp.experts.down_proj
    model.model.layers.{i}.mlp.shared_experts.{g,u,d}_proj.weight          →  layers.{i}.mlp.shared_expert.{...}.weight

The expert-fusion (the last 3 lines above) uses the same
``MergeModulelist`` + ``Concatenate`` :class:`Operation`s that
Qwen3-Omni already relies on
(:mod:`mminf.model.qwen3_omni.qwen3_omni_model`).
"""

from __future__ import annotations

import logging
import re

import torch

from mminf.model.loader.iterators import iter_safetensors_shards
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.utils import (
    KeysAndConverter,
    Operation,
    WeightConverter,
    _apply_operations,
)

logger = logging.getLogger(__name__)


# Outermost prefix on the checkpoint — strip before applying renames.
_CKPT_THINKER_PREFIX = "model."


def build_ling_weight_converters() -> list[WeightConverter]:
    """Per-expert fusion converters for the MoE layers.

    These run AFTER the key-rename pass; the source_patterns are matched
    against the post-rename keys (which preserve the ``mlp.experts.N.*``
    structure from the checkpoint — only the layer-level prefix changes).
    """
    return [
        WeightConverter(
            source_patterns=[
                "mlp.experts.*.gate_proj.weight",
                "mlp.experts.*.up_proj.weight",
            ],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[
                Operation("MergeModulelist", dim=0),  # 256 → (256, inter, hidden)
                Operation("Concatenate", dim=1),       # 2 of (256, inter, hidden) → (256, 2*inter, hidden)
            ],
        ),
        WeightConverter(
            source_patterns=["mlp.experts.*.down_proj.weight"],
            target_patterns="mlp.experts.down_proj",
            operations=[
                Operation("MergeModulelist", dim=0),  # 256 → (256, hidden, inter)
            ],
        ),
    ]


# Per-key rename rules, applied AFTER the ``model.`` outer-prefix strip.
# Order matters: longer matches first so e.g. ``attention.query_key_value``
# isn't half-rewritten by a shorter pattern.
_RENAME_RULES: list[tuple[str, str]] = [
    # Top-level
    ("model.word_embeddings.weight", "embed_tokens.weight"),
    ("model.norm.weight", "norm.weight"),
    # The ``model.lm_head.weight`` key has no ``model.model.*`` prefix in
    # the checkpoint, so after stripping the outer ``model.`` it's just
    # ``lm_head.weight`` — no rename needed.

    # Attention (per layer) — substring replacement so it works for any
    # layer index.
    ("model.layers.{}.attention.query_key_value.weight",
     "layers.{}.self_attn.qkv_proj.weight"),
    ("model.layers.{}.attention.dense.weight",
     "layers.{}.self_attn.dense.weight"),
    ("model.layers.{}.attention.q_norm.weight",
     "layers.{}.self_attn.q_norm.weight"),
    ("model.layers.{}.attention.k_norm.weight",
     "layers.{}.self_attn.k_norm.weight"),

    # Norms (per layer) — strip outer model.
    ("model.layers.{}.input_layernorm.weight",
     "layers.{}.input_layernorm.weight"),
    ("model.layers.{}.post_attention_layernorm.weight",
     "layers.{}.post_attention_layernorm.weight"),

    # MoE routers — checkpoint has ``mlp.gate.weight`` directly; mine has
    # ``mlp.gate.gate.weight`` because LingMoeRouter wraps an nn.Linear.
    # Same for image_gate / audio_gate.
    ("model.layers.{}.mlp.gate.weight",
     "layers.{}.mlp.gate.gate.weight"),
    ("model.layers.{}.mlp.gate.expert_bias",
     "layers.{}.mlp.gate.expert_bias"),
    ("model.layers.{}.mlp.image_gate.weight",
     "layers.{}.mlp.image_gate.gate.weight"),
    ("model.layers.{}.mlp.image_gate.expert_bias",
     "layers.{}.mlp.image_gate.expert_bias"),
    ("model.layers.{}.mlp.audio_gate.weight",
     "layers.{}.mlp.audio_gate.gate.weight"),
    ("model.layers.{}.mlp.audio_gate.expert_bias",
     "layers.{}.mlp.audio_gate.expert_bias"),

    # MoE experts (per-expert per-layer) — preserve the ``mlp.experts.N.*``
    # structure for the WeightConverter to match later.
    ("model.layers.{}.mlp.experts.{}.gate_proj.weight",
     "layers.{}.mlp.experts.{}.gate_proj.weight"),
    ("model.layers.{}.mlp.experts.{}.up_proj.weight",
     "layers.{}.mlp.experts.{}.up_proj.weight"),
    ("model.layers.{}.mlp.experts.{}.down_proj.weight",
     "layers.{}.mlp.experts.{}.down_proj.weight"),

    # MoE shared expert (singular in mminf vs plural in ckpt).
    ("model.layers.{}.mlp.shared_experts.gate_proj.weight",
     "layers.{}.mlp.shared_expert.gate_proj.weight"),
    ("model.layers.{}.mlp.shared_experts.up_proj.weight",
     "layers.{}.mlp.shared_expert.up_proj.weight"),
    ("model.layers.{}.mlp.shared_experts.down_proj.weight",
     "layers.{}.mlp.shared_expert.down_proj.weight"),

    # Dense layer-0 MLP — no rename, just strip the outer model.
    ("model.layers.{}.mlp.gate_proj.weight",
     "layers.{}.mlp.gate_proj.weight"),
    ("model.layers.{}.mlp.up_proj.weight",
     "layers.{}.mlp.up_proj.weight"),
    ("model.layers.{}.mlp.down_proj.weight",
     "layers.{}.mlp.down_proj.weight"),
]


def _compile_rename_rules() -> list[tuple[re.Pattern, str]]:
    """Compile the ``{}``-style rule patterns into regex + format strings.

    Each ``{}`` becomes a numeric capture group; the replacement uses
    ``\1``, ``\2``, ... in declaration order.
    """
    compiled: list[tuple[re.Pattern, str]] = []
    for src, tgt in _RENAME_RULES:
        # Anchor with ^ ... $ so we match the full key, not a substring
        # (avoids accidentally matching nested ``mlp.experts.*.gate_proj``
        # via the dense-MLP rule).
        src_regex = "^" + re.escape(src).replace(r"\{\}", r"(\d+)") + "$"
        # Replacement template: convert each ``\{\}`` (literal) in tgt
        # to a ``\1``, ``\2``, ... backreference.
        n_groups = src.count("{}")
        tgt_template = tgt
        for i in range(n_groups):
            tgt_template = tgt_template.replace("{}", f"\\{i + 1}", 1)
        compiled.append((re.compile(src_regex), tgt_template))
    return compiled


def _rename_key(key: str, compiled: list[tuple[re.Pattern, str]]) -> str | None:
    """Apply rename rules to a single (already-prefix-stripped) ckpt key.

    Returns the renamed key, or ``None`` if no rule matches (caller
    decides whether to raise or skip).
    """
    for regex, template in compiled:
        m = regex.match(key)
        if m:
            return regex.sub(template, key)
    return None


def load_thinker_weights(
    model: LingMoeModel,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> None:
    """Load Ling-2.0 thinker weights from a local snapshot dir into ``model``.

    Args:
        model: an instantiated :class:`LingMoeModel` (constructor sets
            up empty params; this fills them).
        local_dir: path to the HF snapshot (containing
            ``model.safetensors.index.json`` and shards).
        device: where to materialise the tensors (``"cpu"`` / ``"cuda"``
            / ``"cuda:N"``).
        strict: if True, raise when the model has parameters with no
            matching checkpoint keys (after the per-layer index drops
            keys for layers beyond ``model.num_hidden_layers``).
            Default True — silent param holes produce garbage outputs.
    """
    compiled = _compile_rename_rules()
    # Pre-build the set of param keys the *model* expects; anything not
    # in this set (after renaming) gets silently skipped (saves memory
    # when loading e.g. a 1-layer subset of a 32-layer checkpoint).
    target_keys = set(model.state_dict().keys())
    # For the fused experts, the target key after the converter is e.g.
    # ``layers.1.mlp.experts.gate_up_proj`` — that's already in
    # ``target_keys``. The pre-fusion per-expert keys (``...experts.5.gate_proj.weight``)
    # are NOT in target_keys; they're collected separately for the
    # converter to consume.

    # Two buckets:
    #   - per_key_state: directly-loadable tensors keyed by the final
    #     target name.
    #   - per_layer_expert_keys: nested dict
    #     {layer_idx: {sub_pattern: {target_param_name: {expert_key_path: tensor}}}}
    #     where sub_pattern is one of the WeightConverter patterns.
    per_key_state: dict[str, torch.Tensor] = {}
    # For each layer, collect expert tensors so we can run the converters
    # once per layer at the end.
    per_layer_expert: dict[int, dict[str, torch.Tensor]] = {}

    converters = build_ling_weight_converters()
    # Compile expert-key matchers so we know which keys to route to the
    # per-layer expert bucket (vs the direct per-key state).
    # A renamed expert key looks like ``layers.{i}.mlp.experts.{j}.gate_proj.weight``.
    expert_key_re = re.compile(
        r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )

    unmatched_ckpt_keys: list[str] = []

    for raw_key, tensor in iter_safetensors_shards(
        local_dir, device=device, prefix=_CKPT_THINKER_PREFIX,
    ):
        # 1. Strip the outermost ``model.`` (everything starts with it).
        if not raw_key.startswith(_CKPT_THINKER_PREFIX):
            continue
        stripped = raw_key[len(_CKPT_THINKER_PREFIX):]

        # 2. The bare ``lm_head.weight`` survives the strip and lands
        #    straight at the right name — no renaming needed.
        if stripped in target_keys:
            per_key_state[stripped] = tensor
            continue

        # 3. Try the rename rules.
        renamed = _rename_key(stripped, compiled)
        if renamed is None:
            unmatched_ckpt_keys.append(raw_key)
            continue

        # 4. If this is a per-expert pre-fusion key, bucket it for the
        #    converter; otherwise it's a direct load.
        m = expert_key_re.match(renamed)
        if m:
            layer_idx = int(m.group(1))
            # Filter early: only keep keys for layers the model actually has.
            if layer_idx >= model.num_hidden_layers:
                continue
            per_layer_expert.setdefault(layer_idx, {})[renamed] = tensor
        else:
            # Filter directly-loadable per-layer keys for in-range layers too.
            m_layer = re.match(r"^layers\.(\d+)\.", renamed)
            if m_layer and int(m_layer.group(1)) >= model.num_hidden_layers:
                continue
            if renamed in target_keys:
                per_key_state[renamed] = tensor
            elif renamed.startswith("layers."):
                # In-range layer but our model variant doesn't have this
                # specific module (e.g. a dense-MLP-only test loads a
                # MoE layer's gate weight). Silently skip.
                continue
            else:
                unmatched_ckpt_keys.append(raw_key)

    # Apply expert-fusion converters per layer.
    for layer_idx, expert_kvs in per_layer_expert.items():
        for conv in converters:
            target_key = f"layers.{layer_idx}.{conv.target_patterns}"
            if target_key not in target_keys:
                continue
            # Filter the per-expert keys to just the ones this converter's
            # source patterns can match (each converter wants the right
            # subset).
            kac = KeysAndConverter(converter=conv)
            matched_kvs: dict[str, torch.Tensor] = {}
            for pat in conv.source_patterns:
                pat_regex = re.compile(
                    r"^layers\." + str(layer_idx) + r"\." +
                    re.escape(pat).replace(r"\*", r"\d+") + "$"
                )
                for k, v in expert_kvs.items():
                    if pat_regex.match(k):
                        matched_kvs[k] = v
                        kac.append_key(k)
            if not matched_kvs:
                # Converter target exists in the model but no source keys
                # found in the checkpoint — strict mode treats this as
                # missing-param territory; non-strict skips.
                continue
            per_key_state[target_key] = _apply_operations(matched_kvs, conv)

    # Finally, load into the model.
    missing_keys = sorted(target_keys - set(per_key_state.keys()))
    if missing_keys and strict:
        raise KeyError(
            f"Missing thinker parameters after load (strict=True). "
            f"Sample missing keys: {missing_keys[:10]} "
            f"(total {len(missing_keys)})"
        )
    if unmatched_ckpt_keys and strict:
        raise KeyError(
            f"{len(unmatched_ckpt_keys)} checkpoint keys had no rename "
            f"rule and were not directly loadable. "
            f"Sample: {unmatched_ckpt_keys[:10]}"
        )

    _, unexpected = model.load_state_dict(per_key_state, strict=False, assign=True)
    if unexpected and strict:
        raise KeyError(
            f"load_state_dict reported unexpected keys (shouldn't happen "
            f"after our filtering): {unexpected[:10]}"
        )
    logger.info(
        "Loaded %d thinker params into LingMoeModel(num_hidden_layers=%d) from %s",
        len(per_key_state), model.num_hidden_layers, local_dir,
    )
