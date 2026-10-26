"""Qwen3.5's LoRA spec, declared from its modules (``module_lora_spec``).

The LLM node takes every linear of the text stack, as PEFT's ``all-linear`` picks
them, plus the LM head and the token embedding; the vision encoder node takes the
ViT's linears, its merger and its patch embed. Targets, slices, layers and placements
come from the classes the stacks are built of: the fused ``qkv_proj`` (the output gate
inside each query head, so a ``q_proj`` adapter covers it), the delta-net's
``[q|k|v|z|a|pad|b]`` in-projection, the parallel linears' sharding. What this file
states is what the modules cannot:

* the LM head reads one row per request, and the vision merger one per folded block of
  patches (``rows``);
* the layouts adapters on the Hub write the towers under (``adapter_roots``), Tinker's
  split delta-net ``in_proj_q/k/v`` and its ``unembed_tokens`` head
  (``adapter_stacked_params``, ``aliases``), and the MTP head, which is never run
  (``ignore``);
* the token embedding's delta is the node's to add (``deferred``): a vision prompt's
  text is embedded in ``prepare_inputs``, before the step, so ``preprocess`` and
  ``Qwen3_5LanguageModel.embed`` add it on the step's rows;
* the vision targets' names, kept as the shipped serving configs spell them (``names``).

The loader's rules are passed explicitly (the Qwen3.5 roots state no
``checkpoint_rules()`` yet); they are the same objects ``weight_loader`` loads with.
"""

from __future__ import annotations

from mstar.engine.resources.lora import LoraSpec
from mstar.model.components.lora.modules import (
    PER_SEGMENT,
    AdapterRoot,
    join_lora_specs,
    module_lora_spec,
)
from mstar.model.loader.base import CheckpointRules, StackedParamRule
from mstar.model.qwen3_5.config import Qwen3_5Config, Qwen3_5VisionConfig
from mstar.model.qwen3_5.weight_loader import (
    _STACKED_PARAMS,
    qwen3_5_name_remapper,
    qwen3_5_vision_name_remapper,
)

_TEXT = "model.language_model."
# the released checkpoint, the inner multimodal model, the text-only causal LM, the
# bare text model
TEXT_ROOTS = tuple(AdapterRoot(prefix, _TEXT) for prefix in (_TEXT, "language_model.", "model.", ""))
VISION_ROOTS = (AdapterRoot("model.visual.", "model.visual."), AdapterRoot("visual.", "model.visual."))

# Tinker trains the delta-net's q, k and v in-projections apart
SPLIT_QKV_RULES = tuple(
    StackedParamRule(".in_proj_fused.", f".in_proj_{shard}.", shard) for shard in ("q", "k", "v")
)

VISION_NAMES = {
    "patch_embed.proj": "patch_embed",
    "attn.qkv": "attn_qkv",
    "attn.proj": "attn_proj",
    "mlp.linear_fc1": "mlp_fc1",
    "mlp.linear_fc2": "mlp_fc2",
    "merger.linear_fc1": "merger_fc1",
    "merger.linear_fc2": "merger_fc2",
}


def qwen3_5_llm_lora_spec(config: Qwen3_5Config) -> LoraSpec:
    """The LLM node's part (``qwen3_5_lora_spec`` joins it to the vision tower's)."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    return module_lora_spec(
        lambda _: Qwen3_5ForCausalLM(config),
        nodes={"LLM"}, root_path="model",
        rules=CheckpointRules(stacked_params=tuple(_STACKED_PARAMS), remap=qwen3_5_name_remapper),
        adapter_stacked_params=SPLIT_QKV_RULES,
        adapter_roots=TEXT_ROOTS,
        # Tinker's name for the head
        aliases={f"{_TEXT}unembed_tokens": "lm_head"},
        rows={"lm_head": PER_SEGMENT},
        ignore=("mtp.*", "model.mtp.*"),
        embeddings=True,
        # a vision prompt's text is embedded in prepare_inputs, before the step: the
        # node adds the delta itself, on the step's rows (preprocess, and `embed`)
        deferred=("model.embed_tokens",),
    )


def qwen3_5_vision_lora_spec(config: Qwen3_5VisionConfig) -> LoraSpec:
    """The vision encoder node's spec; the tower is not tensor-parallel."""
    from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
    return module_lora_spec(
        lambda _: Qwen3_5VisionModel(config),
        nodes={"vision_encoder"}, root_path="model",
        rules=CheckpointRules(remap=qwen3_5_vision_name_remapper),
        adapter_roots=VISION_ROOTS,
        rows={"merger.*": config.merge_unit},
        names=VISION_NAMES,
    )


def qwen3_5_lora_spec(
    config: Qwen3_5Config, vision_config: Qwen3_5VisionConfig | None = None,
) -> LoraSpec:
    """The model's one spec: the LLM's targets, joined to the vision tower's when there
    is one, each node with a pool of its own."""
    llm = qwen3_5_llm_lora_spec(config)
    if vision_config is None:
        return llm
    return join_lora_specs(llm, qwen3_5_vision_lora_spec(vision_config))
