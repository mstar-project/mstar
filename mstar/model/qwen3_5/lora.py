"""Qwen3.5's LoRA roots, from whose modules the bridge builds the spec.

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
  (``ignore``).

Adapters land where the roots' ``checkpoint_rules()`` load the checkpoint. Every
embedding call is in the step (a vision prompt's ids, media pads included, are embedded
in the forward, ``SampledTokenDecoder._forward``, and the media rows spliced in after),
so the token embedding is an ordinary target.
"""

from __future__ import annotations

from mstar.model.components.lora.modules import PER_SEGMENT, AdapterRoot, LoraRoot
from mstar.model.loader.base import StackedParamRule
from mstar.model.qwen3_5.config import Qwen3_5Config, Qwen3_5VisionConfig

_TEXT = "model.language_model."
# the released checkpoint, the inner multimodal model, the text-only causal LM, the
# bare text model
TEXT_ROOTS = tuple(AdapterRoot(prefix, _TEXT) for prefix in (_TEXT, "language_model.", "model.", ""))
VISION_ROOTS = (AdapterRoot("model.visual.", "model.visual."), AdapterRoot("visual.", "model.visual."))

# Tinker trains the delta-net's q, k and v in-projections apart
SPLIT_QKV_RULES = tuple(
    StackedParamRule(".in_proj_fused.", f".in_proj_{shard}.", shard) for shard in ("q", "k", "v")
)


def qwen3_5_lora_roots(
    config: Qwen3_5Config, vision_config: Qwen3_5VisionConfig | None = None,
) -> tuple[LoraRoot, ...]:
    """The LLM's root, and the vision tower's when there is one, each node with a pool
    of its own. The tower is not tensor-parallel."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    llm = LoraRoot(
        node="LLM", build=lambda _: Qwen3_5ForCausalLM(config),
        adapter_stacked_params=SPLIT_QKV_RULES,
        adapter_roots=TEXT_ROOTS,
        # Tinker's name for the head
        aliases={f"{_TEXT}unembed_tokens": "lm_head"},
        rows={"lm_head": PER_SEGMENT},
        ignore=("mtp.*", "model.mtp.*"),
        embeddings=True,
    )
    if vision_config is None:
        return (llm,)
    from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
    return llm, LoraRoot(
        node="vision_encoder", build=lambda _: Qwen3_5VisionModel(vision_config),
        adapter_roots=VISION_ROOTS,
        rows={"merger.*": vision_config.merge_unit},
    )
