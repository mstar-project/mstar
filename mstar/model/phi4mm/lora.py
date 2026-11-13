"""Phi-4-multimodal's LoRA spec, declared from the LLM's modules (``module_lora_spec``).

The checkpoint's modality adapters, ``vision`` (rank 256) and ``speech`` (rank 320),
ship as PEFT directories beside the weights (``vision-lora/``, ``speech-lora/``), on
the fused ``qkv_proj`` and ``gate_up_proj``, ``o_proj`` and ``down_proj`` of every
layer. The serving config names them under ``resources: lora: adapters:``, and the
model picks one per request from the prompt's media (``Phi4MMModel.process_prompt``).
The same pool serves a deployment's own adapters on text-only requests, given slots
for them beside these two (``max_slots``).
"""

from __future__ import annotations

from mstar.engine.resources.lora import LoraSpec
from mstar.model.components.lora.modules import PER_SEGMENT, module_lora_spec
from mstar.model.phi4mm.config import Phi4MMConfig


def phi4mm_lora_spec(config: Phi4MMConfig) -> LoraSpec:
    from mstar.model.phi4mm.components.language_model import Phi4MiniLM

    return module_lora_spec(
        lambda group: Phi4MiniLM(config, group),
        nodes={"LLM"}, root_path="model",
        rows={"lm_head": PER_SEGMENT},
        embeddings=True,
    )
