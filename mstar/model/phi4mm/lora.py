"""Phi-4-multimodal's LoRA root, the LLM, from whose modules the bridge builds the spec.

The checkpoint's modality adapters, ``vision`` (rank 256) and ``speech`` (rank 320),
ship as PEFT directories beside the weights (``vision-lora/``, ``speech-lora/``), on
the fused ``qkv_proj`` and ``gate_up_proj``, ``o_proj`` and ``down_proj`` of every
layer. The serving config names them under ``resources: lora: adapters:``, and the
model picks one per request from the prompt's media (``Phi4MMModel.process_prompt``).
The same pool serves a deployment's own adapters on text-only requests, given slots
for them beside these two (``max_slots``).
"""

from __future__ import annotations

from pathlib import Path

from mstar.model.components.lora.modules import PER_SEGMENT, LoraRoot
from mstar.model.phi4mm.config import SPEECH_ADAPTER, VISION_ADAPTER, Phi4MMConfig


def phi4mm_lora_root(config: Phi4MMConfig, local_dir: str) -> LoraRoot:
    from mstar.model.phi4mm.components.language_model import Phi4MiniLM

    return LoraRoot(
        node="LLM", build=lambda group: Phi4MiniLM(config, group),
        rows={"lm_head": PER_SEGMENT}, embeddings=True,
        # the checkpoint's own adapters; a deployment that names its own `adapters:`
        # names these too (`prompt_lora_adapters`, checked at startup)
        adapters={
            name: str(Path(local_dir) / f"{name}-lora" / "adapter_model.safetensors")
            for name in (VISION_ADAPTER, SPEECH_ADAPTER)
        },
    )
