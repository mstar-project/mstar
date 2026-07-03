"""Zonos2: multi-codebook autoregressive TTS transformer.

This package implements the Zonos2 model architecture (see
``ARCHITECTURE.md`` and ``../ZONOS2/docs/tts_architecture.md``) on top of
the reusable ``mstar.model.components`` building blocks.

The scope here is the transformer network itself — :class:`Zonos2ForCausalLM`
maps multi-codebook frame tokens to per-codebook logits. The surrounding
TTS serving stack (byte tokenizer / prompt builder, TTS sampler, DAC
vocoder, and the ``Model``-ABC graph-walk wiring) is intentionally out of
scope for this module.
"""
from mstar.model.zonos2.config import Zonos2Config
from mstar.model.zonos2.components.language_model import Zonos2ForCausalLM

__all__ = ["Zonos2Config", "Zonos2ForCausalLM"]
