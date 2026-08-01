"""Zonos2: a multi-codebook autoregressive TTS model.

This package holds the Zonos2 architecture and its TTS serving stack. The stack
has these parts:

* ``prompt`` — the byte tokenizer and the prompt builder.
* ``tts_sampling`` — the multi-codebook sampler.
* ``vocoder`` — the streaming DAC decoder.
* ``zonos2_model`` — the ``Model`` ABC graph-walk wiring.

:class:`Zonos2ForCausalLM` is the transformer core. It maps multi-codebook
frame tokens to per-codebook logits.
"""
from mstar.model.zonos2.components.language_model import Zonos2ForCausalLM
from mstar.model.zonos2.config import Zonos2Config

__all__ = ["Zonos2Config", "Zonos2ForCausalLM"]
