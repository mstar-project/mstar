"""Zonos2: multi-codebook autoregressive TTS.

Implements the Zonos2 model architecture, together with the surrounding
TTS serving stack: the byte tokenizer / prompt builder (``prompt``), the
multi-codebook sampler (``tts_sampling``), the streaming DAC vocoder
(``vocoder``), and the ``Model``-ABC graph-walk wiring (``zonos2_model``).

:class:`Zonos2ForCausalLM` is the transformer core — it maps multi-codebook
frame tokens to per-codebook logits.
"""
from mstar.model.zonos2.config import Zonos2Config
from mstar.model.zonos2.components.language_model import Zonos2ForCausalLM

__all__ = ["Zonos2Config", "Zonos2ForCausalLM"]
