"""The names a model gives a token2wav node's signals, state and request settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple


class Token2WavKeys(NamedTuple):
    # the streamed speech codes the node reads
    codes: str
    # slot part (``SLOT_PARTS``) -> the Value holding it, a request's conv caches and HiFT tails
    state: Mapping[str, str]
    # the Value of host fields holding a request's voice and where its last window starts
    request: str
    # cache family (``CACHE_FAMILIES``) -> its array
    kv: Mapping[str, str]
    # cache family -> its array's read
    attn: Mapping[str, str]
    # the output edge carrying each window's audio
    audio: str = "audio_chunk"
    # the ``step_metadata`` entry naming a request's voice, one of the node's bundled voices
    voice: str = "voice"
