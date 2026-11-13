"""The names a model gives a token2wav node's signals, resources and request settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple


class Token2WavKeys(NamedTuple):
    # the streamed speech codes the node reads
    codes: str
    # the ``RecurrentStateSpec`` holding each request's conv caches and HiFT tails
    state: str
    # cache family (``CACHE_FAMILIES``) -> its ``BoundedKVSpec`` resource key
    kv: Mapping[str, str]
    # the output edge carrying each window's audio
    audio: str = "audio_chunk"
    # the ``step_metadata`` entry naming a request's voice, one of the node's bundled voices
    voice: str = "voice"
