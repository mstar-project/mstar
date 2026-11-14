"""The names a model gives a token2wav node's signals, resources and request settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple


class Token2WavKeys(NamedTuple):
    # the streamed speech codes the node reads
    codes: str
    # the stem of each request's state on the contract: its conv caches (``flow``), HiFT's held-back frames
    # (``hift``) and samples (``tail``), each a plain Value the region that writes it lists, and its host fields
    # (``request``)
    state: str
    # cache family (``CACHE_FAMILIES``) -> its array, whose one read is under ``read(family)``
    kv: Mapping[str, str]
    # the output edge carrying each window's audio
    audio: str = "audio_chunk"
    # the ``step_metadata`` entry naming a request's voice, one of the node's bundled voices
    voice: str = "voice"

    @property
    def flow(self) -> str:
        return f"{self.state}_flow"

    @property
    def hift(self) -> str:
        return f"{self.state}_hift"

    @property
    def tail(self) -> str:
        return f"{self.state}_tail"

    @property
    def request(self) -> str:
        return f"{self.state}_request"

    @property
    def values(self) -> tuple[str, str, str]:
        """The Values that hold a slot per request, each one more than a cache's (its sink)."""
        return self.flow, self.hift, self.tail

    def read(self, family: str) -> str:
        """The key of the ``family`` cache's read: ``t2w_dit_attn`` for ``t2w_dit_kv``."""
        return f"{self.kv[family].removesuffix('_kv')}_attn"
