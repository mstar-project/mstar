"""Speculative decoding: what a model declares, kept import-light; the resource
is reached by its own path (``spec_decode.resource``)."""

from mstar.engine.resources.spec_decode.config import (
    AcceptedTokens,
    DraftKind,
    SpecDecodeConfig,
    SpecDecodeSpec,
    SpecDecodeStep,
    SpecPhase,
)

__all__ = [
    "AcceptedTokens",
    "DraftKind",
    "SpecDecodeConfig",
    "SpecDecodeSpec",
    "SpecDecodeStep",
    "SpecPhase",
]
