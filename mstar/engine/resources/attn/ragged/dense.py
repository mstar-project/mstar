"""Cacheless varlen attention through FA3 (AITER on ROCm), or FA4 on Blackwell.

Neither takes a workspace, so these managers allocate none.
"""

import functools
import logging

from mstar.engine.resources.attn.ragged.base import (
    _RaggedCrossManagerBase,
    _RaggedKeyPrefixManagerBase,
    _RaggedSelfManagerBase,
)
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.dense_wrappers import (
    DenseKeyPrefixPrefillWrapper,
    DenseRaggedCrossPrefillWrapper,
    DenseRaggedPrefillWrapper,
    Fa4RaggedCrossPrefillWrapper,
    Fa4RaggedPrefillWrapper,
)
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)


@functools.cache
def _warn_ragged_fallback(requested: RaggedAttnBackend, reason: str, fallback: RaggedAttnBackend) -> None:
    logger.warning(
        "Ragged attention backend %r requested, but %s; using the ragged %r backend instead.",
        requested.value, reason, fallback.value,
    )


@functools.cache
def _warn_dense_on_flashinfer(what: str, reason: str) -> None:
    logger.warning(
        "%s backend 'dense' requested, but no varlen kernel is available (%s); "
        "running it on FlashInfer's ragged prefill instead.",
        what, reason,
    )


@functools.cache
def fa4_unavailable_reason(device=None) -> str | None:
    """Why FA4 varlen cannot run on ``device`` (the current GPU if None), or None if it can."""
    if current_platform.is_rocm():
        return "FA4 is not built for ROCm"
    capability = current_platform.cuda_capability(device)
    if capability is None or capability[0] != 10:
        return f"FA4 varlen is used on Blackwell (sm_100 / sm_103) only, not compute capability {capability}"
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:  # a broken install raises more than ImportError
        return f"flash-attn-4 (flash_attn.cute) is not importable: {exc!r}"
    return None


class DenseRaggedManager(_RaggedSelfManagerBase):
    wrapper_class = DenseRaggedPrefillWrapper


class DenseRaggedCrossManager(_RaggedCrossManagerBase):
    wrapper_class = DenseRaggedCrossPrefillWrapper


class DenseRaggedKeyPrefixManager(_RaggedKeyPrefixManagerBase):
    wrapper_class = DenseKeyPrefixPrefillWrapper


class Fa4RaggedManager(_RaggedSelfManagerBase):
    wrapper_class = Fa4RaggedPrefillWrapper


class Fa4RaggedCrossManager(_RaggedCrossManagerBase):
    wrapper_class = Fa4RaggedCrossPrefillWrapper
