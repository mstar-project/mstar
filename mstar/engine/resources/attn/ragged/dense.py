"""Cacheless varlen attention through FA3 (AITER on ROCm).

FA3 takes no workspace, so these managers allocate none.
"""

import functools
import logging

from mstar.engine.resources.attn.ragged.base import _RaggedCrossManagerBase, _RaggedSelfManagerBase
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.attn.ragged.dense_wrappers import (
    DenseRaggedCrossPrefillWrapper,
    DenseRaggedPrefillWrapper,
)

logger = logging.getLogger(__name__)


@functools.cache
def _warn_ragged_dense_fallback(reason: str, fallback: RaggedAttnBackend) -> None:
    logger.warning(
        "Ragged attention backend 'dense' requested, but %s; using the ragged %r backend instead.",
        reason, fallback.value,
    )


@functools.cache
def _warn_dense_on_flashinfer(what: str, reason: str) -> None:
    logger.warning(
        "%s backend 'dense' requested, but no varlen kernel is available (%s); "
        "running it on FlashInfer's ragged prefill instead.",
        what, reason,
    )


class DenseRaggedManager(_RaggedSelfManagerBase):
    wrapper_class = DenseRaggedPrefillWrapper


class DenseRaggedCrossManager(_RaggedCrossManagerBase):
    wrapper_class = DenseRaggedCrossPrefillWrapper
