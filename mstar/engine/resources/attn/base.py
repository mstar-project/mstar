"""The attention resource's shared machinery: the spec-time factory, the
custom op every backend attends through, and the workspace pool.

The backends themselves live beside this — `flashinfer`, `cross`, `dense`,
`flex` — and `AttentionManager.build` reaches them through the table in
`backends`, which imports a backend only when it is built, so naming a
backend in a spec does not load the others.
"""

import logging
import os
from typing import NamedTuple

import torch

from mstar.engine.resources.attn.backends import ATTN_BACKENDS
from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.keys import fingerprint

logger = logging.getLogger(__name__)


def resolve_attn_backend(
    backend: AttnBackend, device: torch.device, page_size: int | None = None,
) -> AttnBackend:
    """The concrete backend for ``backend`` on ``device``.

    ``AUTO`` becomes the platform's default (``backends.ATTN_BACKENDS``), which
    may depend on the paged cache's ``page_size``. A backend named in the spec
    or YAML is returned as is.
    """
    return ATTN_BACKENDS.resolve(backend, page_size)


class AttentionManager(AttentionResource):
    prefix_skip_safe = True

    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    def fingerprint(self) -> bytes:
        # the class, not the declared backend: `build` moves dense to another
        # kernel where the FA3 one is unusable, and they do not agree bit for bit
        return fingerprint(type(self).__name__)

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        # checks the spec against the backend it resolves to, then builds it
        # (`AttnBackendRegistry.build`)
        return ATTN_BACKENDS.build(spec, info)

    @property
    def requires_kv_write(self) -> bool:
        """Whether a layer must write this step's K/V through the KV resource
        before calling ``run``.

        False only for backends that take the fresh K/V straight into the
        kernel (the dense one). A layer therefore reads

            if self.attn.requires_kv_write:
                self.kv.write_kv(k, v, layer_idx=i, label=label)
            out = self.attn.run(q, label, kv.layer_view(i), k=k, v=v, layer_idx=i)

        and stays correct whichever backend the spec named.
        """
        return True


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


class EagerSlotKey(NamedTuple):
    """One eager wrapper — the counterpart of ``CGSlotKey``, slotted for the
    same reason (FlashInfer's plan stages into a pinned buffer per wrapper)."""
    label: str
    slot: int
    # decode and prefill take different wrapper classes
    is_decode: bool = False


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


def captured_page_capacity(num_rows: int, num_tokens: int, kv_config: KVConfig) -> int:
    """Page ids a captured wrapper's index buffer holds for a bucket of
    ``num_rows`` rows and ``num_tokens`` tokens.

    Per row, not per pool: rows that matched one cached prefix all name its
    pages, so a step can name more page ids than the pool holds. And at least
    the bucket's own pages: a capture's rows name a SINK_PAGE per page of span,
    whatever the pool's size."""
    return max(
        num_rows * kv_config.max_num_pages,
        -(-num_tokens // kv_config.page_size) + num_rows,
    )


def workspace_bytes() -> int:
    """Size of one FlashInfer workspace buffer (``MSTAR_WORKSPACE_BUFFER_MB``)."""
    return int(os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")) * 1024 * 1024


class WorkspacePool:
    """FlashInfer workspace buffers, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so
    these outlive any single step.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._size = workspace_bytes()
        self._buffers: dict[str, torch.Tensor] = {}

    def get(self, label: str, cg_slot: int | None = None) -> torch.Tensor:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._buffers:
            self._buffers[key] = torch.empty(
                self._size, dtype=torch.uint8, device=self._device
            )
        return self._buffers[key]
