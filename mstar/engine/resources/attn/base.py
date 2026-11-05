"""The attention resource's shared machinery: the spec-time factory and the
custom op every backend attends through.

The backends themselves live beside this — `flashinfer`, `aiter`, `cross`,
`dense`, `flex` — and `AttentionManager.build` reaches them through the table in
`backends`, which imports a backend only when it is built, so naming a
backend in a spec does not load the others.
"""

import logging
from typing import NamedTuple

import torch

from mstar.engine.resources.attn.backends import ATTN_BACKENDS
from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.kv.config import KVGroup, PagedKVConfig
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

    @staticmethod
    def _check_window_pool(spec: AttentionSpec, device: torch.device) -> str:
        """The FlashInfer kernel to read a pool that keeps a window with (`backends.window_pool_kernel`)."""
        from mstar.engine.resources.attn.backends import window_pool_kernel

        return window_pool_kernel(spec, device)

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

    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        """The current plan's query offsets per request, for backends that plan them."""
        raise NotImplementedError(f"{type(self).__name__} plans no query offsets (qo_indptr_buf)")

    # Eager as a whole: a break inside it leaves a compiled frame that holds the
    # bucket's static token count beside its row count, which captured forwards
    # recompile per prefill bucket until Dynamo's recompile limit.
    @torch.compiler.disable
    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        """Each request's last row of ``hidden``, used for sampling from prefill,
        for a backend whose ``qo_indptr_buf`` holds the plan's query offsets."""
        last = (self.qo_indptr_buf(label)[1:] - 1).long()
        return hidden.index_select(0, last)


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    # the int32 arrays' bytes
    page_indices: bytes
    last_page_lens: bytes


class EagerSlotKey(NamedTuple):
    """One eager wrapper — the counterpart of ``CGSlotKey``, slotted for the
    same reason (FlashInfer's plan stages into a pinned buffer per wrapper)."""
    label: str
    slot: int
    # decode and prefill take different wrapper classes; cross-attention plans
    # one wrapper per label either way and leaves this False
    is_decode: bool = False


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


def captured_page_capacity(
    num_rows: int, num_tokens: int, kv_config: PagedKVConfig, group: KVGroup | None = None,
) -> int:
    """Page ids a captured wrapper's index buffer holds for a bucket of ``num_rows`` rows and
    ``num_tokens`` tokens, in pages of ``group`` (the pool's first by default).

    Per row, since rows that matched one cached prefix all name its pages, and at least the
    bucket's own sink pages. A pool that keeps a window caches no prefix and forks nothing, so
    its rows name at most the pool, or in the window's group a window each, plus the step."""
    group = group or kv_config.groups[0]
    step_pages = -(-num_tokens // group.page_size)
    if group.retention is not None:
        # a page more per row, which a row's share of the step can cross into
        return num_rows * (group.retention.max_pages(group.page_size) + 1) + step_pages
    if len(kv_config.groups) > 1:
        return kv_config.num_units + step_pages + num_rows
    return max(num_rows * kv_config.max_num_pages, step_pages + num_rows)
