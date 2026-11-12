"""Attention over a kept window: each row's sink and latest window on a paged array, and the step's own adds."""

import numpy as np
import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.ragged.base import RaggedCrossAttnManager
from mstar.engine.resources.attn.ragged.config import (
    KeptWindowAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.plan import kept_reach
from mstar.engine.resources.step import Segment, StepContext

# the inner plan's labels: the step's queries, and the reach they attend
_QUERIES, _REACH = "queries", "reach"


class KeptWindowAttention(AttentionManager):
    """Each call gathers its rows' reach from the pages, as `kv.plan.kept_reach` gives it from the pool's views, and
    attends it through a ragged cross plan of the step's queries, the kernel a rolling store's window runs on.

    The reach changes length from step to step, so a captured graph can't replay it, and a plan under a lease is
    refused. Every add writes the pages, so the gather reads this step's entries from them too."""

    def __init__(self, inner, kv_cache: str):
        self._inner = inner
        self._kv_cache_name = kv_cache
        self._kv = None
        # each row's reach, staged by `plan`: every entry's page and its offset in the page
        self._pages: torch.Tensor | None = None
        self._offsets: torch.Tensor | None = None

    @classmethod
    def build(cls, spec: KeptWindowAttentionSpec, info: EngineResourceInfo) -> "KeptWindowAttention":
        inner = RaggedCrossAttnManager.build(
            RaggedCrossAttentionSpec(resource_key=spec.resource_key, nodes=spec.nodes, config=spec.config), info,
        )
        return cls(inner, spec.kv_cache)

    def depends_on(self) -> set[str]:
        return {self._kv_cache_name}

    def bind_dependencies(self, resources) -> None:
        self._kv = resources[self._kv_cache_name]

    def can_capture(self) -> bool:
        return False

    @property
    def requires_kv_write(self) -> bool:
        return True

    @property
    def force_double_buffer(self) -> bool:
        return self._inner.force_double_buffer

    def plan(self, step: AttentionStep, ctx: StepContext) -> None:
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        if ctx.slot_lease is not None or ctx.is_preplan:
            raise RuntimeError(
                "a kept-window read is eager-only: its reach changes length from step to step, so a captured "
                "graph has nothing fixed to replay"
            )
        labels = {segment.label for segment in step.segments}
        if len(labels) != 1:
            raise RuntimeError(f"no layout yet: a kept-window read over labels {sorted(labels)}, and it plans one")
        (label,) = labels
        page_size = self._kv.config.tokens_per_page
        queries, reaches, pages, offsets = [], [], [], []
        for view in ctx.plan_results[self._kv_cache_name][label].views:
            slots = kept_reach(view, page_size)
            queries.append(Segment(view.request_id, _QUERIES, view.to_compute))
            reaches.append(Segment(view.request_id, _REACH, len(slots.pages)))
            pages.append(slots.pages)
            offsets.append(slots.offsets)
        self._inner.plan(RaggedCrossAttentionStep(segments=(*queries, *reaches), pairs=((_QUERIES, _REACH),)), ctx)
        device = self._kv.kv_cache.tensor.device
        self._pages = torch.from_numpy(np.concatenate(pages)).to(device)
        self._offsets = torch.from_numpy(np.concatenate(offsets)).to(device)

    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        raise ValueError("no layout yet: at_last_token on a kept-window read, whose plan selects no last token")

    def run(
        self, q: torch.Tensor, label: str | None = None, kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None, v: torch.Tensor | None = None, layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Attend ``q`` over the reach `plan` staged, read from the pages of pool layer ``layer_idx``."""
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        reach = self._kv.kv_cache.read_tokens(layer_idx, self._pages, self._offsets)
        return self._inner.run(q, reach[:, 0], reach[:, 1], label=cross_label(_QUERIES, _REACH))
