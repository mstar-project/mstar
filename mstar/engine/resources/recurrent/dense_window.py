"""Dense whole-read windows over an append-only logical log.

The recurrent pool owns the physical slots and their admission. This resource
adds only the logical append count; its blocks hold a bounded ring of entries.
"""

from dataclasses import dataclass

from mstar.engine.resources.recurrent.config import RecurrentStateSpec, RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext


@dataclass
class DenseWindowSpec(RecurrentStateSpec):
    @property
    def restores_prefix(self) -> bool:
        return False

    @property
    def resource_class(self):
        return DenseWindowPool


@dataclass(frozen=True)
class DenseWindowStep(RecurrentStep):
    # The inherited segments address a slot even for a zero-append read. These
    # are the real append counts, used once per committed step.
    logical_segments: tuple[Segment, ...] = ()


class DenseWindowPool(RecurrentStatePool):
    """A recurrent slot pool whose blocks are entry rings, one per field."""

    supports_context_parallel = False
    supports_split_steps = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = next(iter(self.config.blocks.values())).shape[0]
        self._logical_lengths: dict[tuple[str, str], int] = {}

    def can_capture(self) -> bool:
        # A no-op read returns packed entries whose length varies by row. A
        # captured graph cannot replay that changing output shape.
        return False

    def max_repeats(self, step, ctx) -> int:
        return 1

    def logical_length(self, rid: str, label: str) -> int:
        return self._logical_lengths.get((rid, label), 0)

    def remove_request(self, rid: str):
        super().remove_request(rid)
        for key in [key for key in self._logical_lengths if key[0] == rid]:
            del self._logical_lengths[key]

    def commit(self, step: DenseWindowStep, ctx: StepContext) -> None:
        super().commit(step, ctx)
        if ctx.capture:
            return
        for segment in step.logical_segments:
            if segment.span > 0 and not ctx.is_padding_row(segment.request_id):
                key = (segment.request_id, segment.label)
                self._logical_lengths[key] = self._logical_lengths.get(key, 0) + segment.span
