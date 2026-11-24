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
    supports_drop = False

    def check_drop(self, rid: str, tokens: int) -> str:
        return "dense window entries cannot be rebuilt from token ids"

    def drop(self, rid: str, tokens: int) -> int:
        raise RuntimeError(self.check_drop(rid, tokens))

    @classmethod
    def build(cls, spec, info):
        pool = super().build(spec, info)
        pool.episode_lifetime = spec.episode_lifetime
        return pool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = next(iter(self.config.blocks.values())).shape[0]
        self._logical_lengths: dict[tuple[object, str], int] = {}

    def can_capture(self) -> bool:
        # A no-op read returns packed entries whose length varies by row. A
        # captured graph cannot replay that changing output shape.
        return False

    def max_repeats(self, step, ctx) -> int:
        return 1

    def logical_length(self, rid: str, label: str) -> int:
        owner = self._episode_owner_by_request.get(rid, rid)
        return self._logical_lengths.get((owner, label), 0)

    def held(self, rid: str) -> int:
        """Entries a later read can still reach, after the last committed step."""
        return sum(min(self.logical_length(rid, label), self.window_size) for label in self.config.labels)

    def bind_episode(self, rid: int, slot: int, episode_id: str | None, *, cold: bool, reset: bool) -> None:
        """Use the conductor-assigned owner slot; never allocate a second slot here."""
        super().bind_episode(rid, slot, episode_id, cold=cold, reset=reset)
        if episode_id is not None and reset:
            owner = ("episode", slot)
            for key in [key for key in self._logical_lengths if key[0] == owner]:
                del self._logical_lengths[key]

    def end_episode(self, slot: int) -> None:
        """Free an idle episode's physical row after its last REMOVE_REQUEST."""
        super().end_episode(slot)
        owner = ("episode", slot)
        for key in [key for key in self._logical_lengths if key[0] == owner]:
            del self._logical_lengths[key]

    def remove_request(self, rid: str):
        is_named = rid in self._episode_owner_by_request
        super().remove_request(rid)
        if not is_named:
            for key in [key for key in self._logical_lengths if key[0] == rid]:
                del self._logical_lengths[key]

    def commit(self, step: DenseWindowStep, ctx: StepContext) -> None:
        super().commit(step, ctx)
        if ctx.capture:
            return
        for segment in step.logical_segments:
            if segment.span > 0 and not ctx.is_padding_row(segment.request_id):
                owner = self._episode_owner_by_request.get(segment.request_id, segment.request_id)
                key = (owner, segment.label)
                self._logical_lengths[key] = self._logical_lengths.get(key, 0) + segment.span
