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
    episode_lifetime: bool = False

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
        # An episode keeps one existing recurrent-pool row under an internal
        # owner key while no request runs. The active request aliases that same
        # slot dict, so admission and step addressing remain the pool's own.
        self._episode_owner_by_request: dict[int, tuple[str, int]] = {}

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
        return min(self.logical_length(rid, self.config.labels[0]), self.window_size)

    def bind_episode(self, rid: int, slot: int, episode_id: str | None, *, cold: bool, reset: bool) -> None:
        """Use the conductor-assigned owner slot; never allocate a second slot here."""
        if episode_id is None:
            # An idless call is temporary: the normal request row is freed at
            # REMOVE_REQUEST, even though it consumes an episode admission slot.
            return
        owner = ("episode", slot)
        if cold:
            if owner in self._slots:
                raise ValueError(f"episode slot {slot} already has dense rows")
            self._slots[owner] = self._slots[rid]
        elif owner not in self._slots:
            raise ValueError(f"episode slot {slot} has no dense rows to resume")
        else:
            self._slots[rid] = self._slots[owner]
        if reset:
            self.reset_request(owner)
            for key in [key for key in self._logical_lengths if key[0] == owner]:
                del self._logical_lengths[key]
        self._episode_owner_by_request[rid] = owner

    def end_episode(self, slot: int) -> None:
        """Free an idle episode's physical row after its last REMOVE_REQUEST."""
        owner = ("episode", slot)
        if owner in self._episode_owner_by_request.values():
            raise ValueError(f"episode slot {slot} still has an active request")
        super().remove_request(owner)
        for key in [key for key in self._logical_lengths if key[0] == owner]:
            del self._logical_lengths[key]

    def remove_request(self, rid: str):
        owner = self._episode_owner_by_request.pop(rid, None)
        if owner is not None:
            # Drop only the active alias. `super` can then clear per-request
            # reservations without zeroing the persistent owner's slot.
            self._slots.pop(rid, None)
        super().remove_request(rid)
        if owner is None:
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
