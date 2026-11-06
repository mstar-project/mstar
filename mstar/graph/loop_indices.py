from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class NestedLoopIndices:
    """A snapshot of where execution is across nested loops at a given moment.

    Used by the conductor's stop-loop ordering to decide whether a freshly-
    received stop request is "newer" than a previously-applied one — so we
    don't double-stop the same loop when re-ordering messages.
    """
    loop_name_order: list[str]   # outer → inner
    loop_indices: dict[str, int]
    wg_fwd_pass_idx: int

    def label_context_gt(self, other: "NestedLoopIndices | None", target_loop_name: str | None=None) -> bool:
        """Whether ``self``'s iter indices are strictly greater than ``other``'s,
        in the path leading up to (but not including) ``target_loop_name``.

        Example: if we're stopping the loop ``target_loop_name`` but don't want
        to double-stop it, we can keep the last time it was stopped and only
        re-stop it again when ``new_time.label_context_gt(prev, target) == True``.
        """
        if other is None:
            return True
        if self.wg_fwd_pass_idx > other.wg_fwd_pass_idx:
            return True
        if self.wg_fwd_pass_idx < other.wg_fwd_pass_idx:
            return False
        for name in self.loop_name_order:
            if target_loop_name is not None and name == target_loop_name:
                break

            our_idx = self.loop_indices.get(name, 0)
            their_idx = other.loop_indices.get(name, 0)
            if our_idx > their_idx:
                return True
            if our_idx < their_idx:
                return False
        return False


@dataclass(frozen=True)
class LoopLayout:
    """One Loop's structural context, shared by planning and execution."""
    name: str
    ancestors: tuple[str, ...] = ()
    descendants: tuple[str, ...] = ()

    def advance(self, counts: Mapping[str, int], stride: int = 1) -> dict[str, int]:
        """Counters after crossing this Loop's back-edge, as graph routing
        advances the parent and starts each nested Loop afresh.
        """
        projected = dict(counts)
        projected[self.name] = projected.get(self.name, 0) + stride
        if self.descendants:
            projected.update(dict.fromkeys(self.descendants, 0))
        return projected
