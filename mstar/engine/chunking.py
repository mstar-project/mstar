"""Chunked prefill's state on the engine.

Per (node, walk): the submodule's chunking policy and the token budget a step
runs under, after the serving config's overrides. Per row mid-prefill: its
whole prepared input, which every chunk is cut from. And the count of steps
that ran over their budget, for the warning.
"""

import logging
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from mstar.model.submodule_base import ChunkedPrefillOutputPolicy, ChunkingPolicy, NodeInputs

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch, SubmoduleManagement

logger = logging.getLogger(__name__)

# The fewest tokens the scheduler cuts a row to (one KV page at the default
# page size), or the whole budget when that is smaller. Without it, decode rows
# that fill the budget leave a prompt a token or two a step.
MIN_CHUNK_TOKENS = 128


def min_chunk_tokens(budget: int) -> int:
    return min(MIN_CHUNK_TOKENS, budget)


class ChunkingManagement:
    def __init__(self, submodules: Mapping[str, "SubmoduleManagement"]):
        # the engine's own mapping, which it fills as it loads nodes
        self._submodules = submodules
        # (node, walk) -> the submodule's chunking policy, asked once
        self._policies: dict[tuple[str, str], ChunkingPolicy] = {}
        # the serving config's max_batch_tokens: node -> budget, (node, walk) -> budget; None is off
        self._overrides: dict[str | tuple[str, str], int | None] = {}
        # (node, real walk) -> the combined walk the scheduler batches it under
        self._combined_walk_of: dict[tuple[str, str], str] = {}
        # nodes whose walks' budgets also cap their rows inside a combined step
        # (the serving config's can_override_mixed_max_tokens)
        self._override_mixed: frozenset[str] = frozenset()
        # (node, combined walk) -> walk -> the cap on that walk's rows in a step
        self._walk_caps: dict[tuple[str, str], dict[str, int] | None] = {}
        # (node, walk) -> steps that ran over the budget they were scheduled under
        self._overruns: dict[tuple[str, str], int] = {}
        # nodes the worker chunks nowhere, whatever the model declares
        self._no_chunking: frozenset[str] = frozenset()
        # (rid, node) -> a chunked row's whole input, prepared on its first chunk
        self.prepared: dict[tuple[int, str], NodeInputs] = {}

    def policy(self, node_name: str, graph_walk: str) -> ChunkingPolicy:
        """The submodule's, asked once per (node, walk)."""
        key = (node_name, graph_walk)
        policy = self._policies.get(key)
        if policy is None:
            submodule = self._submodules[node_name].submodule
            policy = self._policies[key] = submodule.get_chunking_policy(graph_walk)
        return policy

    def max_batch_tokens(self, node_name: str, graph_walk: str) -> int | None:
        """The walk's budget, unless the serving config overrides it or the
        worker chunks nothing on the node."""
        if node_name in self._no_chunking:
            return None
        budget = self.policy(node_name, graph_walk).max_batch_tokens
        if budget is None or not self._overrides:
            return budget
        for key in ((node_name, graph_walk), node_name):
            if key in self._overrides:
                return self._overrides[key]
        return budget

    def set_budgets(
        self, overrides: Mapping[str, Any], walks: set[str],
        combined_walk_of: Mapping[tuple[str, str], str] | None = None,
        override_mixed: Collection[str] = (),
        no_chunking: Collection[str] = (),
    ) -> None:
        """Apply a serving config's ``max_batch_tokens`` to this engine's nodes.

        Per node, one budget for every walk the model budgets, or a mapping of
        walk -> budget; null turns chunking off. Only walks the submodule
        budgets can be overridden: elsewhere the setting would do nothing.

        A walk inside a combined walk runs in the combined walk's steps, under
        the combined walk's budget, so its own budget applies there only on a
        node in ``override_mixed`` (``can_override_mixed_max_tokens``), where
        it caps that walk's rows within the step. Elsewhere a per-walk override
        of such a walk would do nothing, so it is refused; the combined walk
        itself can be named instead.

        ``no_chunking`` names nodes this worker chunks nowhere (TP/SP, which
        chunked prefill does not support yet). A budget set on one is still
        validated but has no effect, since the deployment, not the config, is
        what turned chunking off there.
        """
        self._no_chunking = frozenset(no_chunking)
        self._combined_walk_of = dict(combined_walk_of or {})
        self._override_mixed = frozenset(override_mixed)
        self._walk_caps = {}
        combined_walks = {(node, combined) for (node, _), combined in self._combined_walk_of.items()}
        for node, setting in overrides.items():
            if node not in self._submodules:
                continue
            per_walk = setting if isinstance(setting, Mapping) else None
            for walk, budget in (per_walk or {None: setting}).items():
                if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
                    raise ValueError(f"max_batch_tokens for {node}: {budget!r} is not a positive int or null")
                if walk is None:
                    self._overrides[node] = budget
                    continue
                if walk not in walks and (node, walk) not in combined_walks:
                    raise ValueError(f"max_batch_tokens names walk {walk!r} of {node}, which the model never runs")
                combined = self._combined_walk_of.get((node, walk))
                if combined is not None and node not in self._override_mixed:
                    raise ValueError(
                        f"max_batch_tokens for {node}/{walk}: {walk} runs in {combined!r} steps, "
                        f"under {combined!r}'s budget, so this would do nothing; set "
                        f"can_override_mixed_max_tokens for {node} to cap {walk}'s rows within "
                        f"those steps, or budget {node} (or {node}/{combined}) instead"
                    )
                if self.policy(node, walk).max_batch_tokens is None:
                    raise ValueError(f"max_batch_tokens for {node}/{walk}: the model sets no budget there to override")
                self._overrides[(node, walk)] = budget

    def walk_caps(self, node_name: str, graph_walk: str) -> dict[str, int] | None:
        """Under a combined walk on a node with ``can_override_mixed_max_tokens``:
        walk -> the most tokens that walk's rows may run in one step, for the
        walks with a budget of their own. None otherwise."""
        key = (node_name, graph_walk)
        if key not in self._walk_caps:
            caps = None
            if node_name in self._override_mixed:
                caps = {
                    walk: budget
                    for (node, walk), combined in self._combined_walk_of.items()
                    if node == node_name and combined == graph_walk
                    and (budget := self.max_batch_tokens(node, walk)) is not None
                } or None
            self._walk_caps[key] = caps
        return self._walk_caps[key]

    def chunkable(self, node_name: str, graph_walk: str) -> bool:
        """The submodule's say, unless the serving config turned the walk's
        chunking off (a null budget) or the worker chunks nothing on the node."""
        if node_name in self._no_chunking:
            return False
        if self._overrides.get((node_name, graph_walk), 0) is None:
            return False
        return self.policy(node_name, graph_walk).chunkable

    def output_policies(
        self, node_name: str, graph_walk: str,
    ) -> Mapping[str, ChunkedPrefillOutputPolicy]:
        return self.policy(node_name, graph_walk).output_policies

    def check_budget(self, batch: "ExecutingBatch", node_inputs: list[NodeInputs]) -> None:
        """Warn when a step runs over the budget it was scheduled under: that of
        the combined walk it ran in, if any. A row too long to cut runs whole,
        and a speculation merge sizes its continuing rows by estimate, before
        their inputs exist. Logged on the 1st, 2nd, 4th, ... overrun per walk."""
        # the step's label is not always the combined walk: a combined batch
        # whose rows share one real walk is relabelled to it
        # (`ScheduledBatch.relabel_if_uniform`), and it still ran under the
        # combined walk's budget
        walk = batch.step_context.graph_walk
        walk = self._combined_walk_of.get((batch.node_name, walk), walk)
        budget = self.max_batch_tokens(batch.node_name, walk)
        if budget is None:
            return
        total = sum(inp.input_seq_len for inp in node_inputs)
        # a row cut to the chunk floor may run the step over by up to that floor
        if total <= budget + min_chunk_tokens(budget):
            return
        key = (batch.node_name, walk)
        count = self._overruns[key] = self._overruns.get(key, 0) + 1
        if count & (count - 1) == 0:
            logger.warning(
                "%s/%s: step of %d tokens over its budget of %d (%d overruns so far): "
                "a row too long to cut ran whole, or continuing rows ran more tokens "
                "than estimated",
                batch.node_name, walk, total, budget, count,
            )

    def release(self, rid: int, node_name: str) -> None:
        """Drop a row's whole input once its last chunk lands."""
        self.prepared.pop((rid, node_name), None)

    def forget_request(self, rid: int) -> None:
        # over a snapshot: this runs on the main thread, while the gpu thread
        # may still be adding the next step's rows to `prepared`
        for key in [key for key in list(self.prepared) if key[0] == rid]:
            del self.prepared[key]
