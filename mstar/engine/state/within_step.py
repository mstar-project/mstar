"""A handle on an array with ``max_len`` 0, whose read owns the attention's packing and its sequence-parallel exchange.

Apart from `mstar.engine.state.calls` because the exchange and the packing are `mstar.model`'s functions, and the shared
layers in `mstar.model` import `mstar.engine.state`.
"""

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.attn.ragged.config import cross_label
from mstar.engine.resources.step import Segment
from mstar.engine.state.calls import ArrayHandle
from mstar.engine.state.contract import Array
from mstar.model.components.diffusion.attention import joint_attention, sdpa_attention
from mstar.model.components.distributed.sequence_parallel import (
    attention_over_sharded_keys,
    sp_seq_split,
    ulysses_attention,
)


class WithinStepHandle(ArrayHandle):
    """The calls on an array with ``max_len`` 0, which a block compiled alone traces. With ``compile_op`` set, the read
    stays in that graph through ``mstar::ragged_attention``. A check here runs only while not compiling: it would guard
    on the layer, and the block would retrace per layer.

    ``queried`` is the array whose segments lay out the queries of a cross read. Under sequence parallelism the two
    flags pick the exchange: Ulysses when both sides are sharded, none when only the queries are, and a merge of the
    ranks' partial attentions when only the keys are. A read with no resource runs SDPA.
    """

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, *,
        resource_key: str | None = None, queried: Array | None = None, sp_group: CommGroup | None = None,
    ):
        super().__init__(array, resources, positions_key)
        self.resource_key = resource_key or next(iter(array.reads), None)
        self.queried = queried
        self.runs_sdpa = self.first_read is None
        self._sp = sp_group
        sharded = sp_group is not None and sp_group.world_size > 1
        self._keys_sharded = sharded and array.seq_sharded
        self._queries_sharded = sharded and (array if queried is None else queried).seq_sharded
        # Nothing for a step to do here: no plan, and no rank sizes to learn from its segments for a Ulysses exchange.
        self.plans_nothing = self.runs_sdpa and not (self._keys_sharded and self._queries_sharded)
        self._seq_sizes: list[int] | None = None
        # Both sides fix the plan's label, so it is one string for the life of the handle. See `cross_label`.
        self._plan_label = min(array.labels) if queried is None else cross_label(min(queried.labels), min(array.labels))
        # One callable per array, not per layer: Dynamo specializes `ulysses_attention` on the identity of the callable
        # it runs, so a callable per layer would retrace that frame per layer.
        self._attend = sdpa_attention if self.runs_sdpa else functools.partial(
            joint_attention, ragged=functools.partial(self.first_read.run, label=self._plan_label),
        )

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if step.ctx.replays:
            return
        segments = step.state[self.name].segments or step.segments or ()
        if self.queried is not None:
            queries = step.state[self.queried.name].segments or step.segments
            if [s.request_id for s in queries] != [s.request_id for s in segments]:
                raise self._refusal(f"its segments and those of {self.queried.name!r}, which lay out its queries, must "
                                    "name the same requests in the same order, with one segment to one")
        if self._keys_sharded and self._queries_sharded:
            spans = {s.span for s in segments}
            if len(spans) > 1:
                raise self._no_layout(f"a Ulysses exchange over spans {sorted(spans)}, and mixed lengths need a "
                                      "permutation in the unpack")
            self._seq_sizes = sp_seq_split(spans.pop(), self._sp.world_size) if spans else None

    def query_share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        """The query layout this read's plan sees of ``segments``, the query array's in this step."""
        return self._share(segments) if self._queries_sharded and not self._keys_sharded else tuple(segments)

    def key_share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        """The key layout this read's plan sees of ``segments``, this array's in this step."""
        return self._share(segments) if self._keys_sharded and not self._queries_sharded else tuple(segments)

    def _share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        world_size, rank = self._sp.world_size, self._sp.rank
        return tuple(s._replace(span=sp_seq_split(s.span, world_size)[rank]) for s in segments)

    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        """Hold k and v for this layer's read. Every add here is scratch."""
        if keep:
            raise self._refusal("keep=True, and max_len 0 keeps nothing between steps")
        if not (self.runs_sdpa or torch.compiler.is_compiling()):
            self._held_layer = self._layer_id(layer)
            self._record_add(self._held_layer, keep)
        self._held = fields["k"], fields["v"]

    def read(
        self, layer: int | torch.Tensor, q: torch.Tensor | None = None, *,
        start: Mapping[str, torch.Tensor] | None = None, **mask_args: Any,
    ) -> torch.Tensor:
        """Attend ``q`` over this layer's latest add, within each row's segment. ``q`` is (tokens, heads, head_dim), or
        (rows, len, heads, head_dim) for rows of one length, which is the view an exchange and SDPA take."""
        if start is not None:
            raise self._refusal("start, on an array that isn't a Value")
        if q is None:
            raise self._no_layout("a read with no op")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        if not (self.runs_sdpa or torch.compiler.is_compiling()):
            self._record_read(self._layer_id(layer))
        # Dropped here, not at the step's end: a compiled frame would otherwise return k and v from its graph, which
        # keeps the whole projection they are views of alive through the next layer.
        k, v = self._take_hold(layer)
        if q.ndim == 3:
            if self.runs_sdpa:
                raise self._no_layout("SDPA over packed tokens, which can't tell one segment from the next")
            return self.first_read.run(q, k, v, label=self._plan_label)
        if self._keys_sharded and self._queries_sharded:
            return ulysses_attention(self._sp, q, k, v, self._attend, self._seq_sizes, seq_dim=1, head_dim=2)
        if self._keys_sharded:
            return attention_over_sharded_keys(self._sp, q, k, v, self._attend)
        return self._attend(q, k, v)
