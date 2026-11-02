"""Sequence parallelism: whether a step's tokens split over a node's SP ranks, and how attention
reaches the tokens it needs when they do.

A node's SP ranks share each step one of two ways. Its tokens are **split** (each rank runs the
token-local layers on its share) or **copied** (every rank runs them all). The sequence shard
resource (``SeqShardSpec``) decides which, per step, from the step the model declares, and gives
each listed label its per-rank ranges (``SeqRange``): contiguous, or on a node whose attention
runs the ring, striped by KV position. A model brackets its layer stack with ``scatter`` /
``gather`` on that resource; nothing else in model code changes.

The attention resources over the node depend on it (``shard_key``, taken in their specs'
``derive``) and pick an exchange per label from its plan: what their storage needs to attend a
split label's tokens, or a copied label's at the node's head geometry. The exchanges are the
functions at the bottom of this module, each wrapped around the resource's own ``run``:

- ``ulysses``: one all-to-all from tokens to heads, ``run``, and one back (head-stored KV, or a
  cacheless attention declaring ``sp_exchange="ulysses"``);
- ``all_gather``: every rank's q, k and v gathered, ``run`` as on a copied step, this rank's share
  of the output (KV stored by position, ``context_parallel``);
- ``head_slice``: a copied step at a ``tp * sp`` head geometry: this rank's heads, ``run``, gather
  the heads;
- ``sharded_key_merge``: copied queries over split keys (cacheless cross-attention): each rank's
  partial with its log-sum-exp, merged.

The ring over KV stored by position (``sp_exchange: ring``) isn't one of them: it writes and
attends through the KV resource itself (``AttentionManager.exchange``, ``RingAttentionWrapper``).

``sequence_parallel_refusals`` is what a node on SP ranks may not declare; the engine and
``validate_deployment`` both run it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup, gather_sequence, sp_seq_split
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.context_parallel import gather_merge, position_pools, stripe_owners
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep, StepContext
from mstar.utils.h2d import to_device_nonblocking

if TYPE_CHECKING:
    from mstar.model.components.moe_parallel import MoeParallelConfig


# --- declaration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqShardStep(ResourceStep):
    """The step's segments and ``combined_labels`` (``SubmoduleStep`` fills both, unless the step
    sets its own segments), so a combined plan label gets a range over its sources too."""


@dataclass
class SeqShardSpec(NodeResourceSpec):
    """A node's steps split over its SP ranks, or copied.

    ``labels`` are the step labels whose tokens split. ``axis`` 0 splits a label's packed tokens
    into contiguous ranges; ``axis`` 1 splits each row of a ``[B, S, ...]`` layout alike (every row
    of a label one length). ``walks`` lists the walks that may split (None: every walk); the
    others plan copied. A step splits only when it is not one token a row, its listed tokens are
    at least ``max(sp, min_tokens)``, and every listed label has at least ``sp`` tokens (a row,
    for ``axis`` 1), so no rank gets an empty range. ``min_tokens`` is a ``resources:`` override.

    The attention and position resources on the node take this as a dependency in their own
    ``derive``, so every ``declare_step`` branch of the node declares a ``SeqShardStep``; at one
    SP rank it plans copied at no cost.

    Two fields are derived from the attentions on its nodes as configured (``derive``):
    ``splits``, whether any of them takes split steps (every one's ``sp_exchange`` ``none``, set or
    the default over KV stored by position, makes the node run copied, as one without a shard), and
    ``stripe_pool``, the pool a ring attention reads, by whose stored positions each step is
    striped."""

    labels: tuple[str, ...]
    axis: int
    walks: Collection[str] | None = None
    min_tokens: int = field(default=0, kw_only=True)
    splits: bool = field(default=True, init=False, repr=False)
    stripe_pool: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.labels = tuple(self.labels)
        if not self.labels:
            raise ValueError(f"sequence shard {self.resource_key!r} lists no labels")
        if self.axis not in (0, 1):
            raise ValueError(f"sequence shard {self.resource_key!r}: axis must be 0 or 1, not {self.axis!r}")
        if self.walks is not None:
            self.walks = frozenset(self.walks)
        self._check_min_tokens()

    def _check_min_tokens(self) -> None:
        if type(self.min_tokens) is not int or self.min_tokens < 0:
            raise ValueError(f"min_tokens must be a non-negative integer, got {self.min_tokens!r}")

    @property
    def resource_class(self) -> type[Resource]:
        return SequenceShard

    @property
    def splits_steps(self) -> bool:
        return self.splits

    def depends_on(self) -> set[str]:
        # a striped step plans from each row's stored length, which the pool's plan holds. A
        # dependency of the shard, not of the pool, so the pool's prefix cache stays open
        # (`opening_prefix_caches` reads a cache keeper's own dependencies alone)
        return set() if self.stripe_pool is None else {self.stripe_pool}

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        # from raw fields alone, so it doesn't matter which spec derives first
        from mstar.engine.resources.attn.config import AttentionSpec

        attentions = [spec for spec in specs.values() if spec.nodes & self.nodes and spec.attends_split_steps]
        # a kind that takes split steps; a paged one's unset sp_exchange follows its pool's storage
        exchanges = {
            spec.resource_key: spec.resolve_exchange(specs) if isinstance(spec, AttentionSpec) else spec.sp_exchange
            for spec in attentions
        }
        self.splits = any(exchange != "none" for exchange in exchanges.values())
        pools = sorted({spec.config.kv_cache for spec in attentions if exchanges[spec.resource_key] == "ring"})
        self.stripe_pool = pools[0] if len(pools) == 1 else None
        if len(pools) > 1:
            return [f"its ring attentions read KV pools {pools}, whose stored positions would stripe a step two ways"]
        return []

    def apply_yaml_overrides(self, min_tokens: int | None = None):
        """The step size below which a node runs copied: a deployment's trade between the
        exchanges' cost and the token-local layers' saving."""
        if min_tokens is not None:
            self.min_tokens = min_tokens
            self._check_min_tokens()


def take_shard_edge(spec: NodeResourceSpec, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """Set ``spec.shard_key`` to the ``SeqShardSpec`` over its nodes, if any (a ``derive``).
    A shard sharing only some of ``spec``'s nodes is a problem: the spec would split on some of
    its nodes and not others."""
    shards = [other for other in specs.values() if isinstance(other, SeqShardSpec) and other.nodes & spec.nodes]
    spec.shard_key = None
    if not shards:
        return []
    if len(shards) > 1:
        return [f"shares nodes with sequence shards {sorted(s.resource_key for s in shards)}; a node takes one"]
    shard, = shards
    if not spec.nodes <= shard.nodes:
        return [
            f"spans nodes {sorted(spec.nodes - shard.nodes)} without the sequence shard "
            f"{shard.resource_key!r}: declare the SeqShardSpec over {sorted(spec.nodes)} too"
        ]
    spec.shard_key = shard.resource_key
    return []


# --- the resource -----------------------------------------------------------------------------


class SeqRange(NamedTuple):
    """One label's split over the SP ranks: each rank's count of its tokens (of each row's, for a
    ``[B, S, ...]`` layout), and which they are: contiguous in rank order, or ``order``."""

    group: CommGroup
    counts: tuple[int, ...]
    # B of a [B, S, ...] layout split along S (axis 1); None for a packed label (axis 0)
    rows: int | None
    # striped (`SeqShardSpec.stripe_pool`): the step's token indices, rank 0's then rank 1's and
    # on, each rank's in step order, on the device; None for contiguous ranges
    order: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return self.counts[self.group.rank]

    @property
    def local_tokens(self) -> int:
        """The tokens this rank holds of the label, as a packed tensor's first dim."""
        return self.count * (self.rows or 1)

    @property
    def index(self) -> torch.Tensor:
        """Striped: this rank's tokens' indices in the step."""
        start = sum(self.counts[: self.group.rank])
        return self.order[start:start + self.count]

    def local(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """This rank's share of ``x`` (the whole label) along ``dim``: a view of a contiguous range,
        which a captured graph reads in place; a copy of a striped one, as a kernel takes it."""
        if self.order is None:
            return x.narrow(dim, sum(self.counts[: self.group.rank]), self.count)
        return x.index_select(dim, self.index)

    def whole(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """``local``'s inverse: every rank's share of ``x`` gathered, in step order."""
        out = gather_sequence(self.group, x, self.counts, dim)
        if self.order is None:
            return out
        return torch.empty_like(out).index_copy_(dim, self.order, out)


class SeqShardPlan(NamedTuple):
    """A step's split: whether its tokens split, and each split label's ranges."""

    split: bool
    ranges: Mapping[str, SeqRange]


COPIED = SeqShardPlan(False, {})


class SequenceShard(Resource):
    """Plans each step split or copied (``SeqShardSpec``) and brackets the layer stack with
    ``scatter`` / ``gather``. Holds no per-request state and no device buffers: its plan is
    from the declared step, the lease and (striped) the pool's plan, the same on every rank; a
    striped step's ``order`` is copied to the device with it. Off its defaults:
    ``prefix_skip_safe`` (it plans from the step a prefix hit already trimmed), ``supports_preplan``
    (its dependents pre-plan), ``supports_context_parallel`` (a striped shard reads a placed pool's
    plan); it keeps ``can_capture`` True and ``force_double_buffer`` False (a leased step, the
    only kind a pre-plan stages, never stripes)."""

    prefix_skip_safe = True
    supports_split_steps = True
    supports_context_parallel = True

    def __init__(
        self, labels: tuple[str, ...], axis: int, walks: frozenset[str] | None, min_tokens: int,
        group: CommGroup, splits: bool = True, stripe_pool: str | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.labels = labels
        self.axis = axis
        # the walks that may split; None for every walk
        self.walks = walks
        self.min_tokens = min_tokens
        self.group = group
        # `SeqShardSpec.stripe_pool`
        self.stripe_pool = stripe_pool
        self._device = device
        # fixed at bind, so a compiled frame at one SP rank (or on a node whose attentions take no
        # split steps) traces the bracket straight through
        self.can_split = group.world_size > 1 and splits
        self._current: SeqShardPlan | None = None
        self._pending: SeqShardPlan | None = None
        self._preplanned = False
        # set by an attention resource around its run inside an exchange, which `check_bypass` reads
        self.exchanging = False
        # the labels this step's layer stack holds a rank's share of (`scatter`, until `gather`)
        self._scattered: set[str] = set()

    @classmethod
    def build(cls, spec: SeqShardSpec, info: EngineResourceInfo) -> SequenceShard:
        joint = info.joint_comm_group
        group = joint.sp_group if joint is not None else CommGroup.trivial()
        walks = None if spec.walks is None else frozenset(spec.walks)
        return cls(
            spec.labels, spec.axis, walks, spec.min_tokens, group, spec.splits, spec.stripe_pool, info.device,
        )

    def depends_on(self) -> set[str]:
        return set() if self.stripe_pool is None else {self.stripe_pool}

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._pending = None
        self._preplanned = False

    def check_captures(self, node: str, configs, piecewise: Mapping) -> None:
        """Refuse captures this node can't replay when it splits: a piecewise region (its inner
        steps aren't bracketed, and its replay resets padding rows, so ``reset_request``, inside the
        forward), a graph replayed under a walk that decides otherwise than the walk
        it was captured under, and a packed capture whose padding has no one label to join."""
        # the config module reaches back here through the submodule base
        from mstar.engine.accelerator_graph_config import AcceleratorGraphConfigType

        if not self.can_split:
            return
        if piecewise:
            raise ValueError(
                f"node {node!r} splits its steps over SP ranks, which its piecewise captures {sorted(piecewise)} don't"
            )
        for config in configs:
            walks = {config.capture_graph_walk, *config.replay_graph_walks}
            listed = walks if self.walks is None else walks & self.walks
            if listed and listed != walks:
                raise ValueError(
                    f"node {node!r}: a graph captured for walks {sorted(walks)} replays both walks its sequence "
                    f"shard splits ({sorted(listed)}) and walks it doesn't; list all of them or none"
                )
            packed = config.get_config_type() == AcceleratorGraphConfigType.FLASH_INFER_PACKED
            if listed and packed and (len(self.labels) > 1 or self.axis != 0 or config.total_tokens_multiplier > 1):
                raise ValueError(
                    f"node {node!r}: a packed capture of walks {sorted(walks)} pads its replays, which a sequence "
                    "shard splitting them places on its one label: it needs one listed label, axis 0 and no "
                    "total_tokens_multiplier"
                )

    def plan(self, step: SeqShardStep, ctx: StepContext) -> SeqShardPlan:
        assert not (self._preplanned and ctx.is_preplan), (
            "sequence shard preplan is already pending; clear_preplan before planning a different step ahead"
        )
        if self._preplanned:
            # the dependents read this from ctx.plan_results, which the runner cleared
            self._current, self._pending, self._preplanned = self._pending, None, False
            return self._current
        result = self._decide(step, ctx) if self.can_split else COPIED
        if ctx.is_preplan:
            self._pending, self._preplanned = result, True
        else:
            self._current = result
            self._scattered.clear()
        return result

    def commit(self, step: SeqShardStep, ctx: StepContext) -> None:
        # the plan ends with its step: a bracket outside one (a warmup compile) is the identity
        self._current = None
        self._scattered.clear()

    def reset_request(self, rid: str, free: bool = False) -> None:
        # a capture's step ends here rather than at commit. This clears whatever rid it is given, so
        # every caller must run after a step's commit or before the next plan, never mid-forward
        # (see check_captures' piecewise refusal)
        self._current = None
        self._scattered.clear()

    def _decide(self, step: SeqShardStep, ctx: StepContext) -> SeqShardPlan:
        """Split or copied, from inputs every rank of the instance shares: the declared step, the
        walk and the lease."""
        lease = ctx.slot_lease
        bucket = None if lease is None else lease.bucket
        if bucket is not None and not ctx.force_prefill and bucket.bs == bucket.num_tokens:
            # a batched decode replay, which declares its bucket: copied in O(1)
            return COPIED
        if bucket is not None and self.stripe_pool is not None:
            # the ring runs eager only; the lease is the instance's capture decision, so every
            # rank plans the step copied alike
            return COPIED
        sp = self.group.world_size
        segments = step.segments or ()
        tokens = sum(segment.span for segment in segments)
        # a packed replay declares its real rows alone; its capture declared the bucket, so the
        # bucket's padding joins the listed label for both to decide and size alike
        pad = 0
        if bucket is not None and ctx.force_prefill:
            pad = bucket.num_tokens - tokens
        # `one_token_rows`: a decode step never splits
        if not ctx.force_prefill and len(ctx.padded_request_ids) == tokens:
            return COPIED
        if self.walks is not None and ctx.graph_walk not in self.walks:
            return COPIED
        spans: dict[str, list[int]] = {}
        for segment in segments:
            if segment.label in self.labels:
                spans.setdefault(segment.label, []).append(segment.span)
        if pad > 0:
            if len(self.labels) > 1 or self.axis != 0:
                raise ValueError(
                    f"a packed replay pads its step, which sequence shard {self.labels} can't place: "
                    "a packed capture needs one listed label and axis 0"
                )
            spans.setdefault(self.labels[0], []).append(pad)
        if sum(map(sum, spans.values())) < max(sp, self.min_tokens):
            return COPIED
        for label, label_spans in spans.items():
            if self.axis == 1 and len(set(label_spans)) > 1:
                raise ValueError(
                    f"sequence shard on axis 1 splits each row of {label!r} alike, so its rows must be "
                    f"one length; this step's are {label_spans}"
                )
            if (min(label_spans) if self.axis == 1 else sum(label_spans)) < sp:
                return COPIED
        if self.stripe_pool is not None:
            return self._striped(spans, step, ctx)
        ranges = {label: self._range(label_spans) for label, label_spans in spans.items()}
        for sources, plan_label in step.combined_labels.items():
            listed = [source in self.labels for source in sources]
            if not any(listed):
                continue
            if not all(listed):
                split = sorted(source for source in sources if source in self.labels)
                raise ValueError(
                    f"combined label {plan_label!r} mixes split sources {split} with copied ones "
                    f"{sorted(set(sources) - set(split))}"
                )
            combined = [span for source in sources for span in spans.get(source, ())]
            if combined:
                if self.axis == 1 and len(set(combined)) > 1:
                    raise ValueError(f"combined label {plan_label!r}'s rows must be one length; got {combined}")
                ranges[plan_label] = self._range(combined)
        return SeqShardPlan(True, ranges)

    def _striped(self, spans: Mapping[str, list[int]], step: SeqShardStep, ctx: StepContext) -> SeqShardPlan:
        """Each listed label's tokens to the rank storing their KV position (`stripe_owners`), from
        each row's stored length in the pool's plan; copied if a rank would get none of a label.
        Every input is the instance's own (the step and the pool's plan), so every rank decides
        alike."""
        if any(source in self.labels for sources in step.combined_labels for source in sources):
            raise ValueError(
                f"sequence shard {self.labels} stripes by KV {self.stripe_pool!r}'s stored positions, which a "
                f"combined label's rows ({sorted(step.combined_labels.values())}) don't have"
            )
        pool = ctx.plan_results.get(self.stripe_pool) or {}
        world = self.group.world_size
        ranges = {}
        for label, label_spans in spans.items():
            plan = pool.get(label)
            if plan is None:
                raise ValueError(
                    f"sequence shard {self.labels} stripes {label!r} by KV {self.stripe_pool!r}'s stored positions, "
                    "and this step plans no such label there"
                )
            indptrs = plan.cpu_indptrs
            qo = indptrs.qo_indptr.numpy()
            lens = np.diff(qo)
            if lens.tolist() != label_spans:
                raise ValueError(
                    f"label {label!r}'s rows are {label_spans} in the step and {lens.tolist()} in KV "
                    f"{self.stripe_pool!r}'s plan"
                )
            owners = stripe_owners(qo, indptrs.kv_lens.numpy() - lens, world)
            counts = np.bincount(owners, minlength=world)
            if not counts.all():
                return COPIED
            order = to_device_nonblocking(np.argsort(owners, kind="stable"), self._device, torch.int64)
            ranges[label] = SeqRange(self.group, tuple(counts.tolist()), None, order)
        return SeqShardPlan(True, ranges)

    def _range(self, spans: list[int]) -> SeqRange:
        world = self.group.world_size
        if self.axis == 0:
            return SeqRange(self.group, tuple(sp_seq_split(sum(spans), world)), None)
        return SeqRange(self.group, tuple(sp_seq_split(spans[0], world)), len(spans))

    ### Submodule-level functionality

    def split_range(self, label: str) -> SeqRange | None:
        """``label``'s ranges if this step splits it; None when copied, unlisted or outside a step."""
        plan = self._current
        if plan is None or not plan.split:
            return None
        return plan.ranges.get(label)

    def scatter(self, label: str, *xs: torch.Tensor | None, dim: int | None = None):
        """This rank's share of each of ``xs`` (whole on every rank) along ``dim`` (the spec's
        axis by default), when this step splits ``label`` (``SeqRange.local``); the inputs
        unchanged otherwise.
        One tensor in, one out; several in, a tuple out. None passes through."""
        if not self.can_split:
            return xs[0] if len(xs) == 1 else xs
        return self._bracket(label, xs, dim, scatter=True)

    def gather(self, label: str, *xs: torch.Tensor | None, dim: int | None = None):
        """``scatter``'s inverse: every rank's share of each of ``xs`` back whole."""
        if not self.can_split:
            return xs[0] if len(xs) == 1 else xs
        return self._bracket(label, xs, dim, scatter=False)

    @torch.compiler.disable
    def _bracket(self, label: str, xs: tuple, dim: int | None, scatter: bool):
        plan = self._current
        if plan is None or not plan.split:
            # copied, or outside a step (a warmup compile)
            return xs[0] if len(xs) == 1 else xs
        r = plan.ranges.get(label)
        if r is None:
            raise ValueError(
                f"this step splits {sorted(plan.ranges)}, not {label!r}: bracket a label the step "
                "declares to the sequence shard"
            )
        dim = self.axis if dim is None else dim
        if scatter:
            self._scattered.add(label)
            out = tuple(None if x is None else r.local(x, dim) for x in xs)
        else:
            self._scattered.discard(label)
            out = tuple(None if x is None else r.whole(x, dim) for x in xs)
        return out[0] if len(out) == 1 else out

    def check_scattered(self, label: str) -> None:
        """Refuse an exchange of split ``label`` before the layer stack scattered it. Checked
        against the label, which every rank shares, so every rank refuses alike before the
        exchange's collectives (each rank's token count can pass on one rank alone)."""
        if label not in self._scattered:
            raise ValueError(
                f"an exchange of label {label!r}, which this step splits over SP ranks, before the layer stack "
                f"scattered it (it scattered {sorted(self._scattered)}): bracket the stack with the sequence "
                "shard's scatter of that label"
            )

    def check_bypass(self, label: str, what: str) -> None:
        """Refuse a paged ``what`` (run or a KV write) of a split label reached outside an
        exchange: it would see this rank's share of the tokens as the whole step."""
        if not self.exchanging and self.split_range(label) is not None:
            raise ValueError(
                f"{what} reached label {label!r}, which this step splits over SP ranks, outside an "
                "exchange: call the attention through AttentionCallable (or the resource's exchange)"
            )


def check_shard_plan(owner: str, shard_key: str, ctx: StepContext) -> SeqShardPlan:
    """The shard plan a dependent of ``shard_key`` reads; raises if the step doesn't declare it."""
    plan = ctx.plan_results.get(shard_key)
    if plan is None:
        raise ValueError(
            f"{owner} depends on sequence shard {shard_key!r}, which this step doesn't declare: "
            "every declare_step branch of its node declares a SeqShardStep"
        )
    return plan


# --- exchanges ---------------------------------------------------------------------------------

# ``run(q, k, v)`` attends (and for a paged resource writes K/V) on the tensors it is handed, each
# packed ``[N, H, D]``; ``q=None`` writes alone. A ``[B, S, ...]`` label (``SeqRange.rows``) is
# exchanged along S, viewed from the packed tensor.


def _check_entry(r: SeqRange, *xs: torch.Tensor | None) -> None:
    # an invariant, not a refusal: counts differ by rank, so a refusal here could stop one rank before
    # the exchange's collectives and hang the others. The callers refuse an unscattered label first
    # (`SequenceShard.check_scattered`), alike on every rank
    assert all(x is None or x.shape[0] == r.local_tokens for x in xs), (
        f"an exchange over a split label got {[None if x is None else x.shape[0] for x in xs]} tokens where this "
        f"rank holds {r.local_tokens}"
    )


def _seq_view(x: torch.Tensor, r: SeqRange) -> torch.Tensor:
    return x if r.rows is None else x.unflatten(0, (r.rows, -1))


def _packed(x: torch.Tensor, r: SeqRange) -> torch.Tensor:
    return x if r.rows is None else x.flatten(0, 1)


def ulysses(run: Callable, q, k, v, r: SeqRange):
    """Split tokens to split heads: one all-to-all of q, k and v (each ``[n, H, D]``, this rank's
    tokens) to ``[N, H / P, D]``, ``run``, and the output back."""
    _check_entry(r, q, k, v)
    seq, head = (0, 1) if r.rows is None else (1, 2)
    parts = [x for x in (q, k, v) if x is not None]
    exchanged = r.group.all_to_all(
        [_seq_view(x, r) for x in parts], scatter_dim=head, gather_dim=seq, gather_sizes=list(r.counts),
    )
    exchanged = [_packed(x, r) for x in exchanged]
    if q is None:
        run(None, *exchanged)
        return None
    out = run(*exchanged)
    out = r.group.all_to_all(_seq_view(out, r), scatter_dim=seq, gather_dim=head, scatter_sizes=list(r.counts))
    return _packed(out, r)


def all_gather(run: Callable, q, k, v, r: SeqRange):
    """Every rank's q, k and v gathered whole in one collective, in the step's order whatever the
    label's layout, ``run`` on them as on a copied step, then this rank's share of the output.
    ``q`` None: k and v alone, for the write."""
    _check_entry(r, q, k, v)
    parts = [x for x in (q, k, v) if x is not None]
    assert r.rows is None and len({x.dtype for x in parts}) == 1, (
        "all_gather takes a packed label's q, k and v of one dtype"
    )
    whole = r.whole(torch.cat([x.flatten(1) for x in parts], dim=-1))
    pieces = whole.split([math.prod(x.shape[1:]) for x in parts], dim=-1)
    gathered = [piece.unflatten(-1, x.shape[1:]).contiguous() for piece, x in zip(pieces, parts, strict=True)]
    if q is None:
        run(None, *gathered)
        return None
    return r.local(run(*gathered))


def head_slice(run: Callable, q, k, v, group: CommGroup):
    """A copied step at a ``tp * sp`` head geometry: this rank's block of q, k and v's heads (the
    block an all-to-all routes it), ``run``, and every rank's heads of the output gathered."""
    p, rank = group.world_size, group.rank

    def mine(x):
        if x is None:
            return None
        heads = x.shape[1] // p
        return x[:, rank * heads:(rank + 1) * heads].contiguous()

    out = run(mine(q), mine(k), mine(v))
    return None if q is None else group.all_gather(out.contiguous(), dim=1)


def sharded_key_merge(run: Callable, q, k, v, r: SeqRange):
    """Copied queries over split keys: ``run`` (which returns ``(out, lse)``) over this rank's
    keys, and the partials merged by log-sum-exp after one gather (``gather_merge``)."""
    _check_entry(r, k, v)
    if q is None:
        return None
    return gather_merge(r.group, *run(q, k, v))


# --- refusals ----------------------------------------------------------------------------------


def sequence_parallel_refusals(
    specs: Iterable[NodeResourceSpec], node: str, tp: int, sp: int,
    moe_parallel: MoeParallelConfig | None = None,
) -> list[str]:
    """What ``node``, on ``tp`` TP and ``sp`` SP ranks, declares that its SP ranks can't run; each
    a sentence. At one SP rank, only an ``sp_exchange`` set there. Each resource answers for itself
    (``NodeResourceSpec.split_step_refusals``); the checks on built objects
    (``supports_split_steps``, ``check_captures``) are the engine's."""
    specs = [spec for spec in specs if node in spec.nodes]
    if sp == 1:
        from mstar.engine.resources.attn.config import AttentionSpec

        # a deployment's sp_exchange would be silently ignored here (`AttentionSpec.derive` refuses it
        # without a shard)
        return [
            f"node {node!r} runs on one SP rank, so its steps never split, and attention {spec.resource_key!r} "
            f"sets sp_exchange {spec.sp_exchange!r}, which picks how a split step reaches it: drop it"
            for spec in specs if isinstance(spec, AttentionSpec) and spec.sp_exchange is not None
        ]
    try:
        placed = any(position_pools(specs, node).values())
    except ValueError as e:
        return [str(e)]
    shard = next((spec for spec in specs if isinstance(spec, SeqShardSpec)), None)
    if shard is None or not shard.splits:
        # as a node without a shard: its steps run copied (`SeqShardSpec.splits`), so a model's
        # shard is a no-op where no attention takes split steps
        if placed:
            return []
        if shard is not None:
            return [
                f"node {node!r} runs on {sp} SP ranks, and no attention there takes split steps (a paged one "
                "with sp_exchange none, a ragged one without sp_exchange 'ulysses'), so its steps never split "
                "and its SP ranks would only share heads: store its KV by position (context_parallel) or run "
                "it on TP ranks"
            ]
        return [
            f"node {node!r} runs on {sp} SP ranks but neither splits tokens (a SeqShardSpec) nor stores "
            "KV by position (context_parallel on its paged KV pools)"
        ]
    refusals = []
    for spec in specs:
        refusals += spec.split_step_refusals(node, shard, tp, sp)
    if moe_parallel is not None and moe_parallel.parallel == "ep":
        refusals.append(f"node {node!r} splits tokens over SP ranks, which expert parallelism (ep_config) can't")
    return refusals

