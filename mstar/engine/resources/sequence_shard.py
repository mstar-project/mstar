"""Sequence parallelism: whether a step's tokens split over a node's SP ranks, and how attention
reaches the tokens it needs when they do.

A node's SP ranks share each step one of two ways. Its tokens are **split** (each rank runs the
token-local layers on its contiguous share) or **copied** (every rank runs them all). The sequence
shard resource (``SeqShardSpec``) decides which, per step, from the step the model declares, and
gives each listed label its per-rank ranges. A model brackets its layer stack with
``scatter`` / ``gather`` on that resource; nothing else in model code changes.

The attention resources over the node depend on it (``shard_key``, taken in their specs'
``derive``) and pick an exchange per label from its plan: what their storage needs to attend a
split label's tokens, or a copied label's at the node's head geometry. The exchanges are the
functions at the bottom of this module, each wrapped around the resource's own ``run``:

- ``ulysses``: one all-to-all from tokens to heads, ``run``, and one back (head-stored KV, or a
  cacheless attention declaring ``sp_exchange="ulysses"``); a splitting node can't yet store KV
  by position (``context_parallel``);
- ``head_slice``: a copied step at a ``tp * sp`` head geometry: this rank's heads, ``run``, gather
  the heads;
- ``sharded_key_merge``: copied queries over split keys (cacheless cross-attention): each rank's
  partial with its log-sum-exp, merged.

``sequence_parallel_refusals`` is what a node on SP ranks may not declare; the engine and
``validate_deployment`` both run it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.distributed.communication import CommGroup, gather_sequence, sp_seq_split
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.context_parallel import gather_merge, position_pools
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep, StepContext

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

    ``exchanges=False`` declares that no op on the node mixes a listed label's units (a video
    decoder's independent clips): nothing exchanges them, so the node needs no attention, and a
    rank may get an empty range (fewer units than ranks still splits; ``gather`` takes it).

    The attention and position resources on the node take this as a dependency in their own
    ``derive``, so every ``declare_step`` branch of the node declares a ``SeqShardStep``; at one
    SP rank it plans copied at no cost."""

    labels: tuple[str, ...]
    axis: int
    walks: Collection[str] | None = None
    min_tokens: int = field(default=0, kw_only=True)
    exchanges: bool = field(default=True, kw_only=True)

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
        return True

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
    shards = [other for other in specs.values() if other.splits_steps and other.nodes & spec.nodes]
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
    ``[B, S, ...]`` layout), contiguous in rank order."""

    group: CommGroup
    counts: tuple[int, ...]
    # B of a [B, S, ...] layout split along S (axis 1); None for a packed label (axis 0)
    rows: int | None

    @property
    def start(self) -> int:
        return sum(self.counts[: self.group.rank])

    @property
    def count(self) -> int:
        return self.counts[self.group.rank]

    @property
    def local_tokens(self) -> int:
        """The tokens this rank holds of the label, as a packed tensor's first dim."""
        return self.count * (self.rows or 1)


class SeqShardPlan(NamedTuple):
    """A step's split: whether its tokens split, and each split label's ranges."""

    split: bool
    ranges: Mapping[str, SeqRange]


COPIED = SeqShardPlan(False, {})


class SequenceShard(Resource):
    """Plans each step split or copied (``SeqShardSpec``) and brackets the layer stack with
    ``scatter`` / ``gather``. Holds no per-request state and no device buffers: its plan is
    host-only, from the declared step and the lease, the same on every rank. Off its defaults:
    ``prefix_skip_safe`` (it plans from the step a prefix hit already trimmed), ``supports_preplan``
    (its dependents pre-plan), ``resident_refusal`` (None: a one-token-a-row step, what resident
    planning replays, is never split); it keeps ``supports_context_parallel`` False (it depends on no pool),
    ``can_capture`` True and ``force_double_buffer`` False (nothing staged)."""

    prefix_skip_safe = True
    supports_split_steps = True
    # holds no per-request state, so a dropped request leaves nothing behind
    supports_drop = True

    def resident_refusal(self) -> str | None:
        return None

    def __init__(
        self, labels: tuple[str, ...], axis: int, walks: frozenset[str] | None, min_tokens: int,
        group: CommGroup, exchanges: bool = True,
    ):
        self.labels = labels
        self.axis = axis
        # the walks that may split; None for every walk
        self.walks = walks
        self.min_tokens = min_tokens
        self.group = group
        # whether a resource exchanges the split labels' units, so each rank needs some
        # (`SeqShardSpec.exchanges`)
        self.exchanges = exchanges
        # fixed at bind, so a compiled frame at one SP rank traces the bracket straight through
        self.can_split = group.world_size > 1
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
        return cls(spec.labels, spec.axis, walks, spec.min_tokens, group, exchanges=spec.exchanges)

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
        # an exchange needs every rank to hold tokens; a split nothing exchanges takes empty ranges
        least = sp if self.exchanges else 1
        if sum(map(sum, spans.values())) < max(least, self.min_tokens):
            return COPIED
        for label, label_spans in spans.items():
            if self.axis == 1 and len(set(label_spans)) > 1:
                raise ValueError(
                    f"sequence shard on axis 1 splits each row of {label!r} alike, so its rows must be "
                    f"one length; this step's are {label_spans}"
                )
            if (min(label_spans) if self.axis == 1 else sum(label_spans)) < least:
                return COPIED
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
        axis by default), a view, when this step splits ``label``; the inputs unchanged otherwise.
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
            out = tuple(None if x is None else x.narrow(dim, r.start, r.count) for x in xs)
        else:
            self._scattered.discard(label)
            out = tuple(None if x is None else gather_sequence(self.group, x, r.counts, dim) for x in xs)
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
    tokens) to ``[N, H / P, D]``, ``run``, and the output back.

    Eager, as main's exchange is (`CommGroup.all_to_all`): a traced variant measured no faster.
    """
    _check_entry(r, q, k, v)
    all_to_all = CommGroup.all_to_all
    seq, head = (0, 1) if r.rows is None else (1, 2)
    parts = [x for x in (q, k, v) if x is not None]
    exchanged = all_to_all(
        r.group, [_seq_view(x, r) for x in parts], scatter_dim=head, gather_dim=seq, gather_sizes=list(r.counts),
    )
    exchanged = [_packed(x, r) for x in exchanged]
    if q is None:
        run(None, *exchanged)
        return None
    out = run(*exchanged)
    out = all_to_all(r.group, _seq_view(out, r), scatter_dim=seq, gather_dim=head, scatter_sizes=list(r.counts))
    return _packed(out, r)


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
    a sentence. Nothing at one SP rank. Each resource answers for itself
    (``NodeResourceSpec.split_step_refusals``); the checks on built objects
    (``supports_split_steps``, ``check_captures``) are the engine's."""
    if sp == 1:
        return []
    specs = [spec for spec in specs if node in spec.nodes]
    try:
        placed = any(position_pools(specs, node).values())
    except ValueError as e:
        return [str(e)]
    shard = next((spec for spec in specs if spec.splits_steps), None)
    if shard is None:
        if placed:
            return []
        return [
            f"node {node!r} runs on {sp} SP ranks but neither splits tokens (a SeqShardSpec) nor stores "
            "KV by position (context_parallel on its paged KV pools)"
        ]
    refusals = []
    attends = any(spec.attends_split_steps for spec in specs)
    if shard.exchanges and not attends:
        refusals.append(
            f"node {node!r} splits tokens (SeqShardSpec {shard.resource_key!r}) but declares no attention that "
            "can take them: a paged AttentionSpec, or a RaggedAttentionSpec with sp_exchange"
        )
    if not shard.exchanges and attends:
        refusals.append(
            f"node {node!r} declares SeqShardSpec {shard.resource_key!r} with exchanges=False, which can hand a "
            "rank no tokens, beside an attention that exchanges split tokens"
        )
    for spec in specs:
        refusals += spec.split_step_refusals(node, shard, tp, sp)
    if moe_parallel is not None and moe_parallel.parallel == "ep":
        refusals.append(f"node {node!r} splits tokens over SP ranks, which expert parallelism (ep_config) can't")
    return refusals

