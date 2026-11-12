"""The cacheless attention resource: the spec-time factory, and the plan
machinery its backends share.

Sibling of ``attn.base``'s ``AttentionManager``, for attention that is not
backed by a KV cache, e.g., an encoder tower attending within variable-length
segments of one packed forward. There is nothing to page, nothing to advance,
and nothing to carry between steps: the resource holds only planned wrappers.
"""

import functools
import itertools
import logging
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import torch

from mstar.engine.resources.attn.base import EagerSlotKey
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.attn.ragged.kernels import RAGGED_ATTENTION, cudnn_rows, kernel_rows
from mstar.engine.resources.attn.ragged.wrappers import _RaggedPrefillBase
from mstar.engine.resources.base import AttentionResource, CGSlotKey, EngineResourceInfo, head_shards
from mstar.engine.resources.sequence_shard import (
    SeqShardPlan,
    SequenceShard,
    check_shard_plan,
    head_slice,
    sharded_key_merge,
    ulysses,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.kernels.registry import describe_rows
from mstar.kernels.table import Choice, Rows

logger = logging.getLogger(__name__)


def resolve_ragged_attn_backend(
    backend: RaggedAttnBackend, device: torch.device,
) -> RaggedAttnBackend:
    """Ragged counterpart of ``attn.base.resolve_attn_backend``. ``AUTO`` is
    FlashInfer's manager; for self-attention, the ``ragged_attention`` kernel
    table then picks the kernel it runs (``kernels``)."""
    if backend is RaggedAttnBackend.AUTO:
        return RaggedAttnBackend.FLASHINFER
    return backend


def _refuse_cudnn(spec, backend: RaggedAttnBackend, what: str) -> RaggedAttnBackend:
    if backend is RaggedAttnBackend.CUDNN:
        raise ValueError(
            f"ragged attention {spec.resource_key!r}: backend 'cudnn' runs self-attention only, "
            f"not {what}; name 'auto', 'flashinfer' or 'dense'"
        )
    return backend


def _build_ragged(
    spec, info: EngineResourceInfo, manager_cls, kernel_table: bool = False, cudnn: bool = False, **kwargs,
):
    """``kernel_table``: read the ``ragged_attention`` rows for the rank's head
    geometry (``kernels.kernel_rows``) and hand them to the manager. ``cudnn``:
    hand it cuDNN's one row, refused here if cuDNN can't run the geometry."""
    config = spec.config
    joint = info.joint_comm_group
    # Ulysses hands each rank 1/(TP x SP) of the heads, whatever the node's KV pools store by (its own
    # exchange, not theirs); any other ragged attention, or one whose node never splits
    # (`SeqShardSpec.splits`, so it runs copied), splits them over TP
    ulysses = (
        spec.sp_exchange == "ulysses" and spec.shard_key is not None and info.dependency(spec.shard_key).splits
    )
    if joint is None:
        config.shard(1)
    else:
        config.shard(head_shards(joint) if ulysses else joint.tp_group.world_size)
    dtype = config.dtype if config.dtype is not None else info.kv_dtype
    if dtype is None:
        raise ValueError(
            f"ragged attention {spec.resource_key!r}: no activation dtype. Set "
            "RaggedAttentionConfig.dtype on a node without a KV-backed attention "
            "(the engine has no KV dtype to fall back on)."
        )
    if cudnn:
        kwargs["kernel_rows"] = cudnn_rows(
            spec.resource_key, config.num_qo_heads, config.num_kv_heads, config.head_dim, dtype, info.device,
        )
    elif kernel_table:
        rows = kernel_rows(config.num_qo_heads, config.num_kv_heads, config.head_dim, dtype, info.device)
        if rows is not None:
            kwargs["kernel_rows"] = rows
    manager = manager_cls(
        device=info.device, dtype=dtype, config=config, key=spec.resource_key, **kwargs,
    )
    manager._shard_key = spec.shard_key
    manager._ulysses = ulysses
    return manager


def _uses_kernel_table(config: RaggedAttentionConfig) -> bool:
    """A resource that names neither a backend nor a FlashInfer kernel runs the table's."""
    return config.backend is RaggedAttnBackend.AUTO and config.flashinfer_backend is None


def dense_or_flashinfer(device: torch.device, dense, flashinfer, what: str):
    """``dense`` where FA3 (AITER on ROCm) has a varlen kernel for ``device``,
    else ``flashinfer``. The one place that choice is made, for the ragged
    managers and the KV-backed dense backend's wrappers alike; ``what`` names
    the caller in the log line."""
    from mstar.engine.resources.attn.dense import _fa3_unavailable_reason
    from mstar.engine.resources.attn.ragged.dense import _warn_dense_on_flashinfer

    reason = _fa3_unavailable_reason(device)
    if reason is None:
        return dense
    _warn_dense_on_flashinfer(what, reason)
    return flashinfer


def _build_dense_or_flashinfer(spec, info: EngineResourceInfo, dense_cls, flashinfer_cls):
    return _build_ragged(
        spec, info, dense_or_flashinfer(info.device, dense_cls, flashinfer_cls, "Ragged attention"),
    )


class RaggedAttnManager(AttentionResource):
    prefix_skip_safe = True

    # Remains abstract except for build; will build based on the backend.

    @classmethod
    def build(cls, spec: RaggedAttentionSpec, info: EngineResourceInfo):
        backend = resolve_ragged_attn_backend(spec.config.backend, info.device)
        if backend is RaggedAttnBackend.DENSE:
            from mstar.engine.resources.attn.ragged.dense import DenseRaggedManager
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedManager,
            )

            return _build_dense_or_flashinfer(
                spec, info, DenseRaggedManager, FlashInferRaggedManager,
            )
        if backend is RaggedAttnBackend.CUDNN:
            from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager

            return _build_ragged(spec, info, FlashInferRaggedManager, cudnn=True)
        if backend is RaggedAttnBackend.FLASHINFER:
            # Deferred so naming the spec does not load FlashInfer, and because
            # `flashinfer` imports this class back.
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedManager,
            )

            return _build_ragged(spec, info, FlashInferRaggedManager, kernel_table=_uses_kernel_table(spec.config))
        raise ValueError(f"Unknown ragged attention backend {backend!r}")


class RaggedCrossAttnManager(AttentionResource):
    """Cacheless cross-attention between two spans of each request; see
    ``RaggedCrossAttentionSpec``."""

    prefix_skip_safe = True

    @classmethod
    def build(cls, spec: RaggedCrossAttentionSpec, info: EngineResourceInfo):
        backend = _refuse_cudnn(spec, resolve_ragged_attn_backend(spec.config.backend, info.device), "cross-attention")
        if backend is RaggedAttnBackend.DENSE:
            from mstar.engine.resources.attn.ragged.dense import DenseRaggedCrossManager
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedCrossManager,
            )

            return _build_dense_or_flashinfer(
                spec, info, DenseRaggedCrossManager, FlashInferRaggedCrossManager,
            )
        if backend is RaggedAttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedCrossManager,
            )

            return _build_ragged(spec, info, FlashInferRaggedCrossManager)
        raise ValueError(f"Unknown ragged cross attention backend {backend!r}")


class RaggedBlockCausalAttnManager(AttentionResource):
    """Cacheless block-causal self-attention within each span; see
    ``RaggedBlockCausalAttentionSpec``."""

    prefix_skip_safe = True

    @classmethod
    def build(cls, spec: RaggedBlockCausalAttentionSpec, info: EngineResourceInfo):
        backend = _refuse_cudnn(spec, resolve_ragged_attn_backend(spec.config.backend, info.device), "block-causal")
        if backend is RaggedAttnBackend.DENSE:
            from mstar.engine.resources.attn.ragged.dense import _warn_ragged_dense_fallback

            # a block's keys are a prefix of its span, and the prefixes overlap:
            # varlen's one key range per query segment cannot express that
            _warn_ragged_dense_fallback("it has no block-causal kernel", RaggedAttnBackend.FLASHINFER)
            backend = RaggedAttnBackend.FLASHINFER
        if backend is RaggedAttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.ragged.flashinfer import (
                FlashInferRaggedBlockCausalManager,
            )

            return _build_ragged(
                spec, info, FlashInferRaggedBlockCausalManager, block_size=spec.block_size,
            )
        raise ValueError(f"Unknown ragged block-causal attention backend {backend!r}")


# Live managers by op key, which is how the custom op below names one: an op
# takes tensors and scalars, not objects. The op key is the resource key plus a
# number unique in the process, so two engines in one process (tests,
# harnesses) never reach each other's plans. Weak, so an entry goes with the
# engine's last reference to it.
_MANAGERS: weakref.WeakValueDictionary[str, "_RaggedManagerBase"] = weakref.WeakValueDictionary()
_MANAGER_NUMBERS = itertools.count()


@torch.library.custom_op("mstar::ragged_attention", mutates_args=())
def ragged_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key: str, label: str | None,
) -> torch.Tensor:
    """``_MANAGERS[key].run(q, k, v, label)`` as one node of a compiled graph.

    Contract: returns a fresh tensor shaped like ``q`` in the manager's dtype,
    and mutates no argument. Every backend's wrapper output is a new allocation,
    or a clone of the graph-mode buffer the next call overwrites, so it aliases
    neither an input nor anything a later call writes.
    """
    return _MANAGERS[key]._run(q, k, v, label)


@ragged_attention.register_fake
def _(q, k, v, key, label):
    return q.new_empty(q.shape, dtype=_MANAGERS[key]._dtype)


class _RaggedManagerBase:
    """What the self- and cross-attention managers share: persistent wrappers per
    (bucket, slot, label) and per (label, slot), the preplan protocol, and ``run``.
    A subclass names its wrapper class and lays its step out in ``_plan_layouts``;
    extra constructor args (a workspace, a kernel backend) go in ``_build_wrapper``."""

    wrapper_class: type[_RaggedPrefillBase]
    # the sequence shard over its node (`RaggedAttentionSpec.derive`), and whether this
    # resource's heads are split over SP for Ulysses; set by `_build_ragged`
    _shard_key: str | None = None
    _ulysses = False
    # the shard, when it can split (`bind_dependencies`)
    _shard: SequenceShard | None = None

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        key: str,
    ):
        self._key = f"{key}#{next(_MANAGER_NUMBERS)}"
        _MANAGERS[self._key] = self
        self._config = config
        self._device = device
        self._dtype = dtype

        # label -> the wrapper this step's `run` attends through
        self._current_plan_states: dict[str, _RaggedPrefillBase] = {}

        # Persistent, because constructing one allocates the kernel's own
        # buffers and is far from free per step.
        self._eager_plan_states: dict[EagerSlotKey, _RaggedPrefillBase] = {}
        self._cg_plan_states: dict[CGSlotKey, _RaggedPrefillBase] = {}

        self._preplan_states: dict[str, _RaggedPrefillBase] = {}
        self._preplanned = False

        self._kwargs = dict(
            device=device,
            q_data_type=dtype,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            sm_scale=config.sm_scale,
        )

    def depends_on(self) -> set[str]:
        return set() if self._shard_key is None else {self._shard_key}

    def bind_dependencies(self, resources) -> None:
        if self._shard_key is None or not resources[self._shard_key].can_split:
            return
        self._shard = resources[self._shard_key]
        logger.info(
            "%s %r takes split steps over %d SP ranks%s", type(self).__name__, self._key,
            self._shard.group.world_size, " (Ulysses)" if self._ulysses else "",
        )

    @property
    def max_segments_per_request(self) -> int:
        """Segments one request may bring to a captured plan, as configured
        (after YAML overrides); a captured wrapper holds this many per row."""
        return self._config.max_segments_per_request

    @property
    def max_segments_per_step(self) -> int | None:
        """Segments a one-row replay's plan holds for its whole step, as
        configured; None for a resource sized per request."""
        return self._config.max_segments_per_step

    @property
    def force_double_buffer(self):
        # plan H2Ds each layout into static buffers that a captured replay reads
        return True

    def _build_wrapper(self, label: str, slot: int, **wrapper_kwargs) -> _RaggedPrefillBase:
        """A new wrapper of the manager's own kernel for one (label, slot).
        ``wrapper_kwargs`` size a captured one (``_graph_kwargs``) or override one
        of the kernel's defaults (a FlashInfer backend)."""
        return self.wrapper_class(**self._kwargs, **wrapper_kwargs)

    def _graph_kwargs(self, bucket: BucketKey) -> dict:
        """What sizes a captured wrapper for ``bucket``. Its static buffers are
        fixed for its lifetime, so they are sized off the config's segment
        ceiling at the bucket's batch size (``max_segments_for``) and the
        bucket's token count — NOT off the first plan's ``num_rows``, which is
        only one of the layouts the bucket will replay."""
        # the bucket's own token count is the ceiling; the per-request override
        # is for a runner that buckets by batch size alone
        max_tokens = max(bucket.num_tokens, self._config.max_tokens_for(bucket.bs) or 0)
        return dict(
            accelerator_graph=True,
            max_num_segments=self._config.max_segments_for(bucket.bs),
            max_total_tokens=max_tokens,
        )

    def _cg_wrapper(self, lease: SlotLease, label: str) -> _RaggedPrefillBase:
        """The captured-graph wrapper for one (bucket, slot, label), built on the
        first plan for that key, like the paged backend's."""
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is None:
            wrapper = self._cg_plan_states[key] = self._build_wrapper(
                label, lease.slot, **self._graph_kwargs(lease.bucket),
            )
        return wrapper

    def _eager_wrapper(self, label: str, slot: int) -> _RaggedPrefillBase:
        """The persistent eager wrapper for one (label, slot).

        Slotted for the same reason the captured ones are, and on the same
        workspace where the backend has one; see `FlashInferManager._eager_wrapper`.
        """
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            wrapper = self._eager_plan_states[key] = self._build_wrapper(label, slot)
        return wrapper

    def _wrapper_for(self, lease: SlotLease | None, slot: int, label: str) -> _RaggedPrefillBase:
        if lease is not None:
            return self._cg_wrapper(lease, label)
        return self._eager_wrapper(label, slot)

    @property
    def supports_preplan(self):
        return True

    @staticmethod
    def _group_segments_by_label(
        segments: Iterable[Segment],
    ) -> dict[str, list[Segment]]:
        res: dict[str, list[Segment]] = {}
        for seg in segments:
            res.setdefault(seg.label, []).append(seg)
        return res

    @staticmethod
    def _cu_seqlens(segments: list[Segment]) -> torch.Tensor:
        # On the CPU: the plans want it there anyway, and building it on
        # device would sync (fatally so under preplan). Pinned and freshly
        # built per plan: the graph wrapper hands this straight to a
        # non-blocking H2D, so a reused buffer could be overwritten while its
        # DMA is still in flight. A fresh pinned allocation is held by the
        # caching host allocator until the copy retires, keeping the plan async
        # without a per-step sync.
        cu = [0]
        for seg in segments:
            cu.append(cu[-1] + seg.span)
        return torch.tensor(cu, dtype=torch.int32, pin_memory=torch.cuda.is_available())

    def plan(self, step, ctx: StepContext):
        self.reset_default_cursors()

        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a captured graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "ragged attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )

        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # layout; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan \
            else self._current_plan_states

        # A label's segments are its whole layout — there is no cache to read
        # them off, so an undeclared label is unattendable. Clear rather than
        # update, so last step's wrapper can't be attended through by mistake.
        plan_states.clear()
        shard_plan = None
        if self._shard_key is not None:
            shard_plan = check_shard_plan(f"ragged attention {self._key!r}", self._shard_key, ctx)
            if self._shard is None:
                # one SP rank: nothing splits
                shard_plan = None
        self._plan_layouts(step, ctx, plan_states, shard_plan)

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # The dropped plan's H2Ds went on the plan stream, and the step that runs
        # instead waits on no event from it, so it must stage its layout itself.
        for wrapper in self._preplan_states.values():
            wrapper.forget_plan()
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality

    def num_segments(self, label: str | None = None) -> int:
        """Real (unpadded) segment count this step planned under ``label``."""
        return self._wrapper(label).num_segments

    def _plan_layouts(
        self, step, ctx: StepContext, plan_states: dict[str, _RaggedPrefillBase], shard_plan: SeqShardPlan | None,
    ) -> None:
        """Plan each label's wrapper into ``plan_states``, against ``shard_plan`` (this step's split,
        None at one SP rank) where the node's sequence shard can split."""
        raise NotImplementedError

    def _exchange(self, label: str) -> Callable | None:
        """The exchange ``label``'s ``run`` goes through on this step, ``(run, q, k, v)``, read off
        the shard's current plan (as the paged ``AttentionManager.exchange`` does); None for none."""
        raise NotImplementedError

    def _wrapper(self, label: str | None) -> _RaggedPrefillBase:
        if label is None:
            label = self._default_label
        wrapper = self._current_plan_states.get(label)
        if wrapper is None:
            raise KeyError(
                f"ragged attention has no plan for label {label!r}; this step "
                f"planned {sorted(self._current_plan_states)}. Every label a "
                "forward attends must carry a segment in the step declaration."
            )
        return wrapper

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        label: str | None = None, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One layer's attention through the plan for ``label``; with ``return_lse``
        also the log-sum-exp ``[total_tokens, H]`` (fp32, natural log), for merging
        attentions over disjoint key sets. The plan is the same either way.

        Traced by ``torch.compile`` with ``config.compile_op``, this is the
        ``mstar::ragged_attention`` op: the region keeps one graph, and Inductor
        fuses the pointwise work on either side of the attention. Otherwise, and
        with ``return_lse`` (the op returns only the output), a compiled caller
        breaks here, and an eager one calls the kernel directly.
        """
        if self._config.compile_op and not return_lse and torch.compiler.is_compiling():
            return ragged_attention(q, k, v, self._key, label)
        return self._run(q, k, v, label, return_lse)

    @torch.compiler.disable
    def _run(self, q, k, v, label, return_lse=False):
        if label is None:
            label = self._default_label
        wrapper = self._wrapper(label)
        exchange = None if self._shard is None else self._exchange(label)
        if exchange is None:
            if return_lse:
                return wrapper.run(q, k, v, return_lse=True)
            return wrapper.run(q, k, v)
        if return_lse:
            raise ValueError(
                f"ragged attention {self._key!r}: label {label!r} reaches its kernel through an exchange over SP "
                "ranks this step (a split label, or any label of a Ulysses resource), which returns the "
                "attention without its log-sum-exp"
            )
        return exchange(wrapper.run, q, k, v)


def _flashinfer_wrapper(manager, label: str, slot: int, config, **graph_kwargs) -> _RaggedPrefillBase:
    """The FlashInfer manager's own wrapper, on the row's ``backend`` (the
    manager's ``flashinfer_backend`` without one)."""
    backend = (config or {}).get("backend")
    return manager._build_wrapper(label, slot, **({} if backend is None else {"backend": backend}), **graph_kwargs)


def _cudnn_wrapper(manager, label: str, slot: int, config, **graph_kwargs) -> _RaggedPrefillBase:
    from mstar.engine.resources.attn.ragged.cudnn_wrappers import CudnnRaggedPrefillWrapper

    return CudnnRaggedPrefillWrapper(**manager._kwargs, **graph_kwargs)


@dataclass(frozen=True)
class WrapperFactory:
    """How a self-attention manager builds one implementation's wrapper for a
    ``ragged_attention`` row: ``build(manager, label, slot, row config, **graph_kwargs)``."""

    build: Callable[..., _RaggedPrefillBase]
    # Runs only a layout of equal-length segments, and once captured only the
    # layout it was first planned with (the graph bakes it in).
    layout_bound: bool = False


# A row's ``impl`` -> its wrapper. A new implementation adds its factory here and
# its ``Impl`` (with what it can't run) to ``kernels.RAGGED_ATTENTION``.
WRAPPER_FACTORIES: Mapping[str, WrapperFactory] = MappingProxyType({
    "flashinfer": WrapperFactory(_flashinfer_wrapper),
    "cudnn": WrapperFactory(_cudnn_wrapper, layout_bound=True),
})

# What a layout runs where its row can't take it, and what a resource without a
# table runs: FlashInfer at the resource's ``flashinfer_backend``. A table bounds
# its measured sizes with a ``"*"`` row naming it.
DEFAULT_ROW = Choice("flashinfer")


def uniform_lengths(lens: Sequence[int]) -> bool:
    """Every non-empty segment has one length and the empty ones come last."""
    n = len(lens)
    while n and lens[n - 1] == 0:
        n -= 1
    return n > 0 and all(length == lens[0] for length in lens[:n])


class _EagerRowKey(NamedTuple):
    """An eager wrapper of a table-reading manager: one per (label, slot) and
    implementation (a row's ``Choice.describe()``), so a slot can run another
    kernel at its next plan."""

    label: str
    slot: int
    impl: str


class _RaggedSelfManagerBase(_RaggedManagerBase, RaggedAttnManager):
    """Self-attention: every label in the step is its own layout, each segment
    attending within itself.

    With ``kernel_rows`` (a ``ragged_attention`` table's, or the ``cudnn``
    backend's one row), a layout runs the row its longest query segment falls
    in, through that row's ``WRAPPER_FACTORIES`` entry. A layout-bound row takes
    only equal-length segments, and a captured bucket only from a step that
    declares ``AttentionStep.fixed_layout``; otherwise the layout runs
    ``DEFAULT_ROW``. Eagerly every plan picks its row. A captured bucket picks
    once, at the plan that builds its wrapper: a graph can't switch kernels.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        key: str,
        kernel_rows: Rows | None = None,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._kernel_rows = kernel_rows
        self._eager_row_states: dict[_EagerRowKey, _RaggedPrefillBase] = {}
        if kernel_rows is None:
            return
        unknown = sorted({choice.impl for _, choice in kernel_rows} - WRAPPER_FACTORIES.keys())
        if unknown:
            raise ValueError(f"ragged attention {key!r}: rows name {unknown}, which have no wrapper here")
        logger.info("ragged attention %r (%d/%d heads, head dim %d, %s): %s", key, config.num_qo_heads,
                    config.num_kv_heads, config.head_dim, str(dtype).removeprefix("torch."),
                    describe_rows(kernel_rows))

    supports_split_steps = True

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        super().plan(step, ctx)

    def _plan_layouts(self, step: AttentionStep, ctx: StepContext, plan_states, shard_plan) -> None:
        for label, segments in self._group_segments_by_label(step.segments or ()).items():
            cu_seqlens = self._cu_seqlens(segments)
            if self._kernel_rows is None:
                wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            else:
                wrapper = self._table_wrapper(ctx.slot_lease, ctx.slot, label, [seg.span for seg in segments], step)
            wrapper.plan(cu_seqlens=cu_seqlens, causal=step.causal)
            plan_states[label] = wrapper
            if shard_plan is not None and label in shard_plan.ranges and not self._ulysses:
                raise ValueError(
                    f"ragged attention {self._key!r} doesn't list label {label!r}, which this step splits over "
                    "SP ranks: list it in its spec's labels"
                )

    def _exchange(self, label: str) -> Callable | None:
        r = self._shard.split_range(label)
        if r is not None:
            self._shard.check_scattered(label)
            return functools.partial(ulysses, r=r)
        return functools.partial(head_slice, group=self._shard.group) if self._ulysses else None

    def _table_wrapper(
        self, lease: SlotLease | None, slot: int, label: str, spans: list[int], step: AttentionStep,
    ) -> _RaggedPrefillBase:
        if lease is not None:
            key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
            wrapper = self._cg_plan_states.get(key)
            if wrapper is None:
                choice = self._row_for(spans, layout_bound_ok=step.fixed_layout)
                wrapper = self._cg_plan_states[key] = self._build_row(
                    choice, label, lease.slot, **self._graph_kwargs(lease.bucket),
                )
            return wrapper
        choice = self._row_for(spans, layout_bound_ok=True)
        key = _EagerRowKey(label=label, slot=slot, impl=choice.describe())
        wrapper = self._eager_row_states.get(key)
        if wrapper is None:
            wrapper = self._eager_row_states[key] = self._build_row(choice, label, slot)
        return wrapper

    def _build_row(self, choice: Choice, label: str, slot: int, **graph_kwargs) -> _RaggedPrefillBase:
        return WRAPPER_FACTORIES[choice.impl].build(self, label, slot, choice.config, **graph_kwargs)

    def _row_for(self, spans: list[int], layout_bound_ok: bool) -> Choice:
        """The row a layout of ``spans`` runs. ``layout_bound_ok``: its wrapper may
        bake the layout in (eager, or a captured bucket whose step fixes it)."""
        choice = RAGGED_ATTENTION.row_for(self._kernel_rows, max(spans, default=0))
        if WRAPPER_FACTORIES[choice.impl].layout_bound and not (layout_bound_ok and uniform_lengths(spans)):
            return DEFAULT_ROW
        return choice


class _RaggedCrossManagerBase(_RaggedManagerBase, RaggedCrossAttnManager):
    """Cross-attention: each ``(q_label, kv_label)`` pair of the step is one plan,
    run under ``cross_label(q_label, kv_label)``."""

    supports_split_steps = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # plan label -> its pair's key label, a fixed fact of the label, for `_exchange`
        self._kv_labels: dict[str, str] = {}

    def _plan_layouts(self, step: RaggedCrossAttentionStep, ctx: StepContext, plan_states, shard_plan) -> None:
        by_label = self._group_segments_by_label(step.segments or ())
        split = shard_plan is not None and shard_plan.split
        for q_label, kv_label in step.pairs:
            q_segments, kv_segments = by_label[q_label], by_label[kv_label]
            if [s.request_id for s in q_segments] != [s.request_id for s in kv_segments]:
                raise ValueError(
                    f"ragged cross-attention {q_label!r} <- {kv_label!r}: the two labels "
                    "must be declared for the same requests in the same order"
                )
            label = cross_label(q_label, kv_label)
            self._kv_labels[label] = kv_label
            if split:
                # a pair with both sides split is refused at load (`split_read_problems`)
                q_range, kv_range = shard_plan.ranges.get(q_label), shard_plan.ranges.get(kv_label)
                assert q_range is None or kv_range is None, (
                    f"cross-attention {q_label!r} <- {kv_label!r} splits both sides, which no exchange serves"
                )
                if q_range is not None:
                    # each rank's queries over every key, each replicated; load refuses a packed
                    # (axis 0) split, so each row keeps ``count`` of its queries
                    q_segments = [s._replace(span=q_range.count) for s in q_segments]
                elif kv_range is not None:
                    kv_segments = [s._replace(span=kv_range.count) for s in kv_segments]
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            wrapper.plan(self._cu_seqlens(q_segments), self._cu_seqlens(kv_segments))
            plan_states[label] = wrapper

    def _exchange(self, label: str) -> Callable | None:
        # split queries attend their rank's share locally; split keys merge every rank's partial
        # (load refuses both split)
        kv_label = self._kv_labels[label]
        r = self._shard.split_range(kv_label)
        if r is None:
            return None
        self._shard.check_scattered(kv_label)
        return functools.partial(_merged_over_keys, r=r)


def _merged_over_keys(run, q, k, v, r):
    """``sharded_key_merge`` over a ragged wrapper's ``run``, asked for its log-sum-exp."""
    return sharded_key_merge(functools.partial(run, return_lse=True), q, k, v, r)
