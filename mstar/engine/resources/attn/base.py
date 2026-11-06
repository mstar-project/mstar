"""The attention resource's shared machinery: the spec-time factory and the
custom op every backend attends through.

The backends themselves live beside this — `flashinfer`, `cross`, `dense`,
`flex` — and `AttentionManager.build` reaches them by deferred import, so
naming a backend in a spec does not load the others.
"""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend, SpExchange
from mstar.engine.resources.attn.workspace import WorkspacePool
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
    PagedKernel,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.context_parallel import (
    OneSlotPages,
    SPContextParallelWrapper,
    SPPlacement,
    TPContextParallelWrapper,
    TPPlacement,
)
from mstar.engine.resources.kv.config import KVConfig, KVGroup, PagedKVConfig, RingKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.sequence_shard import (
    SeqShardPlan,
    SequenceShard,
    all_gather,
    check_shard_plan,
    head_slice,
    ring,
    ulysses,
)
from mstar.platforms import current_platform

if TYPE_CHECKING:
    from mstar.engine.resources.context_parallel import (
        ContextParallel,
        ContextParallelWrapper,
        OwnTokensKernel,
        ShareCapture,
    )

logger = logging.getLogger(__name__)
_BACKEND_KV_CONFIG: dict[AttnBackend, type[KVConfig]] = {
    AttnBackend.DENSE: PagedKVConfig,
    AttnBackend.FLASHINFER: PagedKVConfig,
    AttnBackend.FLASHINFER_MLA: PagedKVConfig,
    AttnBackend.FLEX: RingKVConfig,
    AttnBackend.XPU_PAGED: PagedKVConfig,
}


def resolve_attn_backend(backend: AttnBackend, device: torch.device) -> AttnBackend:
    """The concrete backend for ``backend`` on ``device``.

    ``AUTO`` becomes the platform's default, FlashInfer on every platform for
    now. A backend named in the spec or YAML is returned as is.
    """
    if backend is AttnBackend.AUTO:
        return AttnBackend.FLASHINFER
    return backend


class AttentionManager(AttentionResource):
    prefix_skip_safe = True
    # the sequence shard over its node (`AttentionSpec.derive`), set by `build`; and the shard
    # itself where it can split (`bind_dependencies`)
    _shard_key: str | None = None
    _shard: SequenceShard | None = None
    # how split steps reach it (`AttentionSpec.resolve_exchange`), set by `build`
    _split_exchange: SpExchange = "ulysses"

    @property
    def splits(self) -> bool:
        """Whether this node's steps may split over SP ranks, so a call goes through `exchange`."""
        return self._shard is not None

    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    def depends_on(self):
        keys = {self._kv_cache_name}
        if self._shard_key is not None:
            keys.add(self._shard_key)
        return keys

    def bind_dependencies(self, resources) -> None:
        if self._shard_key is None or not resources[self._shard_key].can_split:
            return
        # stored by head, heads over every TP and SP rank: a split step's tokens trade for heads, a
        # copied step's heads are sliced. Stored by position, heads over TP: every rank attends a
        # copied step, and a split step's tokens are gathered or passed round (`exchange`)
        shard = self._shard = resources[self._shard_key]
        kv = resources[self._kv_cache_name]
        kv.bind_sequence_shard(shard)
        split = self._split_exchange
        if split == "ring":
            # every rank binds the same resources in the same order
            shard.group.connect_pairs(self._device)
        logger.info(
            "attention over KV %r takes split steps over %d SP ranks: split labels by %s, copied ones %s",
            self._kv_cache_name, shard.group.world_size, split,
            "by head slice" if kv.placement is None else "over the placed pool",
        )

    @torch.compiler.disable
    def exchange(self, run, q, k, v, label: str | None = None):
        """``run(q, k, v)`` (a K/V write and the attention, ``q=None`` the write alone) for this
        step's ``label`` through the exchange its pool takes: of a split label's tokens, or of a
        copied label's heads at a ``tp * sp`` geometry (`sequence_shard`)."""
        label = self._default_label if label is None else label
        shard = self._shard
        r = shard.split_range(label)
        # the paged run and K/V writes refuse a call reached outside it (`check_bypass`)
        shard.exchanging = True
        try:
            if r is not None:
                shard.check_scattered(label)
            return self._exchange_over_pool(run, q, k, v, r)
        finally:
            shard.exchanging = False

    def _exchange_over_pool(self, run, q, k, v, r):
        """``exchange``'s ``run`` over a pool stored by head, heads over every TP and SP rank: a split
        step's tokens (range ``r``) trade for heads, a copied step's heads are sliced."""
        return head_slice(run, q, k, v, self._shard.group) if r is None else ulysses(run, q, k, v, r)

    def _check_sequence_shard(self, ctx) -> SeqShardPlan | None:
        """At plan: the step declares the node's sequence shard, whose plan this returns where the
        shard can split."""
        if self._shard_key is None:
            return None
        plan = check_shard_plan(f"attention over KV {self._kv_cache_name!r}", self._shard_key, ctx)
        return plan if self._shard is not None else None

    def fingerprint(self) -> bytes:
        # the class, not the declared backend: `build` moves dense to another
        # kernel where the FA3 one is unusable, and they do not agree bit for bit
        return fingerprint(type(self).__name__)

    @staticmethod
    def _check_window_pool(spec: AttentionSpec, device: torch.device) -> str:
        """The FlashInfer kernel to read a pool that keeps a window with. Only the kernels that
        take any page size read its per-group page sizes and strided pages: FA2, and FA3 on
        SM90 only. ``auto`` picks one of them below SM100, and from SM100 can send short rows
        without a window to cuDNN, so there it becomes FA2. What doesn't depend on the device
        is refused before any build (`derive_kv_groups`)."""
        kernel = spec.config.flashinfer_backend
        capability = current_platform.cuda_capability(device)
        sm100 = capability is not None and capability >= (10, 0)
        # FlashInfer has fa3 for SM90 only
        fa3 = capability is not None and capability[0] == 9
        allowed = ("auto", "fa2", "fa3") if fa3 else ("auto", "fa2")
        if kernel not in allowed:
            raise ValueError(
                f"KV {spec.config.kv_cache!r} keeps a window, whose pages only FlashInfer's fa2 and fa3 "
                f"(SM90 only) kernels read, so flashinfer_backend={kernel!r} can't on this GPU; "
                f"use {' or '.join(allowed)}"
            )
        return "fa2" if sm100 and kernel == "auto" else kernel

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        manager = cls._build_backend(spec, info)
        manager._shard_key = spec.shard_key
        manager._split_exchange = spec.resolve_exchange(info.dependencies)
        return manager

    @classmethod
    def _build_backend(cls, spec: AttentionSpec, info: EngineResourceInfo):
        backend = resolve_attn_backend(spec.config.backend, info.device)
        if spec.config.sliding_window is not None and backend != AttnBackend.FLASHINFER:
            raise ValueError(
                f"sliding_window is unsupported for attention backend {backend!r}; "
                "use the flashinfer backend"
            )
        # the KV resource's own config, not a copy; per-rank head counts, and
        # `shard` is idempotent so both builders can call it
        kv_config = info.dependency(spec.config.kv_cache).config
        windowed = isinstance(kv_config, PagedKVConfig) and len(kv_config.groups) > 1
        window = spec.config.sliding_window
        keeps_window = isinstance(kv_config, PagedKVConfig) and window is not None and window < kv_config.max_seq_len
        if keeps_window and not windowed:
            # the pool derives its window from its attentions, which resolving the specs does
            raise ValueError(
                f"attention {spec.resource_key!r} declares a sliding_window, but KV {spec.config.kv_cache!r} "
                "keeps no window: resolve the specs (resolve_spec_dependencies) before building them"
            )
        flashinfer_backend = spec.config.flashinfer_backend
        if windowed:
            # the backend as resolved for this device; `derive_kv_groups` refuses a declared one
            # earlier, so validation reports it
            if backend != AttnBackend.FLASHINFER:
                raise ValueError(
                    f"KV {spec.config.kv_cache!r} keeps a window, whose pages only the flashinfer "
                    f"backend reads, not attention {spec.resource_key!r}'s {backend.value}"
                )
            flashinfer_backend = cls._check_window_pool(spec, info.device)
            if flashinfer_backend != spec.config.flashinfer_backend:
                logger.info(
                    "attention %r reads KV %r, which keeps a window: flashinfer_backend=%r "
                    "runs as %r on this GPU",
                    spec.resource_key, spec.config.kv_cache, spec.config.flashinfer_backend, flashinfer_backend,
                )
        # Before `shard`, so a mismatch is reported rather than half-applied to
        # a config the KV resource also holds.
        if backend not in _BACKEND_KV_CONFIG:
            raise ValueError(f"Unknown attention backend {backend!r}")
        required = _BACKEND_KV_CONFIG[backend]
        if not isinstance(kv_config, required):
            raise TypeError(
                f"attention backend {backend.value!r} requires a "
                f"{required.__name__}, but KV resource {spec.config.kv_cache!r} "
                f"holds a {type(kv_config).__name__}. "
            )
        if spec.config.sm_scale is not None and backend != AttnBackend.FLASHINFER_MLA:
            raise ValueError(
                f"attention backend {backend.value!r} ignores sm_scale; only "
                f"{AttnBackend.FLASHINFER_MLA.value!r} reads it"
            )
        # `shard` lives on the KVConfig base, so this is unchanged for both.
        kv_config.shard(info.head_shards)
        if isinstance(kv_config, PagedKVConfig) and kv_config.context_parallel:
            # the backends whose partials' lse base the merge is known to read right
            lse_known = ("auto", "fa2", "fa3")
            unsupported = [name for name, refused in (
                ("sliding_window", spec.config.sliding_window is not None),
                # before it degrades to FlashInfer on some hosts, so a config loads on none
                (f"backend {backend.value!r}", backend == AttnBackend.DENSE),
                (f"flashinfer_backend {flashinfer_backend!r}", flashinfer_backend not in lse_known),
            ) if refused]
            if unsupported:
                raise ValueError(
                    f"attention {spec.resource_key!r}: {', '.join(unsupported)} is unsupported over KV pool "
                    f"{spec.config.kv_cache!r}, which sets context_parallel"
                )
        if backend == AttnBackend.DENSE:
            # A dense backend runs on FlashAttention-3 (AITER on ROCm). Where
            # that has no kernel for the GPU or does not import, the same pass
            # runs on FlashInfer's ragged prefill, and where FlashInfer cannot
            # run either it degrades to the paged backend AUTO picks rather
            # than failing to serve.
            # All three are drop-in for each other: the step declaration is
            # identical (see DenseAttentionManager) and `requires_kv_write`
            # tells the layer which one it is talking to.
            from mstar.engine.resources.attn.dense import (
                DenseAttentionManager,
                _dense_unavailable_reason,
                _warn_dense_fallback,
            )
            from mstar.engine.resources.attn.ragged.base import dense_or_flashinfer
            from mstar.engine.resources.attn.ragged.dense_wrappers import DensePrefixedPrefillWrapper
            from mstar.engine.resources.attn.ragged.wrappers import FlashInferPrefixedPrefillWrapper

            reason = _dense_unavailable_reason(info.device)
            if reason is None:
                return DenseAttentionManager(
                    kv_cache=spec.config.kv_cache,
                    device=info.device,
                    dtype=info.kv_dtype,
                    kv_config=kv_config,
                    wrapper_class=dense_or_flashinfer(
                        info.device, DensePrefixedPrefillWrapper, FlashInferPrefixedPrefillWrapper, "Attention",
                    ),
                    flashinfer_backend=spec.config.flashinfer_backend,
                )
            backend = resolve_attn_backend(AttnBackend.AUTO, info.device)
            _warn_dense_fallback(reason, backend)

        if backend == AttnBackend.FLASHINFER_MLA:
            from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager

            return FlashInferMLAManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
                backend=spec.config.flashinfer_backend,
                sm_scale=spec.config.sm_scale,
            )
        if backend == AttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.flashinfer import FlashInferManager

            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
                backend=flashinfer_backend,
                splits=spec.config.splits,
                max_rows_in_combined=info.max_rows_in_combined,
                decode_tensor_cores=spec.config.decode_tensor_cores,
                sliding_window=spec.config.sliding_window,
                decode_fmha=spec.config.decode_fmha,
                layers=spec.config.layers,
            )

        if backend == AttnBackend.FLEX:
            from mstar.engine.resources.attn.flex import FlexAttentionManager

            return FlexAttentionManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
            )

        if backend == AttnBackend.XPU_PAGED:
            from mstar.engine.resources.attn.xpu import (
                XPUPagedAttentionManager,
                _xpu_paged_unavailable_reason,
            )

            if info.device.type != "xpu":
                raise RuntimeError(
                    "attention backend 'xpu_paged' requires an XPU device; "
                    f"got {info.device}"
                )
            reason = _xpu_paged_unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "attention backend 'xpu_paged' requires a working "
                    f"vllm-xpu-kernels installation ({reason})"
                )
            return XPUPagedAttentionManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                kv_config=kv_config,
            )
        raise ValueError(f"Unknown attention backend {backend!r}")

    @property
    def requires_kv_write(self) -> bool:
        """Whether a layer must write this step's K/V through the KV resource
        before calling ``run``.

        False only for backends that take the fresh K/V straight into the
        kernel (the dense one). A layer therefore reads

            if self.attn.requires_kv_write:
                self.kv.write_kv(k, v, layer_idx=i, label=label)
            out = self.attn.run(q, label, kv.layer_view(i), k=k, v=v, layer_idx=i)

        and stays correct whichever backend the spec named. Whether ``run`` also
        reads ``k``/``v`` is ``reads_step_kv``, which a fused write path asks.
        """
        return True

    @torch.compiler.disable
    def reads_step_kv(self, label: str | None = None) -> bool:
        """Whether this step's ``run`` under ``label`` attends the ``k``/``v`` it is passed rather
        than only the pages (whether the layer writes them is ``requires_kv_write``). A layer whose
        fused kernel writes the K/V but leaves ``k`` unrotated rotates it in place first when this
        is True. (Where ``splits``, it doesn't fuse the write at all: the call goes through the
        exchange, which writes this rank's share.)

        Every step of a backend that takes the fresh K/V straight into the kernel. Fixed once the
        step is planned, so a captured bucket replays the same answer.
        """
        return not self.requires_kv_write

    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        """The current plan's query offsets per request, for backends that plan them."""
        raise NotImplementedError(f"{type(self).__name__} plans no query offsets (qo_indptr_buf)")

    # Eager as a whole: a break inside it leaves a compiled frame that holds the
    # bucket's static token count beside its row count, which captured forwards
    # recompile per prefill bucket until Dynamo's recompile limit.
    @torch.compiler.disable
    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        """Each request's last row of ``hidden``, used for sampling from prefill,
        for a backend whose ``qo_indptr_buf`` holds the plan's query offsets."""
        last = (self.qo_indptr_buf(label)[1:] - 1).long()
        return hidden.index_select(0, last)


class PlacedAttention(AttentionManager):
    """What a backend supplies to attend a placed pool, the one object placed attention builds its
    kernels from: its paged kernel, and the kernel a step's own tokens run on, one-slot pages over
    it unless the backend has a better one for the rows. The rest (the split of the own tokens, the
    exchanges, the merge) is ``ContextParallelWrapper``'s."""

    supports_context_parallel = True
    # the KV pool's placement, and the attention it takes (`bind_dependencies`)
    _placement: "ContextParallel | None" = None
    _placed: "type[ContextParallelWrapper] | None" = None

    def __init__(self, kv_cache: str, device: torch.device, dtype: torch.dtype, kv_config: KVConfig):
        self._kv_cache_name, self._device, self._dtype, self._kv_config = kv_cache, device, dtype, kv_config
        self._workspaces = WorkspacePool(device, owner=kv_cache)
        # label to the wrapper it attends on, this step's and a step planned ahead's
        self._current_plan_states: dict = {}
        self._preplan_states: dict = {}
        self._preplanned = False

    def bind_dependencies(self, resources) -> None:
        super().bind_dependencies(resources)
        placement = resources[self._kv_cache_name].placement
        if placement is not None:
            self._placement = placement
            wrappers = {SPPlacement: SPContextParallelWrapper, TPPlacement: TPContextParallelWrapper}
            self._placed = wrappers[type(placement)]
            cfg = self._kv_config
            self._bind_placement(self._placed.local_geometry(placement, cfg.tokens_per_page, cfg.num_qo_heads))

    @abstractmethod
    def _bind_placement(self, geometry: dict[str, int]) -> None:
        """Takes on this rank's ``page_size`` and ``num_qo_heads`` over the pool."""

    @abstractmethod
    def paged_kernel(
        self, workspace: torch.Tensor, decode: bool, kwargs: dict, max_total_tokens: int | None,
    ) -> PagedKernel:
        """The paged kernel over this rank's slots, for a one-token-a-row step or not, from the
        backend's wrapper arguments ``kwargs`` with the placed geometry; or, with ``page_size`` 1,
        over pages of one slot. A captured one's sizes are ``batch_size``, ``max_num_pages`` and
        ``max_total_tokens``."""

    def own_tokens_kernel(
        self, workspace: torch.Tensor, kwargs: dict, captured: "ShareCapture | None", whole: bool,
    ) -> "OwnTokensKernel":
        """The kernel a multi-token step attends its own tokens on, at this rank's own heads: for
        rows ``whole`` (each over its own tokens, TP placement) or not (SP's zigzag share, whose two
        chunks' keys overlap), sized for ``captured``. One-slot pages over ``paged_kernel``, which
        take any rows with no copy."""
        kwargs = {**kwargs, "page_size": 1, "num_qo_heads": self._kv_config.num_qo_heads}
        if captured is None:
            return OneSlotPages(self.paged_kernel(workspace, False, kwargs, None), self._dtype)
        # a row's keys are at most the bucket's, and a zigzag row has two chunks; the padding and
        # leftover rows' one key each fit in the batch_size spare indices the kernel adds
        kwargs.update(batch_size=captured.rows, max_num_pages=2 * captured.tokens)
        return OneSlotPages(self.paged_kernel(workspace, False, kwargs, captured.queries), self._dtype, captured)

    def _context_parallel_wrapper(
        self, label: str, slot: int, decode: bool, kwargs: dict, max_total_tokens: int | None = None,
    ):
        """The ``ContextParallelWrapper`` a step attends a placed pool on: the context's kernel and,
        for a step that isn't one token a row, its own tokens'.

        Both take the label's workspace although both are planned for the one step: a FlashInfer
        wrapper keeps its plan in an int workspace of its own, and uses this one only for the
        split-KV partials of a ``run``, which it merges within that ``run``; the two run one after
        the other on one stream."""
        workspace = self._workspaces.get(label, slot)
        context = self.paged_kernel(workspace, decode, kwargs, max_total_tokens)
        if not isinstance(context, PagedKernel):
            raise ValueError(f"attention over a placed pool takes a PagedKernel, not a {type(context).__name__}")
        new = None if decode else self._placed.own_tokens(
            self, self._placement, self._device, workspace, kwargs, max_total_tokens,
        )
        return self._placed(
            self._placement, self._kv_config.tokens_per_page, self._dtype, context, new, self._device,
            kwargs.get("batch_size"), max_total_tokens,
        )

    def _exchange_over_pool(self, run, q, k, v, r):
        if self._placement is None:
            return super()._exchange_over_pool(run, q, k, v, r)
        # stored by position, heads over TP: every rank attends a copied step, and a split step's
        # tokens are gathered or passed round
        if r is None:
            return run(q, k, v)
        return (ring if self._split_exchange == "ring" else all_gather)(run, q, k, v, r)

    @torch.compiler.disable
    def reads_step_kv(self, label: str | None = None) -> bool:
        # over a placed pool, a step that attends its own tokens apart from the pool
        if self._placement is not None:
            return self._current_plan_states[label or self._default_label].reads_step_kv
        return super().reads_step_kv(label)

    def clear_preplan(self):
        if self._placement is not None:
            # the dropped plan's H2Ds went on the plan stream, which the step that runs instead
            # waits on no event from, so its own-token wrappers must not skip their plans
            for wrapper in self._preplan_states.values():
                wrapper.forget_plan()
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    @torch.compiler.disable
    def _attend_placed(
        self, label: str, q: torch.Tensor, kv_cache_layer: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None,
    ) -> torch.Tensor:
        """``run`` over a placed pool (``ContextParallelWrapper.attend``)."""
        if self._shard is not None:
            self._shard.check_bypass(label, "attention run")
        return self._current_plan_states[label].attend(q, kv_cache_layer, k, v)


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    # the int32 arrays' bytes; the indptr says which pages are whose
    page_indptr: bytes
    page_indices: bytes
    last_page_lens: bytes


class EagerSlotKey(NamedTuple):
    """One eager wrapper — the counterpart of ``CGSlotKey``, slotted for the
    same reason (FlashInfer's plan stages into a pinned buffer per wrapper)."""
    label: str
    slot: int
    # decode and prefill take different wrapper classes
    is_decode: bool = False
    # a ring step's (`RingAttentionWrapper`), beside the label's copied one
    ring: bool = False


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


def captured_page_capacity(
    num_rows: int, num_tokens: int, kv_config: PagedKVConfig, group: KVGroup | None = None,
) -> int:
    """Page ids a captured wrapper's index buffer holds for a bucket of ``num_rows`` rows and
    ``num_tokens`` tokens, in pages of ``group`` (the pool's first by default).

    Per row, since rows that matched one cached prefix all name its pages, and at least the
    bucket's own sink pages. A pool that keeps a window caches no prefix and forks nothing, so
    its rows name at most the pool, or in the window's group a window each, plus the step."""
    group = group or kv_config.groups[0]
    step_pages = -(-num_tokens // group.page_size)
    if group.retention is not None:
        # a page more per row, which a row's share of the step can cross into
        return num_rows * (group.retention.max_pages(group.page_size) + 1) + step_pages
    if len(kv_config.groups) > 1:
        return kv_config.num_units + step_pages + num_rows
    return max(num_rows * kv_config.max_num_pages, step_pages + num_rows)


def one_token_rows(rows: int, tokens: int, force_prefill: bool) -> bool:
    """Whether a step is the decode kernel's kind, every row querying one token: from a captured
    bucket's shape (fixed for its graph) or an eager plan's."""
    return not force_prefill and rows == tokens

