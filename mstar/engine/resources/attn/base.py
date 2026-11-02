"""The attention resource's shared machinery: the spec-time factory, the
custom op every backend attends through, and the workspace pool.

The backends themselves live beside this — `flashinfer`, `cross`, `dense`,
`flex` — and `AttentionManager.build` reaches them by deferred import, so
naming a backend in a spec does not load the others.
"""

import logging
import os
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend, SpExchange
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.kv.config import KVConfig, KVGroup, PagedKVConfig, RingKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.sequence_shard import (
    SeqRange,
    SeqShardPlan,
    SequenceShard,
    all_gather,
    check_shard_plan,
    head_slice,
    ulysses,
)
from mstar.platforms import current_platform

if TYPE_CHECKING:
    from mstar.engine.resources.context_parallel import ContextParallel

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
    # the KV pool's placement (`bind_dependencies`)
    _placement: "ContextParallel | None" = None
    # a backend's refusal of a placed plan its planner may refuse on one rank alone
    # (`ContextParallelWrapper`'s ``check``)
    _plan_check = None
    # the sequence shard over its node (`AttentionSpec.derive`), set by `build`; and the shard
    # itself where it can split (`bind_dependencies`)
    _shard_key: str | None = None
    _shard: SequenceShard | None = None
    # how split steps reach it (`AttentionSpec.split_exchange`), set by `build`
    _split_exchange: SpExchange = "ulysses"
    # the KV resource, which the ring writes and reads itself (`exchange`)
    _kv = None

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
        kv = resources[self._kv_cache_name]
        placement = kv.placement
        if placement is not None:
            self._placement = placement
            self._bind_placement(placement)
        if self._shard_key is None or not resources[self._shard_key].can_split:
            return
        # stored by head, heads over every TP and SP rank: a split step's tokens trade for heads, a
        # copied step's heads are sliced. Stored by position, heads over TP: every rank attends a
        # copied step, and a split step's tokens are gathered or passed round (`exchange`)
        shard = self._shard = resources[self._shard_key]
        self._kv = kv
        kv.bind_sequence_shard(shard)
        split = self._split_exchange
        if split == "ring":
            self._warm_ring(shard.group)
        logger.info(
            "attention over KV %r takes split steps over %d SP ranks: split labels by %s, copied ones %s",
            self._kv_cache_name, shard.group.world_size, split,
            "by head slice" if placement is None else "over the placed pool",
        )

    def _warm_ring(self, group) -> None:
        """One one-element hop per ring offset, at bind: NCCL connects a pair of ranks on its first
        exchange, which here comes before warmup and the capture plan read free memory, and a
        routing fault hangs at startup rather than in a request. Every rank binds the same
        resources in the same order."""
        p = group.world_size
        one = torch.zeros(1, device=self._device)
        for i in range(1, p):
            to, back = [0] * p, [0] * p
            to[(group.rank + i) % p] = back[(group.rank - i) % p] = 1
            group.all_to_all_rows(one, to, back)

    @torch.compiler.disable
    def exchange(self, run, q, k, v, label: str | None = None):
        """``run(q, k, v)`` (a K/V write and the attention, ``q=None`` the write alone) for this
        step's ``label`` through the exchange its pool takes: of a split label's tokens, or of a
        copied label's heads at a ``tp * sp`` geometry (`sequence_shard`). The ring writes and
        attends through the KV resource itself, not ``run``."""
        label = self._default_label if label is None else label
        shard = self._shard
        r = shard.split_range(label)
        # the paged run and K/V writes refuse a split label reached without it (`check_bypass`)
        shard.exchanging = True
        try:
            if r is None:
                return head_slice(run, q, k, v, shard.group) if self._placement is None else run(q, k, v)
            shard.check_scattered(label)
            if self._placement is None:
                return ulysses(run, q, k, v, r)
            if self._split_exchange == "ring":
                return self._ring(label, q, k, v, r)
            return all_gather(run, q, k, v, r)
        finally:
            shard.exchanging = False

    def _ring(self, label: str, q, k, v, r: SeqRange) -> torch.Tensor | None:
        """This rank's tokens' K/V into the slots it stores them in (all of them, as the step is
        striped by stored position), then, ``q`` given, the ring (`RingAttentionWrapper`), at the KV
        resource's layer cursor (`AttentionCallable.set_layer_idx`)."""
        self._kv.write_kv(k, v, label=label, tokens=r.index)
        return None if q is None else self._current_plan_states[label].attend(q, self._kv.layer_view())

    def _check_sequence_shard(self, ctx) -> SeqShardPlan | None:
        """At plan: the step declares the node's sequence shard, whose plan this returns where the
        shard can split."""
        if self._shard_key is None:
            return None
        plan = check_shard_plan(f"attention over KV {self._kv_cache_name!r}", self._shard_key, ctx)
        return plan if self._shard is not None else None

    def _bind_placement(self, placement: "ContextParallel") -> None:
        """This rank's share of a context-parallel pool. Reached only for a backend that supports
        one (``supports_context_parallel``): the engine refuses the others at load."""

    # A backend that supports a placed pool supplies the two wrappers a step attends it on;
    # `_context_parallel_wrapper` composes them and `_attend_placed` runs them.
    # ``run``'s query parts, as `_attend_placed` passes them
    _query_parts = 1

    def _context_wrapper(self, workspace: torch.Tensor, decode: bool, kwargs: dict, max_total_tokens: int | None):
        """The paged wrapper over this rank's slots, from ``kwargs`` (the backend's wrapper
        arguments with the placed geometry, and any capture sizes), for a one-token-a-row step or
        not."""
        raise NotImplementedError

    def _own_tokens_wrapper(self, workspace: torch.Tensor, kwargs: dict, max_total_tokens: int | None):
        """The wrapper a multi-token step attends its own tokens on under TP placement, from the
        keys ``run`` is passed: ``run(*queries, *keys, return_lse=True)``, out 0 and lse -inf on a
        captured bucket's padding."""
        raise NotImplementedError

    def _context_parallel_wrapper(
        self, label: str, slot: int, decode: bool, kwargs: dict, max_total_tokens: int | None = None,
    ):
        """The ``ContextParallelWrapper`` a step attends a placed pool on: the context's wrapper and,
        for a step that isn't one token a row, its own tokens'.

        Both take the label's workspace although both are planned for the one step: a FlashInfer
        wrapper keeps its plan in an int workspace of its own, and uses this one only for the
        split-KV partials of a ``run``, which it merges within that ``run``; the two run one after
        the other on one stream."""
        from mstar.engine.resources.context_parallel import ContextParallelWrapper, ZigzagWrapper

        workspace = self._workspaces.get(label, slot)
        context = self._context_wrapper(workspace, decode, kwargs, max_total_tokens)
        if decode:
            new = None
        elif self._placement.over_sp:
            # SP partners hold the same heads, so each attends its zigzag share of the own tokens, on
            # the context's kernel over pages of one slot
            def paged(max_total_tokens=None, **captured):
                return self._context_wrapper(workspace, False, {**kwargs, **captured, "page_size": 1}, max_total_tokens)

            new = ZigzagWrapper(
                self._placement, paged, self._query_parts, self._device, self._dtype,
                kwargs.get("batch_size") or 0, max_total_tokens or 0, check=self._plan_check,
            )
        else:
            new = self._own_tokens_wrapper(workspace, kwargs, max_total_tokens)
        return ContextParallelWrapper(
            self._placement, self._kv_config.tokens_per_page, self._dtype, context, new, self._device,
            kwargs.get("batch_size"), max_total_tokens, check=self._plan_check,
        )

    def _ring_wrapper(self, label: str, slot: int):
        """The eager ``RingAttentionWrapper`` a split step of ``label`` plans on ``slot`` under
        ``sp_exchange`` ring: one context wrapper per partner, over the label's workspace (they run
        one after another on one stream)."""
        from mstar.engine.resources.context_parallel import RingAttentionWrapper

        key = EagerSlotKey(label=label, slot=slot, ring=True)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            workspace = self._workspaces.get(label, slot)
            paged = [
                self._context_wrapper(workspace, False, dict(self._wrapper_kv_kwargs), None)
                for _ in range(self._placement.size)
            ]
            wrapper = self._eager_plan_states[key] = RingAttentionWrapper(
                self._placement, self._kv_config.tokens_per_page, self._dtype, paged, self._device,
                check=self._plan_check,
            )
        return wrapper

    @torch.compiler.disable
    def _attend_placed(
        self, label: str, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        new_keys: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        """``run`` over a placed pool, ``queries`` and ``new_keys`` as this backend's ``run`` takes
        them (``ContextParallelWrapper.attend``)."""
        if self._shard is not None:
            self._shard.check_bypass(label, "attention run")
        return self._current_plan_states[label].attend(queries, kv_cache_layer, new_keys)

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
        manager._split_exchange = spec.split_exchange
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
        is True.

        Every step of a backend that takes the fresh K/V straight into the kernel; on a paged
        backend over a context-parallel pool, a step that attends its own tokens apart from the
        pool. Fixed once the step is planned, so a captured bucket replays the same answer.
        """
        if self._placement is not None:
            return self._current_plan_states[label or self._default_label].reads_step_kv
        return not self.requires_kv_write


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    # the int32 arrays' bytes
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


def workspace_bytes() -> int:
    """Size of one FlashInfer workspace buffer (``MSTAR_WORKSPACE_BUFFER_MB``)."""
    return int(os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")) * 1024 * 1024


class WorkspacePool:
    """FlashInfer workspace buffers, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so
    these outlive any single step.
    """

    def __init__(self, device: torch.device, size_mb: int | None = None):
        self._device = device
        self._size = workspace_bytes() if size_mb is None else size_mb * 1024 * 1024
        self._buffers: dict[str, torch.Tensor] = {}

    def get(self, label: str, cg_slot: int | None = None) -> torch.Tensor:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._buffers:
            self._buffers[key] = torch.empty(
                self._size, dtype=torch.uint8, device=self._device
            )
        return self._buffers[key]
