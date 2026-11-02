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

from mstar.engine.resources.attn.config import AttentionSpec, AttnBackend
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo
from mstar.engine.resources.kv.config import KVConfig, KVGroup, PagedKVConfig, RingKVConfig
from mstar.engine.resources.kv.keys import fingerprint
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

    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    def depends_on(self):
        return {self._kv_cache_name}

    def bind_dependencies(self, resources) -> None:
        placement = resources[self._kv_cache_name].placement
        if placement is not None:
            self._placement = placement
            self._bind_placement(placement)

    def _bind_placement(self, placement: "ContextParallel") -> None:
        """This rank's share of a context-parallel pool. Reached only for a backend that supports
        one (``supports_context_parallel``): the engine refuses the others at load."""

    # A backend that supports a placed pool supplies the two wrappers a step attends it on;
    # `_context_parallel_wrapper` composes them and `_attend_placed` runs them.

    def _context_wrapper(self, workspace: torch.Tensor, decode: bool, kwargs: dict, max_total_tokens: int | None):
        """The paged wrapper over this rank's slots, from ``kwargs`` (the backend's wrapper
        arguments with the placed geometry, and any capture sizes), for a one-token-a-row step or
        not."""
        raise NotImplementedError

    def _own_tokens_wrapper(self, workspace: torch.Tensor, kwargs: dict, max_total_tokens: int | None):
        """The wrapper a multi-token step attends its own tokens on, from the keys ``run`` is
        passed: ``run(*queries, *keys, return_lse=True)``, out 0 and lse -inf on a captured
        bucket's padding."""
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
        from mstar.engine.resources.context_parallel import ContextParallelWrapper

        workspace = self._workspaces.get(label, slot)
        context = self._context_wrapper(workspace, decode, kwargs, max_total_tokens)
        new = None if decode else self._own_tokens_wrapper(workspace, kwargs, max_total_tokens)
        return ContextParallelWrapper(
            self._placement, self._kv_config.tokens_per_page, self._dtype, context, new, self._device,
            kwargs.get("batch_size"), max_total_tokens,
        )

    @torch.compiler.disable
    def _attend_placed(
        self, label: str, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        new_keys: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        """``run`` over a placed pool, ``queries`` and ``new_keys`` as this backend's ``run`` takes
        them (``ContextParallelWrapper.attend``)."""
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
