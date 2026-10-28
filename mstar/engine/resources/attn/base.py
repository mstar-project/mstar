"""The attention resource's shared machinery: the spec-time factory, the
custom op every backend attends through, and the workspace pool.

The backends themselves live beside this — `flashinfer`, `cross`, `dense`,
`flex` — and `AttentionManager.build` reaches them by deferred import, so
naming a backend in a spec does not load the others.
"""

import logging
import os
from typing import NamedTuple

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

    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    def fingerprint(self) -> bytes:
        # the class, not the declared backend: `build` degrades dense to paged
        # where the FA3 kernel is unusable, and the two do not agree bit for bit
        return fingerprint(type(self).__name__)

    @staticmethod
    def _refuse_unread(spec: AttentionSpec, backend: AttnBackend) -> None:
        """A backend that ignores ``sliding_window`` or ``sm_scale`` would attend
        with the wrong mask or scale; refuse the spec instead."""
        reads_scale = (AttnBackend.FLASHINFER, AttnBackend.FLASHINFER_MLA)
        for name, set_, readers in (
            ("sliding_window", spec.config.sliding_window is not None, (AttnBackend.FLASHINFER,)),
            ("sm_scale", spec.config.sm_scale is not None, reads_scale),
        ):
            if set_ and backend not in readers:
                raise ValueError(
                    f"attention backend {backend!r} ignores {name}; only "
                    f"{', '.join(repr(b.value) for b in readers)} read it "
                    f"(KV resource {spec.config.kv_cache!r})"
                )

    @staticmethod
    def _check_window_pool(
        spec: AttentionSpec, backend: AttnBackend, kv_config: PagedKVConfig, device: torch.device,
    ) -> str:
        """The FlashInfer kernel to read a pool with sliding layers with. Only FlashInfer reads
        the per-group page sizes and strided pages, and only the kernels that take any page
        size: FA2, and FA3 on SM90 only. ``auto`` picks one of them below SM100, and from
        SM100 can send short rows without a window to cuDNN, so there it becomes FA2. An
        attention reads the group that keeps its window, so without one it needs a group
        that keeps everything."""
        name = spec.config.kv_cache
        if backend != AttnBackend.FLASHINFER:
            raise ValueError(f"KV {name!r} declares sliding_layers, which only the flashinfer backend reads")
        kernel = spec.config.flashinfer_backend
        capability = current_platform.cuda_capability(device)
        sm100 = capability is not None and capability >= (10, 0)
        # FlashInfer has fa3 for SM90 only; no capability is a CPU device or a GPU not NVIDIA's
        allowed = ("auto", "fa2", "fa3") if capability is None or capability[0] == 9 else ("auto", "fa2")
        if kernel not in allowed:
            raise ValueError(
                f"KV {name!r} declares sliding_layers, whose pages only FlashInfer's fa2 and fa3 "
                f"(SM90 only) kernels read, so flashinfer_backend={kernel!r} can't on this GPU; "
                f"use {' or '.join(allowed)}"
            )
        if spec.config.sliding_window not in (None, kv_config.sliding_window):
            raise ValueError(
                f"attention {spec.resource_key!r} has sliding_window={spec.config.sliding_window}, but "
                f"KV {name!r} keeps a window of {kv_config.sliding_window} for its sliding layers"
            )
        if spec.config.sliding_window is None and kv_config.groups[0].retention is not None:
            raise ValueError(
                f"every layer of KV {name!r} slides, so attention {spec.resource_key!r} needs its "
                f"sliding_window={kv_config.sliding_window}"
            )
        return "fa2" if sm100 and kernel == "auto" else kernel

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        backend = resolve_attn_backend(spec.config.backend, info.device)
        cls._refuse_unread(spec, backend)
        # the KV resource's own config, not a copy; per-rank head counts, and
        # `shard` is idempotent so both builders can call it
        kv_config = info.dependency(spec.config.kv_cache).config
        flashinfer_backend = spec.config.flashinfer_backend
        if isinstance(kv_config, PagedKVConfig) and kv_config.sliding_layers:
            flashinfer_backend = cls._check_window_pool(spec, backend, kv_config, info.device)
            if flashinfer_backend != spec.config.flashinfer_backend:
                logger.info(
                    "attention %r reads KV %r, which has sliding_layers: flashinfer_backend=%r "
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
        # `shard` lives on the KVConfig base, so this is unchanged for both.
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        if backend == AttnBackend.DENSE:
            # A dense backend needs the FlashAttention-3 kernel; where the
            # wheel does not match the installed torch/CUDA build or has no
            # kernel for the GPU, degrade to the paged backend AUTO picks
            # rather than failing to serve.
            # The two are drop-in for each other: the step declaration is
            # identical (see DenseAttentionManager) and `requires_kv_write`
            # tells the layer which one it is talking to.
            from mstar.engine.resources.attn.dense import (
                DenseAttentionManager,
                _fa3_unavailable_reason,
                _warn_dense_fallback,
            )

            reason = _fa3_unavailable_reason(info.device)
            if reason is None:
                return DenseAttentionManager(
                    kv_cache=spec.config.kv_cache,
                    device=info.device,
                    dtype=info.kv_dtype,
                    kv_config=kv_config,
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
                sm_scale=spec.config.sm_scale,
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

        and stays correct whichever backend the spec named.
        """
        return True


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


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
    """Page ids a captured wrapper's index buffer holds for a bucket of
    ``num_rows`` rows and ``num_tokens`` tokens, in pages of ``group`` (the
    pool's first by default).

    Per row, not per pool: rows that matched one cached prefix all name its
    pages, so a step can name more page ids than the pool holds. And at least
    the bucket's own pages: a capture's rows name a SINK_PAGE per page of span,
    whatever the pool's size.

    A pool with sliding layers keeps no prefix cache and forks nothing, so its
    real rows share no pages and name at most the pool, and padding rows their
    sink pages on top. A group with a retention holds less than a page past it
    per row before the step, plus what the step adds."""
    group = group or kv_config.groups[0]
    page_size = group.page_size
    step_pages = -(-num_tokens // page_size)
    if group.retention is not None:
        held = -(-(group.retention.context_budget + page_size - 1) // page_size) + 1
        return num_rows * held + step_pages
    if kv_config.sliding_layers:
        return kv_config.num_units + step_pages + num_rows
    return max(
        num_rows * kv_config.max_num_pages,
        step_pages + num_rows,
    )


class WorkspacePool:
    """FlashInfer workspace buffers, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so
    these outlive any single step.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._size = int(
            os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")
        ) * 1024 * 1024
        self._buffers: dict[str, torch.Tensor] = {}

    def get(self, label: str, cg_slot: int | None = None) -> torch.Tensor:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._buffers:
            self._buffers[key] = torch.empty(
                self._size, dtype=torch.uint8, device=self._device
            )
        return self._buffers[key]
