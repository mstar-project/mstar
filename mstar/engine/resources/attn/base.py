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
from mstar.engine.resources.kv.config import KVConfig, PagedKVConfig, RingKVConfig
from mstar.engine.resources.kv.keys import fingerprint

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
        # the class, not the declared backend: `build` moves dense to another
        # kernel where the FA3 one is unusable, and they do not agree bit for bit
        return fingerprint(type(self).__name__)

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
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
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
                backend=spec.config.flashinfer_backend,
                splits=spec.config.splits,
                max_rows_in_combined=info.max_rows_in_combined,
                decode_tensor_cores=spec.config.decode_tensor_cores,
                sliding_window=spec.config.sliding_window,
                decode_fmha=spec.config.decode_fmha,
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


def captured_page_capacity(num_rows: int, num_tokens: int, kv_config: KVConfig) -> int:
    """Page ids a captured wrapper's index buffer holds for a bucket of
    ``num_rows`` rows and ``num_tokens`` tokens.

    Per row, not per pool: rows that matched one cached prefix all name its
    pages, so a step can name more page ids than the pool holds. And at least
    the bucket's own pages: a capture's rows name a SINK_PAGE per page of span,
    whatever the pool's size."""
    return max(
        num_rows * kv_config.max_num_pages,
        -(-num_tokens // kv_config.page_size) + num_rows,
    )


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
