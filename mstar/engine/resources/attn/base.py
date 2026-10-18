"""The attention resource's shared machinery: the spec-time factory, the
custom op every backend attends through, and the workspace pool.

The backends themselves live beside this — `flashinfer`, `aiter`, `cross`,
`dense`, `flex` — and `AttentionManager.build` reaches them by deferred import,
so naming a backend in a spec does not load the others.
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
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)
_BACKEND_KV_CONFIG: dict[AttnBackend, type[KVConfig]] = {
    AttnBackend.DENSE: PagedKVConfig,
    AttnBackend.FLASHINFER: PagedKVConfig,
    AttnBackend.FLEX: RingKVConfig,
    AttnBackend.XPU_PAGED: PagedKVConfig,
    AttnBackend.AITER: PagedKVConfig,
}


def resolve_attn_backend(
    backend: AttnBackend, device: torch.device, page_size: int | None = None,
) -> AttnBackend:
    """The backend a spec's ``backend`` runs on ``device``.

    The one place ``AUTO`` becomes a kernel choice: a platform whose paged
    attention is better served by another backend answers differently here,
    for every model that left the choice to the engine. Any other value is
    returned as is, so a spec or YAML that names a backend gets that backend.

    ``page_size`` is the paged cache's, when there is one: on ROCm it decides
    between AITER and FlashInfer (see ``_rocm_paged_backend``).
    """
    if backend is not AttnBackend.AUTO:
        return backend
    if current_platform.is_rocm():
        return _rocm_paged_backend(page_size)
    return AttnBackend.FLASHINFER


def _rocm_paged_backend(page_size: int | None) -> AttnBackend:
    """AITER's unified attention, unless the page size is one whose page tails
    it reads unmasked and AMD's FlashInfer build, which masks them, is there."""
    from mstar.engine.resources.attn import aiter as rocm

    aiter_reason = rocm.aiter_paged_unavailable_reason()
    flashinfer_reason = rocm.flashinfer_rocm_unavailable_reason()
    tails_safe = page_size is None or rocm.nan_safe_page_size(page_size)
    if aiter_reason is None and (tails_safe or flashinfer_reason is not None):
        return AttnBackend.AITER
    if flashinfer_reason is None:
        return AttnBackend.FLASHINFER
    raise RuntimeError(
        "No paged attention backend on this ROCm install: AITER's unified "
        f"attention is not importable ({aiter_reason}), nor is AMD's FlashInfer "
        f"build ({flashinfer_reason}). Install amd-aiter (AITER) or amd-flashinfer."
    )


def flashinfer_kwargs(kv_config: KVConfig, backend: str) -> dict:
    """``FlashInferManager``'s platform arguments for a spec naming ``backend``.

    On ROCm only AMD's FlashInfer build runs, and its ``auto`` cannot reach an
    AITER kernel at page 128 (it falls back to fa2 after a failed probe, or
    takes a per-token gather for prefill), so ``auto`` is pinned to fa2 there.
    Decode runs without tensor cores, its faster path on ROCm, for the GQA
    groups that kernel dispatches.
    """
    if not current_platform.is_rocm():
        return {"backend": backend}
    from mstar.engine.resources.attn import aiter as rocm

    reason = rocm.flashinfer_rocm_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            "attention backend 'flashinfer' on ROCm needs AMD's FlashInfer build "
            f"(amd-flashinfer), which is not importable ({reason}); name backend "
            "'aiter' or 'auto' instead"
        )
    group = kv_config.num_qo_heads // kv_config.num_kv_heads
    return {
        "backend": "fa2" if backend == "auto" else backend,
        "decode_tensor_cores": group not in rocm.FLASHINFER_ROCM_DECODE_GROUPS,
    }


class AttentionManager(AttentionResource):
    prefix_skip_safe = True

    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    def fingerprint(self) -> bytes:
        # the class, not the declared backend: `build` degrades dense to paged
        # where the FA3 kernel is unusable, and the two do not agree bit for bit
        return fingerprint(type(self).__name__)

    @classmethod
    def build(cls, spec: AttentionSpec, info: EngineResourceInfo):
        # the KV resource's own config, not a copy; per-rank head counts, and
        # `shard` is idempotent so both builders can call it
        kv_config = info.dependency(spec.config.kv_cache).config
        backend = resolve_attn_backend(
            spec.config.backend, info.device, getattr(kv_config, "page_size", None),
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
            # wheel does not match the installed torch/CUDA build, degrade to
            # the paged backend AUTO picks rather than failing to serve.
            # The two are drop-in for each other: the step declaration is
            # identical (see DenseAttentionManager) and `requires_kv_write`
            # tells the layer which one it is talking to.
            from mstar.engine.resources.attn.dense import (
                DenseAttentionManager,
                _fa3_unavailable_reason,
                _warn_dense_fallback,
            )

            reason = _fa3_unavailable_reason()
            if reason is None:
                return DenseAttentionManager(
                    kv_cache=spec.config.kv_cache,
                    device=info.device,
                    dtype=info.kv_dtype,
                    kv_config=kv_config,
                )
            backend = resolve_attn_backend(
                AttnBackend.AUTO, info.device, kv_config.page_size,
            )
            _warn_dense_fallback(reason, backend)

        if backend == AttnBackend.FLASHINFER:
            from mstar.engine.resources.attn.flashinfer import FlashInferManager

            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
                **flashinfer_kwargs(kv_config, spec.config.flashinfer_backend),
            )

        if backend == AttnBackend.AITER:
            if not current_platform.is_rocm():
                raise RuntimeError(
                    "attention backend 'aiter' runs on AMD GPUs (ROCm); this "
                    f"process runs on {current_platform.name}"
                )
            from mstar.engine.resources.attn.aiter import AiterPagedAttentionManager

            return AiterPagedAttentionManager(
                kv_cache=spec.config.kv_cache,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
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
    # decode and prefill take different wrapper classes; cross-attention plans
    # one wrapper per label either way and leaves this False
    is_decode: bool = False


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


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
