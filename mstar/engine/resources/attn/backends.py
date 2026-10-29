"""Which paged attention backends exist, what each needs, and what ``AUTO`` picks.

``AttentionManager.build`` reads this table instead of branching on the
backend. Each entry names its builder, imported only when that backend is
built (naming a backend in a spec loads no other), the KV config it needs,
where it runs, which spec options it reads, and a probe that says why it
cannot run here. ``AUTO`` resolves through a policy per platform, so a
platform's defaults are declared beside its kernels and ``attn.base`` stays
platform-neutral: a backend for a new platform is one more entry and, if it
should be that platform's default, one more policy.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch

from mstar.engine.resources.attn.config import AttnBackend
from mstar.engine.resources.kv.config import KVConfig, PagedKVConfig, RingKVConfig
from mstar.platforms import current_platform

if TYPE_CHECKING:
    from mstar.engine.resources.attn.config import AttentionSpec
    from mstar.engine.resources.base import EngineResourceInfo

# a "module:function" reference, imported when it is first needed, or the
# same keyed by platform name ("default" for the rest)
Ref = str | Mapping[str, str]


def _load(ref: Ref | None) -> Callable[..., Any] | None:
    if ref is None:
        return None
    if not isinstance(ref, str):
        ref = ref.get(current_platform.name) or ref.get("default")
        if ref is None:
            return None
    module, _, name = ref.partition(":")
    return getattr(importlib.import_module(module), name)


@dataclasses.dataclass(frozen=True)
class AttnBackendEntry:
    backend: AttnBackend
    # (spec, info, kv_config) -> the manager
    builder: Ref
    # the KV config the backend reads its cache through
    kv_config: type[KVConfig]
    # the platforms (``current_platform.name``) and device types it runs on;
    # empty for any
    platforms: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    # (device) -> why it cannot run there, or None
    probe: Ref | None = None
    # what the probe checks, for the error when it fails
    needs: str = ""
    # when the probe fails: raise, or build AUTO's backend after `on_fallback`
    # (reason, fallback backend) warns
    falls_back_to_auto: bool = False
    on_fallback: Ref | None = None
    # the AttentionConfig options it honours
    sliding_window: bool = False
    sm_scale: bool = False


class AttnBackendRegistry:
    """The entries of one attention resource family, and its ``AUTO`` policies:
    a backend, or a ``Ref`` to ``(page_size) -> backend``, per platform name
    with ``"default"`` for the rest."""

    def __init__(self, family: str, auto: Mapping[str, AttnBackend | str]):
        if "default" not in auto:
            raise ValueError(f"{family}: AUTO needs a default policy")
        self.family = family
        self._entries: dict[AttnBackend, AttnBackendEntry] = {}
        self._auto: dict[str, AttnBackend | str] = dict(auto)

    def register(self, entry: AttnBackendEntry) -> None:
        if entry.backend is AttnBackend.AUTO or entry.backend in self._entries:
            raise ValueError(f"{self.family}: cannot register {entry.backend!r}")
        self._entries[entry.backend] = entry

    def set_auto(self, platform: str, policy: AttnBackend | str) -> None:
        self._auto[platform] = policy

    def __iter__(self) -> Iterator[AttnBackendEntry]:
        return iter(self._entries.values())

    def __contains__(self, backend: AttnBackend) -> bool:
        return backend in self._entries

    def entry(self, backend: AttnBackend) -> AttnBackendEntry:
        if backend not in self._entries:
            raise ValueError(f"Unknown {self.family} backend {backend!r}")
        return self._entries[backend]

    def auto_policy(self, platform: str) -> AttnBackend | str:
        return self._auto.get(platform, self._auto["default"])

    def resolve(self, backend: AttnBackend, page_size: int | None = None) -> AttnBackend:
        """``backend``, or for ``AUTO`` the platform's default. The platform is
        only asked when a policy other than the default exists."""
        if backend is not AttnBackend.AUTO:
            return backend
        policy = self._auto["default"] if len(self._auto) == 1 else self.auto_policy(current_platform.name)
        if isinstance(policy, AttnBackend):
            return policy
        return _load(policy)(page_size)

    def supporting(self, option: str) -> list[str]:
        """The backends that honour an AttentionConfig option, by value."""
        return [entry.backend.value for entry in self if getattr(entry, option)]

    def build(self, spec: AttentionSpec, info: EngineResourceInfo):
        """The manager ``spec`` asks for, after the checks every backend gets."""
        # An unhonoured window is refused before the KV dependency is read;
        # AUTO then resolves again with the cache's page size.
        self._check_window(spec, self.resolve(spec.config.backend))
        kv_config = info.dependency(spec.config.kv_cache).config
        page_size = getattr(kv_config, "page_size", None)
        backend = self.resolve(spec.config.backend, page_size)
        self._check_window(spec, backend)
        entry = self.entry(backend)
        # Before `shard`, so a mismatch is reported rather than half-applied
        # to a config the KV resource also holds.
        self._check(entry, spec, kv_config)
        if info.joint_comm_group is not None:
            # per-rank head counts; `shard` is idempotent, so both builders can call it
            kv_config.shard(info.joint_comm_group.world_size)
        if entry.platforms and current_platform.name not in entry.platforms:
            raise RuntimeError(
                f"attention backend {backend.value!r} runs on {' or '.join(entry.platforms)}; "
                f"this process runs on {current_platform.name}"
            )
        if entry.devices and info.device.type not in entry.devices:
            raise RuntimeError(
                f"attention backend {backend.value!r} requires an "
                f"{' or '.join(d.upper() for d in entry.devices)} device; got {info.device}"
            )
        probe = _load(entry.probe)
        reason = probe(info.device) if probe is not None else None
        if reason is not None:
            if not entry.falls_back_to_auto:
                raise RuntimeError(
                    f"attention backend {backend.value!r} requires a working {entry.needs} ({reason})"
                )
            fallback = self.resolve(AttnBackend.AUTO, page_size)
            warn = _load(entry.on_fallback)
            if warn is not None:
                warn(reason, fallback)
            entry = self.entry(fallback)
            self._check(entry, spec, kv_config)
        return _load(entry.builder)(spec, info, kv_config)

    def _check_window(self, spec: AttentionSpec, backend: AttnBackend) -> None:
        if spec.config.sliding_window is not None and not (backend in self and self.entry(backend).sliding_window):
            raise ValueError(
                f"sliding_window is unsupported for attention backend {backend!r}; "
                f"use the {' or '.join(self.supporting('sliding_window'))} backend"
            )

    def _check(self, entry: AttnBackendEntry, spec: AttentionSpec, kv_config: KVConfig) -> None:
        if not isinstance(kv_config, entry.kv_config):
            raise TypeError(
                f"attention backend {entry.backend.value!r} requires a "
                f"{entry.kv_config.__name__}, but KV resource {spec.config.kv_cache!r} "
                f"holds a {type(kv_config).__name__}. "
            )
        if spec.config.sm_scale is not None and not entry.sm_scale:
            raise ValueError(
                f"attention backend {entry.backend.value!r} ignores sm_scale; only "
                f"{' and '.join(repr(v) for v in self.supporting('sm_scale'))} reads it"
            )


# --- builders and probes, each importing only its own backend ------------------


def _build_dense(spec, info, kv_config):
    from mstar.engine.resources.attn.dense import DenseAttentionManager

    # a varlen kernel: FlashAttention-3, or AITER's on ROCm; `requires_kv_write`
    # tells the layer it is not talking to a paged backend
    return DenseAttentionManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        dtype=info.kv_dtype,
        kv_config=kv_config,
    )


def _dense_probe(device: torch.device) -> str | None:
    from mstar.engine.resources.attn import dense

    return dense._varlen_unavailable_reason()


def _warn_dense_fallback(reason: str, fallback: AttnBackend) -> None:
    from mstar.engine.resources.attn import dense

    dense._warn_dense_fallback(reason, fallback)


# FlashInferManager's platform arguments: AMD's FlashInfer build takes its own
# (`attn.aiter.flashinfer_rocm_kwargs`)
FLASHINFER_KWARGS: Ref = {"rocm": "mstar.engine.resources.attn.aiter:flashinfer_rocm_kwargs"}


def flashinfer_kwargs(kv_config: KVConfig, backend: str) -> dict:
    """``FlashInferManager``'s arguments for a spec naming FlashInfer kernel
    ``backend``, on this platform."""
    hook = _load(FLASHINFER_KWARGS)
    return {"backend": backend} if hook is None else hook(kv_config, backend)


def _build_flashinfer(spec, info, kv_config):
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    return FlashInferManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        dtype=info.kv_dtype,
        kv_config=kv_config,
        splits=spec.config.splits,
        max_rows_in_combined=info.max_rows_in_combined,
        sliding_window=spec.config.sliding_window,
        **flashinfer_kwargs(kv_config, spec.config.flashinfer_backend),
    )


def _build_flashinfer_mla(spec, info, kv_config):
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager

    return FlashInferMLAManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        dtype=info.kv_dtype,
        kv_config=kv_config,
        backend=spec.config.flashinfer_backend,
        sm_scale=spec.config.sm_scale,
    )


def _build_flex(spec, info, kv_config):
    from mstar.engine.resources.attn.flex import FlexAttentionManager

    return FlexAttentionManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        dtype=info.kv_dtype,
        kv_config=kv_config,
    )


def _build_xpu_paged(spec, info, kv_config):
    from mstar.engine.resources.attn.xpu import XPUPagedAttentionManager

    return XPUPagedAttentionManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        kv_config=kv_config,
    )


def _xpu_paged_probe(device: torch.device) -> str | None:
    from mstar.engine.resources.attn.xpu import _xpu_paged_unavailable_reason

    return _xpu_paged_unavailable_reason()


_HERE = __name__

# What AttentionManager.build serves. AUTO is FlashInfer, and on ROCm AITER or
# AMD's FlashInfer build by the cache's page size.
ATTN_BACKENDS = AttnBackendRegistry("attention", auto={
    "default": AttnBackend.FLASHINFER,
    "rocm": "mstar.engine.resources.attn.aiter:rocm_paged_backend",
})
for _entry in (
    AttnBackendEntry(
        AttnBackend.DENSE, builder=f"{_HERE}:_build_dense", kv_config=PagedKVConfig,
        probe=f"{_HERE}:_dense_probe", falls_back_to_auto=True, on_fallback=f"{_HERE}:_warn_dense_fallback",
    ),
    AttnBackendEntry(
        AttnBackend.FLASHINFER, builder=f"{_HERE}:_build_flashinfer", kv_config=PagedKVConfig,
        sliding_window=True,
    ),
    AttnBackendEntry(
        AttnBackend.FLASHINFER_MLA, builder=f"{_HERE}:_build_flashinfer_mla", kv_config=PagedKVConfig,
        sm_scale=True,
    ),
    AttnBackendEntry(AttnBackend.FLEX, builder=f"{_HERE}:_build_flex", kv_config=RingKVConfig),
    AttnBackendEntry(
        AttnBackend.XPU_PAGED, builder=f"{_HERE}:_build_xpu_paged", kv_config=PagedKVConfig,
        devices=("xpu",), probe=f"{_HERE}:_xpu_paged_probe", needs="vllm-xpu-kernels installation",
    ),
    AttnBackendEntry(
        AttnBackend.AITER, builder="mstar.engine.resources.attn.aiter:build_aiter_paged",
        kv_config=PagedKVConfig, platforms=("rocm",),
    ),
):
    ATTN_BACKENDS.register(_entry)
