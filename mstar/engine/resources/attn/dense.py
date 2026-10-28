"""Attention as one dense varlen pass: FlashAttention-3, AITER on ROCm, or
FlashInfer's ragged prefill where neither has a kernel for the GPU."""

import functools
import logging
from dataclasses import dataclass, field, replace

import torch

from mstar.engine.resources.attn.base import AttentionManager, EagerSlotKey, WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep, AttnBackend, refuse_context_only
from mstar.engine.resources.attn.ragged.wrappers import FlashInferPrefixedPrefillWrapper
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import StepContext
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

# imported here, not per `run`: the wheel may be absent, so keep the failure
try:
    from fa3_fwd_interface import flash_attn_varlen_func
except Exception as _fa3_exc:  # noqa: BLE001
    flash_attn_varlen_func = None
    _VARLEN_IMPORT_ERROR: str | None = f"{type(_fa3_exc).__name__}: {_fa3_exc}"
else:
    _VARLEN_IMPORT_ERROR = None

# ROCm has no FA3; AITER's varlen FMHA takes the same call, so dense runs on it
# there instead of degrading to the paged backend
if flash_attn_varlen_func is None and current_platform.is_rocm():
    try:
        from aiter import flash_attn_varlen_func
    except Exception as _aiter_exc:  # noqa: BLE001
        _VARLEN_IMPORT_ERROR = f"{_VARLEN_IMPORT_ERROR}; aiter: {type(_aiter_exc).__name__}: {_aiter_exc}"
    else:
        _VARLEN_IMPORT_ERROR = None


def _fa3_has_kernel(capability: tuple[int, int]) -> bool:
    # fa3-fwd ships sm_90a and sm_80 kernels and no PTX, so a GPU of any other
    # architecture, Blackwell's among them, has no image to launch
    return capability == (9, 0) or capability[0] == 8


@functools.cache
def _fa3_unavailable_reason(device: torch.device | None = None) -> str | None:
    """why no varlen kernel can run on ``device`` (the current GPU if None): none
    imported (fa3-fwd, and AITER on ROCm), or fa3-fwd has none for its capability"""
    if _VARLEN_IMPORT_ERROR is not None:
        return _VARLEN_IMPORT_ERROR
    if device is not None and torch.device(device).type != "cuda":
        return None
    capability = current_platform.cuda_capability(device)
    if capability is not None and not _fa3_has_kernel(capability):
        return f"fa3-fwd has no kernel for compute capability {capability[0]}.{capability[1]}"
    return None


@functools.cache
def _warn_dense_fallback(reason: str, fallback: AttnBackend) -> None:
    logger.warning(
        "Attention backend 'dense' requested but no varlen kernel is "
        "available (%s); using the paged %r backend instead, though it also "
        "runs on FlashInfer and cannot run where FlashInfer is unusable.",
        reason, fallback.value,
    )


@functools.cache
def _flashinfer_ragged_unavailable_reason(device: torch.device | None = None) -> str | None:
    """why the dense pass cannot run on FlashInfer's ragged prefill on ``device``"""
    if device is not None and torch.device(device).type != "cuda":
        return f"FlashInfer needs a CUDA device, not {torch.device(device).type}"
    if current_platform.is_rocm():
        return "FlashInfer is not built for ROCm"
    try:
        import flashinfer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def _log_dense_on_flashinfer(reason: str, backend: str) -> None:
    logger.info(
        "Attention backend 'dense': %s; running the dense pass on FlashInfer's "
        "ragged prefill (backend %r).", reason, backend,
    )


@dataclass(frozen=True)
class DenseSegment:
    """(request, label) of dense plan; frozen prefix and fresh tokens"""
    request_id: str
    label: str
    # the prefix's pages, in stream order; a host tuple, moved to the device
    # only when the prefix is gathered (a per-step H2D from pageable memory
    # would sync the stream)
    pages: tuple[int, ...]
    prefix_len: int
    q_len: int
    generation: int  # of the stream, at plan time; see CacheStream.generation


@dataclass(frozen=True)
class DensePlan:
    """dense plan layout. once per step and read by each layer `run`"""
    segments: tuple[DenseSegment, ...]
    cu_q: tuple[int, ...]
    cu_k: tuple[int, ...]
    max_q: int
    max_k: int
    causal: bool
    # what the backend's kernel planned this layout into (see `_plan_kernel`)
    kernel: object = None


@dataclass
class _GatheredPrefix:
    """One stream's gathered prefix, per layer, tagged with what it was read
    from so a stale gather cannot be served."""
    """gathered prefix of one stream per-layer. tagged with what read from so stale cannot serve"""
    generation: int
    length: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


class DenseAttentionManager(AttentionManager):
    """Attention as one dense varlen pass (FA3, AITER on ROCm) over a contiguous
    [frozen prefix | fresh tokens] sequence.

    For a workload that recomputes every one of its K/V every step and only
    reuses a small frozen prefix — diffusion denoise, where the prefix is the
    text conditioning — the paged path's per-step full-buffer K/V write and
    ``wrapper.plan`` are pure overhead. This gathers the prefix once per
    (stream, layer), concatenates it with the freshly projected K/V, and runs
    one kernel. The gathered prefix is reused across steps: it is keyed on the
    stream's ``generation``, so a fork, a reset, or a page-table move
    invalidates it (``CacheStream.generation``).

    The layer does not write its K/V to the pages here (``requires_kv_write``
    is False) — nothing would ever read it back.

    Eager-only. The gather and the concatenation are shape-dependent and
    allocate, so there is nothing to capture; a node that runs this walk under
    a graph should name the paged backend for it.

    NOTE on the step declaration: the fresh tokens are declared as ordinary
    spans on ``KVStep`` with ``commit=False``, exactly as they would be for the
    paged backend, so the query lengths arrive as the KV plan's ``to_compute``
    and the prefix as ``length - to_compute``. That is what keeps the backend a
    spec-time choice: the same declaration runs either way, and positions still
    take their packing off the KV plan output. The alternative — declaring the
    fresh tokens zero-span and carrying their lengths on ``AttentionStep``'s
    own segments — is what the v0 dense path did and would save the pages that
    are reserved here and never written; it costs the swappability, since a
    model would then have to declare its step differently per backend, and a
    zero-span label yields no packing for any resource that derives from it
    (``PositionManager`` would only work for labels supplying explicit pos_ids).
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._on_cuda = device.type == "cuda" and torch.cuda.is_available()

        # plan label -> this step's layout
        self._current_plans: dict[str, DensePlan] = {}
        # (request, label) -> the prefix gathered out of the pages
        self._prefix_cache: dict[tuple[str, str], _GatheredPrefix] = {}

    def depends_on(self):
        return {self._kv_cache_name}

    def can_capture(self) -> bool:
        return False

    @property
    def requires_kv_write(self) -> bool:
        return False

    # the gathered prefix is per-request state, so unlike the paged backend
    # these are not no-ops
    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._drop_prefixes(rid)

    def remove_request(self, rid: str):
        self._drop_prefixes(rid)

    def reset_request(self, rid: str, free: bool=False):
        del free
        self._drop_prefixes(rid)

    def _drop_prefixes(self, rid: str) -> None:
        for key in [k for k in self._prefix_cache if k[0] == rid]:
            del self._prefix_cache[key]

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        if ctx.slot_lease is not None or ctx.is_preplan:
            raise RuntimeError(
                "dense attention is eager-only: the prefix gather and the "
                "concatenation allocate and are shaped by the step, so there is "
                "nothing for a captured graph to replay"
            )
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"Dense Attention Manager expected plan result from {self._kv_cache_name}"
        )

        plans = {}
        for plan_label, kv_out in plan_outputs.items():
            plan = self._build_plan(kv_out, step.causal)
            plans[plan_label] = replace(plan, kernel=self._plan_kernel(plan_label, plan, ctx.slot))
        self._current_plans = plans

    def _plan_kernel(self, label: str, plan: DensePlan, slot: int) -> object:
        """Plan the varlen kernel on one label's layout; `_varlen` gets it back."""
        del label, slot
        return (
            torch.tensor(plan.cu_q, dtype=torch.int32, device=self._device),
            torch.tensor(plan.cu_k, dtype=torch.int32, device=self._device),
        )

    def _varlen(self, plan: DensePlan, q: torch.Tensor, key: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        cu_q, cu_k = plan.kernel
        out = flash_attn_varlen_func(
            q, key, val,
            cu_q, cu_k, plan.max_q, plan.max_k,
            # a causal plan aligns the query block to the end of the key
            # sequence, so the fresh tokens see the whole prefix; only
            # non-causal generation attention is exercised today
            causal=plan.causal,
        )
        return out[0] if isinstance(out, tuple) else out

    def _build_plan(self, kv_out: KVPlanOutput, causal: bool) -> DensePlan:
        """The per-segment gather layout plus the cumulative lengths one varlen
        kernel needs, in the packed order the KV plan defined."""
        page_size = self._kv_config.page_size
        segments: list[DenseSegment] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for view in kv_out.views:
            assert view.start == 0, (
                f"dense attention reads a stream from its first page; "
                f"{view.label!r} of {view.request_id!r} starts at {view.start}"
            )
            prefix_len = view.length - view.to_compute
            n_pages = (prefix_len + page_size - 1) // page_size
            segments.append(DenseSegment(
                request_id=view.request_id,
                label=view.label,
                pages=tuple(view.page_idxs[:n_pages]),
                prefix_len=prefix_len,
                q_len=view.to_compute,
                generation=view.generation,
            ))
            cu_q.append(cu_q[-1] + view.to_compute)
            cu_k.append(cu_k[-1] + view.length)
            max_q = max(max_q, view.to_compute)
            max_k = max(max_k, view.length)

        return DensePlan(
            segments=tuple(segments),
            cu_q=tuple(cu_q),
            cu_k=tuple(cu_k),
            max_q=max_q,
            max_k=max_k,
            causal=causal,
        )

    def _prefix_kv(
        self, segment: DenseSegment, layer_idx: int, kv_cache_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This segment's frozen prefix for one layer, gathered out of the
        pages on first use and held until the stream moves under it."""
        key = (segment.request_id, segment.label)
        entry = self._prefix_cache.get(key)
        if (
            entry is None
            or entry.generation != segment.generation
            or entry.length != segment.prefix_len
        ):
            entry = self._prefix_cache[key] = _GatheredPrefix(
                generation=segment.generation, length=segment.prefix_len
            )

        cached = entry.layers.get(layer_idx)
        if cached is not None:
            return cached

        cfg = self._kv_config
        # [n_pages, 2, page_size, num_kv_heads, head_dim] -> the prefix's rows
        pages = kv_cache_layer[
            torch.tensor(segment.pages, dtype=torch.long, device=kv_cache_layer.device)
        ]
        k_pref = pages[:, 0].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        v_pref = pages[:, 1].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        # the gather already copied; the slice keeps the trailing slots of the
        # last page out, and `clone` keeps the whole page span from being held
        cached = (
            k_pref[:segment.prefix_len].clone(), v_pref[:segment.prefix_len].clone()
        )
        entry.layers[layer_idx] = cached
        return cached

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """One layer's dense attention. ``k``/``v`` are this step's freshly
        projected K/V, packed in the plan's segment order; they are never
        written to the pages."""
        if label is None:
            label = self._default_label
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        assert k is not None and v is not None and layer_idx is not None, (
            "dense attention runs on the step's own K/V: pass k, v and "
            "layer_idx (the layer writes nothing through the KV resource "
            "under this backend — see `requires_kv_write`)"
        )
        plan = self._current_plans[label]

        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        offset = 0
        for segment in plan.segments:
            if segment.prefix_len:
                k_pref, v_pref = self._prefix_kv(segment, layer_idx, kv_cache_layer)
                k_parts.append(k_pref)
                v_parts.append(v_pref)
            k_parts.append(k[offset:offset + segment.q_len])
            v_parts.append(v[offset:offset + segment.q_len])
            offset += segment.q_len

        key = torch.cat(k_parts, dim=0)
        val = torch.cat(v_parts, dim=0)
        if q.dtype != key.dtype:
            q = q.to(key.dtype)
        return self._varlen(plan, q, key, val)


class FlashInferDenseAttentionManager(DenseAttentionManager):
    """The dense pass on FlashInfer's ragged prefill: the same gathered prefix
    and concatenation, one ``BatchPrefillWithRaggedKVCacheWrapper`` run per
    layer. ``AttentionManager.build`` picks it where FA3 has no kernel for the
    GPU (Blackwell): there ``backend="auto"`` resolves to FlashInfer's cuDNN or
    CUTLASS FMHA, about 4x the paged prefill's FA2 kernel on a long
    non-causal step.
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
        backend: str = "auto",
    ):
        super().__init__(kv_cache=kv_cache, device=device, dtype=dtype, kv_config=kv_config)
        self._backend = backend
        self._workspaces = WorkspacePool(device)
        self._wrappers: dict[EagerSlotKey, FlashInferPrefixedPrefillWrapper] = {}

    @property
    def force_double_buffer(self):
        # FlashInfer's plan stages its schedule in a pinned buffer per wrapper
        # and H2Ds it on the stream, so each slot keeps its own wrapper
        return True

    def _wrapper(self, label: str, slot: int):
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._wrappers.get(key)
        if wrapper is None:
            cfg = self._kv_config
            wrapper = self._wrappers[key] = FlashInferPrefixedPrefillWrapper(
                self._workspaces.get(label, slot),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                backend=self._backend,
                device=self._device,
                q_data_type=self._dtype,
            )
        return wrapper

    def _plan_kernel(self, label: str, plan: DensePlan, slot: int) -> object:
        wrapper = self._wrapper(label, slot)
        # host layouts: a device one would cost FlashInfer's plan a sync
        wrapper.plan(
            torch.tensor(plan.cu_q, dtype=torch.int32),
            torch.tensor(plan.cu_k, dtype=torch.int32),
            causal=plan.causal,
        )
        return wrapper

    def _varlen(self, plan: DensePlan, q: torch.Tensor, key: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        return plan.kernel.run(q, key, val)
