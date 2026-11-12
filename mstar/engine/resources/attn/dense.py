"""Attention as one dense varlen pass: FlashAttention-3, AITER on ROCm, or
FlashInfer's ragged prefill where neither has a kernel for the GPU."""

import functools
import logging
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.base import AttentionManager, EagerSlotKey
from mstar.engine.resources.attn.config import AttentionStep, AttnBackend, refuse_context_only
from mstar.engine.resources.attn.ragged.wrappers import PrefixedPrefillWrapper, _FlashInferRaggedPrefillBase
from mstar.engine.resources.attn.workspace import Workspace, workspace_bytes
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.keys import fingerprint
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
# FA3's fixed-length kernel, for a one-segment plan (`DensePrefixedPrefillWrapper`); varlen runs without it
try:
    from fa3_fwd_interface import flash_attn_func
except Exception:  # noqa: BLE001
    flash_attn_func = None

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


def _flashinfer_version() -> str:
    try:
        import flashinfer
    except Exception:  # noqa: BLE001
        return "(not importable)"
    return flashinfer.__version__


def _dense_unavailable_reason(device: torch.device | None = None) -> str | None:
    """why neither FA3 (AITER on ROCm) nor FlashInfer can run the dense pass on
    ``device``; None if either can"""
    fa3 = _fa3_unavailable_reason(device)
    if fa3 is None:
        return None
    flashinfer = _flashinfer_ragged_unavailable_reason(device)
    return None if flashinfer is None else f"{fa3}; {flashinfer}"


def _views_as_int64(t: torch.Tensor) -> bool:
    """whether ``t`` reads as int64 elements over its last dim: rows of whole,
    aligned 8-byte words"""
    size = t.element_size()
    return (
        t.stride(-1) == 1
        and (t.shape[-1] * size) % 8 == 0
        and (t.storage_offset() * size) % 8 == 0
        and all((stride * size) % 8 == 0 for stride in t.stride()[:-1])
    )


def _copy_rows(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)``, through int64 views when both have one: the same
    bytes, but a strided ``src`` (V sliced out of a fused QKV projection)
    moves 8 bytes per element instead of 2. On Qwen-Image-2.1 (H200) that is
    1.6 ms of a 111.5 ms denoise step instead of 3.4."""
    if src.dtype == dst.dtype and _views_as_int64(src) and _views_as_int64(dst):
        dst.view(torch.int64).copy_(src.view(torch.int64))
    else:
        dst.copy_(src)


def _gather_rows(dst: torch.Tensor, layer: torch.Tensor, rows: torch.Tensor) -> None:
    """Write rows ``rows`` of ``layer`` read as [pages * 2 * page_size, heads,
    head_dim] into ``dst``: one gather, straight into place, through int64
    views when both have one. A layer that isn't one block (a pool with groups
    of two page sizes) is indexed by page and slot instead, through a copy."""
    if layer.is_contiguous() and layer.dtype == dst.dtype:
        flat = layer.view(-1, *layer.shape[-2:])
        if _views_as_int64(flat) and _views_as_int64(dst):
            torch.index_select(flat.view(torch.int64), 0, rows, out=dst.view(torch.int64))
        else:
            torch.index_select(flat, 0, rows, out=dst)
        return
    per_page = 2 * layer.shape[2]
    plane, slot = (rows % per_page).div(layer.shape[2], rounding_mode="floor"), rows % layer.shape[2]
    dst.copy_(layer[rows.div(per_page, rounding_mode="floor"), plane, slot])


@dataclass(frozen=True)
class DenseSegment:
    """(request, label) of dense plan; frozen prefix and fresh tokens"""
    request_id: str
    label: str
    # the prefix's pages, in stream order
    pages: tuple[int, ...]
    prefix_len: int
    q_len: int


@dataclass(frozen=True)
class DensePlan:
    """dense plan layout. once per step and read by each layer `run`"""
    segments: tuple[DenseSegment, ...]
    cu_q: tuple[int, ...]
    cu_k: tuple[int, ...]
    max_q: int
    max_k: int
    causal: bool
    # the wrapper this layout was planned into
    wrapper: PrefixedPrefillWrapper | None = None
    # On the device: every segment's prefix rows in a layer read as [pages * 2 * page_size, heads, head_dim],
    # K's for all segments then V's. A segment's K rows start at `rows_at`, and its V rows `prefix_rows` later.
    rows: torch.Tensor | None = None
    rows_at: tuple[int, ...] = ()
    prefix_rows: int = 0


class DenseAttentionManager(AttentionManager):
    """Attention as one dense varlen pass over a contiguous
    [frozen prefix | fresh tokens] sequence, through a prefixed ragged wrapper:
    FA3's (AITER on ROCm), or FlashInfer's where FA3 has no kernel for the GPU
    (``ragged.base.dense_or_flashinfer`` picks; ``AttentionManager.build``
    passes the class).

    For a workload that recomputes every one of its K/V every step and only
    reuses a small frozen prefix — diffusion denoise, where the prefix is the
    text conditioning — the paged path's per-step full-buffer K/V write and
    ``wrapper.plan`` are pure overhead. Each layer gathers the prefix out of
    the pages straight into place beside the freshly projected K/V, and runs
    one kernel. Nothing is held between layers or steps: a copy of every
    layer's prefix, held for the whole denoise, was several GiB per request
    outside the pool, and the gather moves the same bytes the concatenation
    of a held copy did. So a fork, a reset or a page-table move has nothing
    here to invalidate.

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

    # it holds nothing per request
    supports_drop = True

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
        wrapper_class: type[PrefixedPrefillWrapper],
        flashinfer_backend: str = "auto",
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._on_cuda = device.type == "cuda" and torch.cuda.is_available()

        # plan label -> this step's layout
        self._current_plans: dict[str, DensePlan] = {}

        self._wrapper_class = wrapper_class
        self._wrappers: dict[EagerSlotKey, PrefixedPrefillWrapper] = {}
        self._wrapper_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            device=device,
            q_data_type=dtype,
        )
        self._on_flashinfer = issubclass(wrapper_class, _FlashInferRaggedPrefillBase)
        self._flashinfer_backend = flashinfer_backend if self._on_flashinfer else None
        if self._on_flashinfer:
            # One workspace for every wrapper, taken now rather than at the first
            # plan: it is scratch during `run` only (a plan writes the wrapper's
            # own buffers), and the runs are serialized on the node's stream. Off
            # CUDA nothing runs, so it is empty.
            buffer = (
                torch.empty(workspace_bytes(), dtype=torch.uint8, device=device) if self._on_cuda
                else torch.empty(0, dtype=torch.uint8)
            )
            self._wrapper_kwargs.update(workspace_buffer=Workspace(buffer, kv_cache), backend=flashinfer_backend)
            if self._on_cuda:
                self._check_flashinfer_backend()

    def fingerprint(self) -> bytes:
        # the kernel, not just the class: FA3 and each FlashInfer backend
        # differ in the last bits
        return fingerprint(type(self).__name__, self._wrapper_class.__name__, self._flashinfer_backend)

    def depends_on(self):
        return {self._kv_cache_name}

    def can_capture(self) -> bool:
        return False

    @property
    def force_double_buffer(self):
        # FlashInfer's plan stages its schedule in a pinned buffer per wrapper
        # and H2Ds it on the stream, so each slot keeps its own wrapper
        return self._on_flashinfer

    def _check_flashinfer_backend(self) -> None:
        """Plan and run a tiny layout on the configured kernel, so a backend
        this GPU or FlashInfer build cannot run stops the server at startup
        instead of failing every request."""
        cfg = self._kv_config
        try:
            wrapper = self._wrapper_class(**self._wrapper_kwargs)
            wrapper.plan(torch.tensor([0, 16], dtype=torch.int32), torch.tensor([0, 32], dtype=torch.int32))
            q = torch.zeros(16, cfg.num_qo_heads, cfg.head_dim, device=self._device, dtype=self._dtype)
            kv = torch.zeros(32, cfg.num_kv_heads, cfg.head_dim, device=self._device, dtype=self._dtype)
            wrapper.run(q, kv, kv)
            # a launch failure is reported asynchronously
            torch.cuda.synchronize(self._device)
        except Exception as exc:  # noqa: BLE001
            capability = current_platform.cuda_capability(self._device) if self._device.type == "cuda" else None
            raise ValueError(
                f"flashinfer_backend={self._flashinfer_backend!r} cannot run the dense attention "
                f"on this GPU (compute capability {'.'.join(map(str, capability or ()))}) with "
                f"FlashInfer {_flashinfer_version()}: {type(exc).__name__}: {exc}"
            ) from exc

    def _wrapper(self, label: str, slot: int) -> PrefixedPrefillWrapper:
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._wrappers.get(key)
        if wrapper is None:
            wrapper = self._wrappers[key] = self._wrapper_class(**self._wrapper_kwargs)
        return wrapper

    @property
    def requires_kv_write(self) -> bool:
        return False

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

        # Only the labels the step attends: a cfg rank's KV step holds every branch's label, and its attention
        # reads only its own branches' (a combined plan label carries its streams' labels on its views).
        attends = None if step.segments is None else {segment.label for segment in step.segments}
        self._current_plans = {
            plan_label: self._build_plan(kv_out, step.causal, EagerSlotKey(label=plan_label, slot=ctx.slot))
            for plan_label, kv_out in plan_outputs.items()
            if attends is None or any(view.label in attends for view in kv_out.views)
        }

    def _build_plan(self, kv_out: KVPlanOutput, causal: bool, slot_key: EagerSlotKey) -> DensePlan:
        """The per-segment gather layout plus the cumulative lengths one varlen
        kernel needs, in the packed order the KV plan defined."""
        page_size = self._kv_config.tokens_per_page
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
            ))
            cu_q.append(cu_q[-1] + view.to_compute)
            cu_k.append(cu_k[-1] + view.length)
            max_q = max(max_q, view.to_compute)
            max_k = max(max_k, view.length)

        # Fresh pinned host layouts per step: the FA3 wrapper H2Ds them non-blocking
        # (from pageable memory that copy would sync the stream), FlashInfer's plan
        # reads them on the host, and neither happens when the wrapper already
        # holds this layout.
        wrapper = self._wrapper(slot_key.label, slot_key.slot)
        pin = self._on_cuda
        wrapper.plan(
            torch.tensor(cu_q, dtype=torch.int32, pin_memory=pin),
            torch.tensor(cu_k, dtype=torch.int32, pin_memory=pin),
            causal=causal,
        )
        rows, rows_at = self._prefix_rows(segments, page_size)
        return DensePlan(
            segments=tuple(segments),
            cu_q=tuple(cu_q),
            cu_k=tuple(cu_k),
            max_q=max_q,
            max_k=max_k,
            causal=causal,
            wrapper=wrapper,
            rows=rows,
            rows_at=rows_at,
            prefix_rows=rows_at[-1] if rows_at else 0,
        )

    def _prefix_rows(
        self, segments: list[DenseSegment], page_size: int,
    ) -> tuple[torch.Tensor | None, tuple[int, ...]]:
        """The rows every layer's gather reads, built once per step and moved to the device in one copy from
        pinned memory (from pageable memory it would sync the stream)."""
        rows_at = [0]
        for segment in segments:
            rows_at.append(rows_at[-1] + segment.prefix_len)
        if not rows_at[-1]:
            return None, tuple(rows_at)
        slots = torch.arange(page_size, dtype=torch.long)
        k_rows = torch.cat([
            (torch.tensor(segment.pages, dtype=torch.long)[:, None] * (2 * page_size) + slots).flatten()[
                :segment.prefix_len
            ]
            for segment in segments if segment.prefix_len
        ])
        host = torch.cat([k_rows, k_rows + page_size])
        if self._on_cuda:
            host = host.pin_memory()
        return host.to(self._device, non_blocking=True), tuple(rows_at)

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

        # [prefix | fresh] per segment, in the plan's order
        key = k.new_empty((plan.cu_k[-1], *k.shape[1:]))
        val = v.new_empty((plan.cu_k[-1], *v.shape[1:]))
        offset = 0
        for segment, start, rows_at in zip(plan.segments, plan.cu_k, plan.rows_at, strict=False):
            fresh = start + segment.prefix_len
            if segment.prefix_len:
                rows = plan.rows[rows_at:rows_at + segment.prefix_len]
                _gather_rows(key[start:fresh], kv_cache_layer, rows)
                rows = plan.rows[plan.prefix_rows + rows_at:plan.prefix_rows + rows_at + segment.prefix_len]
                _gather_rows(val[start:fresh], kv_cache_layer, rows)
            _copy_rows(key[fresh:fresh + segment.q_len], k[offset:offset + segment.q_len])
            _copy_rows(val[fresh:fresh + segment.q_len], v[offset:offset + segment.q_len])
            offset += segment.q_len
        return plan.wrapper.run(q, key, val)

