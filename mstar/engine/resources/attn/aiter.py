"""Paged attention on AMD GPUs with AITER's Triton unified attention.

One launch per layer covers a mixed batch (prefill, extend, decode) and reads
the paged cache in place. The kernel wants a dense block table instead of
FlashInfer's CSR, so ``plan`` converts the KV plan's indptrs and stages them
from pinned host memory in one copy per label.

The kernel is causal only. A single query sees its whole row under the causal
mask, so non-causal steps with one query per row run here too, and so does
cross-attention (one kernel row per query token). Non-causal steps with longer
rows (DiT denoise, prefix-LM) go to AMD's FlashInfer build (fa2), because
AITER 0.1.25's paged CK varlen kernel fails to generate its module
(ROCm/aiter#4456).

A combined walk's step (decode rows beside a prompt) is one launch too, so this
backend ignores ``AttentionConfig.splits``, and a packed capture plans for its
whole token count whatever its rows hold.

AITER and FlashInfer are imported on first use. Nothing here runs on NVIDIA.
"""

import functools
import importlib
import logging
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.attn.base import AttentionManager, EagerSlotKey
from mstar.engine.resources.attn.config import AttentionStep, AttnBackend, refuse_context_only
from mstar.engine.resources.attn.cross import CrossAttentionManager
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutputs, PagedIndptrs
from mstar.engine.resources.step import SlotLease, StepContext

logger = logging.getLogger(__name__)

# The kernel reads the block table one KV tile at a time, so a row's table must
# cover its last tile even past the row's last page. AITER's tiles are at most
# this many tokens.
_TILE_TOKENS = 128
# in int32s, so each section of a staged plan starts 64-byte aligned
_ALIGN = 16
# dtypes and head dims checked against a torch reference
_DTYPES = (torch.bfloat16, torch.float16)
_MIN_HEAD_DIM, _MAX_HEAD_DIM = 32, 256
# GQA groups FlashInfer-ROCm's fa2 decode dispatches without tensor cores
FLASHINFER_ROCM_DECODE_GROUPS = frozenset({1, 2, 3, 4, 8})


@functools.cache
def aiter_paged_unavailable_reason() -> str | None:
    """Why AITER's unified attention can't be imported, or None."""
    try:
        from aiter.ops.triton.attention.unified_attention import (  # noqa: F401
            unified_attention,
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def flashinfer_rocm_unavailable_reason() -> str | None:
    """Why AMD's FlashInfer build can't be imported, or None.

    Upstream FlashInfer imports on ROCm and then fails on its first JIT build,
    so this also imports ``flashinfer.rocm``, which only the AMD fork has.
    """
    try:
        importlib.import_module("flashinfer")
        importlib.import_module("flashinfer.rocm")
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def nan_safe_page_size(page_size: int) -> bool:
    """Whether the kernel masks K/V loads past a row's length at this page size.

    The kernel skips that mask when its KV tile equals the page size. Its tiles
    are 16 to 64 tokens, so at a power-of-two page size up to 64, a NaN left in
    the tail of a row's last page reaches P·V (0 * NaN = NaN).
    """
    return not (page_size <= 64 and page_size & (page_size - 1) == 0)


def rocm_paged_backend(page_size: int | None) -> AttnBackend:
    """``AUTO`` on ROCm: AITER's unified attention, unless AITER reads page
    tails unmasked at this page size and AMD's FlashInfer build (which masks
    them) is installed."""
    aiter_reason = aiter_paged_unavailable_reason()
    flashinfer_reason = flashinfer_rocm_unavailable_reason()
    tails_safe = page_size is None or nan_safe_page_size(page_size)
    if aiter_reason is None and (tails_safe or flashinfer_reason is not None):
        return AttnBackend.AITER
    if flashinfer_reason is None:
        return AttnBackend.FLASHINFER
    raise RuntimeError(
        "No paged attention backend on this ROCm install: AITER's unified "
        f"attention is not importable ({aiter_reason}), nor is AMD's FlashInfer "
        f"build ({flashinfer_reason}). Install amd-aiter (AITER) or amd-flashinfer."
    )


def flashinfer_rocm_kwargs(kv_config: PagedKVConfig, backend: str) -> dict:
    """``FlashInferManager``'s arguments on ROCm, where only AMD's FlashInfer
    build runs. Its ``auto`` can't use an AITER kernel at page 128 (it falls
    back to fa2 after a failed attempt, or gathers per token for prefill), so
    ``auto`` is pinned to fa2. Decode skips tensor cores, the faster path on
    ROCm, for the GQA groups that kernel dispatches."""
    reason = flashinfer_rocm_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            "attention backend 'flashinfer' on ROCm needs AMD's FlashInfer build "
            f"(amd-flashinfer), which is not importable ({reason}); name backend "
            "'aiter' or 'auto' instead"
        )
    group = kv_config.num_qo_heads // kv_config.num_kv_heads
    return {
        "backend": "fa2" if backend == "auto" else backend,
        "decode_tensor_cores": group not in FLASHINFER_ROCM_DECODE_GROUPS,
    }


@functools.cache
def _unified_attention():
    from aiter.ops.triton.attention.unified_attention import unified_attention

    return unified_attention


def _aiter_unified_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> None:
    """Causal paged attention of ``q`` [tokens, Hq, D] into ``out``.

    ``max_seqlen_q`` and ``max_seqlen_k`` only pick the kernel variant and grid
    on the host. The kernel reads the real lengths from ``cu_seqlens_q`` and
    ``seqused_k``.
    """
    _unified_attention()(
        q=q, k=k_cache, v=v_cache, out=out,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=True, window_size=(-1, -1),
        block_table=block_table, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )


# One op with a fake, so a traced caller sees one node. Defined with `Library`
# instead of `torch.library.custom_op`, whose Python wrapper cost about 45 us
# per eager call on an MI300X host (this one costs about 14).
_LIB = torch.library.Library("mstar", "FRAGMENT")
_LIB.define(
    "aiter_unified_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor(a!) out, "
    "Tensor cu_seqlens_q, Tensor seqused_k, Tensor block_table, int max_seqlen_q, "
    "int max_seqlen_k, float softmax_scale) -> ()"
)
_LIB.impl("aiter_unified_attention", _aiter_unified_attention, "CUDA")


@torch.library.register_fake("mstar::aiter_unified_attention", lib=_LIB)
def _aiter_unified_attention_fake(
    q, k_cache, v_cache, out, cu_seqlens_q, seqused_k, block_table,
    max_seqlen_q, max_seqlen_k, softmax_scale,
) -> None:
    return None


def check_aiter_paged(kv_config: PagedKVConfig, dtype: torch.dtype) -> None:
    """Refuse at build time a cache the kernel wasn't checked for."""
    reason = aiter_paged_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            f"attention backend 'aiter' needs AITER's unified attention ({reason})"
        )
    if dtype not in _DTYPES:
        raise ValueError(
            f"attention backend 'aiter': KV dtype {dtype} is not supported; "
            f"use one of {[str(d) for d in _DTYPES]}"
        )
    if not _MIN_HEAD_DIM <= kv_config.head_dim <= _MAX_HEAD_DIM:
        raise ValueError(
            f"attention backend 'aiter': head_dim {kv_config.head_dim} is outside "
            f"[{_MIN_HEAD_DIM}, {_MAX_HEAD_DIM}]"
        )
    if not nan_safe_page_size(kv_config.page_size):
        logger.warning(
            "attention backend 'aiter' at page_size %d: the kernel does not mask "
            "the tail of a row's last page, so the KV pages must stay finite; "
            "page_size 128 or the 'flashinfer' backend does not need that.",
            kv_config.page_size,
        )


def _round(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class _Staged(NamedTuple):
    """One plan label's kernel inputs, device views plus the host ints that
    pick the kernel variant. Built per label per step, hence a NamedTuple."""
    cu_seqlens_q: torch.Tensor  # [rows + 1]
    seqused_k: torch.Tensor  # [rows]
    block_table: torch.Tensor  # [rows, width]
    max_seqlen_q: int
    max_seqlen_k: int
    num_tokens: int  # query tokens the kernel reads (q can be longer, padded to a bucket)


class _PlanImage:
    """Pinned host buffer for one plan's int32 inputs, ``[cu_seqlens_q |
    seqused_k | block_table]``, and the device buffer it is sent to in one copy.

    Every plan for the same key rewrites it in place. Table entries past a row's
    pages and tail tile keep stale values from earlier plans, which is fine
    because the kernel never reads them.
    """

    def __init__(self, numel: int, device: torch.device):
        pinned = device.type == "cuda" and torch.cuda.is_available()
        self.numel = numel
        self._host = torch.zeros(numel, dtype=torch.int32, pin_memory=pinned)
        self.np = self._host.numpy()
        self.dev = torch.zeros(numel, dtype=torch.int32, device=device)
        self._copied = torch.cuda.Event() if pinned else None
        self._pending = False

    def wait(self) -> None:
        """Block until the last copy out of the host buffer has finished."""
        if self._pending:
            self._copied.synchronize()
            self._pending = False

    def send(self, n: int) -> None:
        if self._copied is None:
            self.dev[:n].copy_(self._host[:n])
            return
        self.dev[:n].copy_(self._host[:n], non_blocking=True)
        self._copied.record(torch.cuda.current_stream(self.dev.device))
        self._pending = True


def _image_numel(rows: int, width: int) -> int:
    return _round(rows + 1) + _round(rows) + rows * width


class _Rows(NamedTuple):
    """Kernel rows on the host: FlashInfer's CSR plus each row's KV length."""
    qo_indptr: np.ndarray
    kv_lens: np.ndarray
    kv_indptr: np.ndarray
    kv_indices: np.ndarray


def row_kv_lens(kv_indptr: np.ndarray, last_page_len: np.ndarray, page_size: int) -> np.ndarray:
    """Each row's KV length from FlashInfer's CSR. At least 1, since a row with
    no pages still points at ``SINK_PAGE``."""
    counts = kv_indptr[1:] - kv_indptr[:-1]
    lens = (counts - 1) * page_size + last_page_len
    return np.where(counts > 0, lens, 1).astype(np.int32)


def kernel_rows(ip: PagedIndptrs, page_size: int) -> _Rows:
    kv_indptr = ip.paged_kv_indptr.numpy()
    return _Rows(
        qo_indptr=ip.qo_indptr.numpy(),
        kv_lens=row_kv_lens(kv_indptr, ip.paged_kv_last_page_len.numpy(), page_size),
        kv_indptr=kv_indptr,
        kv_indices=ip.paged_kv_indices.numpy(),
    )


def per_query_rows(rows: _Rows, num_rows: int | None = None) -> _Rows:
    """One kernel row per query token, over its request's pages.

    A single query attends every key of its row even under the causal mask, so
    this computes non-causal attention. ``num_rows`` pads to a captured
    bucket's row count with rows that point at ``SINK_PAGE``.
    """
    spans = np.diff(rows.qo_indptr)
    tokens = int(rows.qo_indptr[-1])
    counts = np.repeat(np.diff(rows.kv_indptr), spans)
    kv_lens = np.repeat(rows.kv_lens, spans)
    starts = np.repeat(rows.kv_indptr[:-1], spans)
    kv_indptr = np.zeros(tokens + 1, dtype=np.int64)
    np.cumsum(counts, out=kv_indptr[1:])
    nnz = int(kv_indptr[-1])
    kv_indices = rows.kv_indices[np.repeat(starts - kv_indptr[:-1], counts) + np.arange(nnz)]
    if num_rows is not None:
        if num_rows < tokens:
            raise RuntimeError(
                f"{tokens} query tokens do not fit the {num_rows} kernel rows a "
                "captured cross-attention step was sized for"
            )
        pad = num_rows - tokens
        kv_indptr = np.concatenate((kv_indptr, np.full(pad, nnz, dtype=np.int64)))
        kv_lens = np.concatenate((kv_lens, np.ones(pad, dtype=np.int32)))
        tokens = num_rows
    return _Rows(
        qo_indptr=np.arange(tokens + 1, dtype=np.int32),
        kv_lens=kv_lens, kv_indptr=kv_indptr, kv_indices=kv_indices,
    )


def stage_plan(
    image: _PlanImage, rows: _Rows, width: int, pad_cols: int,
    max_seqlen_q: int, max_seqlen_k: int,
) -> _Staged:
    """Write one plan into ``image``, send it, and return its device views.

    Each block table row holds the row's pages, then ``pad_cols`` entries of
    ``SINK_PAGE`` so its tail tile reads a valid page.
    """
    num_rows = rows.qo_indptr.shape[0] - 1
    kv_indptr = rows.kv_indptr
    counts = kv_indptr[1:] - kv_indptr[:-1]
    if num_rows and int(counts.max()) + pad_cols > width:
        raise RuntimeError(
            f"a row of {int(counts.max())} pages does not fit a block table "
            f"{width} wide"
        )
    k_off = _round(num_rows + 1)
    bt_off = k_off + _round(num_rows)
    image.wait()
    host = image.np
    host[:num_rows + 1] = rows.qo_indptr
    host[k_off:k_off + num_rows] = rows.kv_lens
    starts = bt_off + np.arange(num_rows, dtype=np.int64) * width
    nnz = int(kv_indptr[-1])
    if nnz:
        host[np.repeat(starts - kv_indptr[:-1], counts) + np.arange(nnz)] = rows.kv_indices
    tails = starts + counts
    host[(tails[:, None] + np.arange(pad_cols)).ravel()] = SINK_PAGE
    image.send(int(tails[-1]) + pad_cols if num_rows else bt_off)
    dev = image.dev
    return _Staged(
        cu_seqlens_q=dev[:num_rows + 1],
        seqused_k=dev[k_off:k_off + num_rows],
        block_table=dev[bt_off:bt_off + num_rows * width].view(num_rows, width),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_tokens=int(rows.qo_indptr[-1]),
    )


def single_query_rows(qo_indptr: np.ndarray) -> bool:
    """Whether every row of a plan has one query token."""
    return bool(
        qo_indptr[-1] == qo_indptr.shape[0] - 1 and (np.diff(qo_indptr) == 1).all()
    )


class _UnifiedPlans:
    """Staging and the kernel call, shared by both AITER managers."""

    _device: torch.device
    _dtype: torch.dtype
    _scale: float
    _cg_images: dict
    _eager_images: dict

    def _eager_image(self, label: str, slot: int, numel: int) -> _PlanImage:
        key = EagerSlotKey(label=label, slot=slot)
        image = self._eager_images.get(key)
        if image is None or image.numel < numel:
            if image is not None:
                # its pinned memory goes with it, so no copy may still read it
                image.wait()
            image = self._eager_images[key] = _PlanImage(max(numel, 4096), self._device)
        return image

    def _cg_image(self, lease: SlotLease, label: str, numel: int) -> _PlanImage:
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        image = self._cg_images.get(key)
        if image is None:
            image = self._cg_images[key] = _PlanImage(numel, self._device)
        elif image.numel != numel:
            raise RuntimeError(
                f"AITER attention: bucket {lease.bucket} label {label!r} now "
                "plans a different number of kernel rows than it was captured with"
            )
        return image

    def _unified(
        self, q: torch.Tensor, plan: _Staged, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        # a q shorter than the plan is read past its end, which shows up
        # later as an async memory fault
        assert q.shape[0] >= plan.num_tokens, (
            f"AITER attention: q has {q.shape[0]} tokens, the plan covers "
            f"{plan.num_tokens}"
        )
        qk = q if q.dtype == self._dtype else q.to(self._dtype)
        out = torch.empty(qk.shape, dtype=self._dtype, device=qk.device)
        torch.ops.mstar.aiter_unified_attention(
            qk, kv_cache_layer[:, 0], kv_cache_layer[:, 1], out,
            plan.cu_seqlens_q, plan.seqused_k, plan.block_table,
            plan.max_seqlen_q, plan.max_seqlen_k, self._scale,
        )
        return out if out.dtype == q.dtype else out.to(q.dtype)


# marks a plan label that the non-causal fallback runs this step
_DELEGATED = object()


class AiterPagedAttentionManager(_UnifiedPlans, AttentionManager):
    """Paged attention on AITER's unified attention (see the module docstring).

    A captured step stages into static buffers per (bucket, slot, label). The
    block table is as wide as the pool, and ``max_seqlen_q``/``max_seqlen_k``
    are pinned to the bucket's query length and the cache's max context, so
    every replay needs the kernel variant its capture recorded. An eager step
    stages into per-(label, slot) buffers sized to the step, with the batch's
    own maxima.
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
    ):
        check_aiter_paged(kv_config, dtype)
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._scale = kv_config.head_dim ** -0.5
        self._pad_cols = -(-_TILE_TOKENS // kv_config.page_size)
        self._cg_width = kv_config.max_num_pages + self._pad_cols

        self._cg_images: dict[CGSlotKey, _PlanImage] = {}
        self._eager_images: dict[EagerSlotKey, _PlanImage] = {}
        # the (kernel rows, max_seqlen_q) or fallback each captured key was
        # first planned with, since a replay must run what its capture recorded
        self._cg_kinds: dict[CGSlotKey, object] = {}

        self._current_plans: dict[str, object] = {}
        self._preplan_plans: dict[str, object] = {}
        self._preplanned = False
        self._preplan_delegated = False
        # FlashInfer manager for non-causal steps with multi-query rows
        self._fallback: AttentionManager | None = None

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # plan copies into device buffers that a replay reads
        return True

    def _noncausal_fallback(self) -> AttentionManager:
        if self._fallback is None:
            reason = flashinfer_rocm_unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "attention backend 'aiter': a non-causal step with more than "
                    "one query per row needs AMD's FlashInfer build, which is not "
                    f"importable ({reason}); AITER's unified attention is causal only"
                )
            from mstar.engine.resources.attn.flashinfer import FlashInferManager

            cfg = self._kv_config
            self._fallback = FlashInferManager(
                kv_cache=self._kv_cache_name, device=self._device,
                dtype=self._dtype, kv_config=cfg, backend="fa2",
                decode_tensor_cores=(
                    cfg.num_qo_heads // cfg.num_kv_heads
                    not in FLASHINFER_ROCM_DECODE_GROUPS
                ),
            )
        return self._fallback

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager buffers are per slot, "
            "like the captured ones"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # this step was preplanned, so promote it
            if self._preplan_delegated:
                self._fallback.plan(step, ctx)
            self._current_plans = self._preplan_plans
            self._preplan_plans = {}
            self._preplanned = False
            self._preplan_delegated = False
            return self._current_plans

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"AITER attention expected plan result from {self._kv_cache_name}"
        )
        page_size = self._kv_config.page_size
        rows = {
            label: kernel_rows(out.cpu_indptrs, page_size)
            for label, out in plan_outputs.items()
        }
        delegate = not step.causal and not all(
            single_query_rows(r.qo_indptr) for r in rows.values()
        )
        if delegate:
            if lease is not None:
                for label in rows:
                    self._check_kind(lease, label, "flashinfer fa2")
            self._noncausal_fallback().plan(step, ctx)
            plans = dict.fromkeys(rows, _DELEGATED)
        elif lease is not None:
            plans = {
                label: self._stage_captured(
                    label, r, lease, ctx.force_prefill, step.row_queries,
                )
                for label, r in rows.items()
            }
        else:
            plans = {label: self._stage_eager(label, r, ctx.slot) for label, r in rows.items()}

        if ctx.is_preplan:
            self._preplan_plans = plans
            self._preplan_delegated = delegate
            self._preplanned = True
        else:
            self._current_plans = plans
        # label -> `_Staged` (or the fallback's marker): the device views a
        # speculative step rewrites in-graph (`SpecDecodeResource`)
        return plans

    def _check_kind(self, lease: SlotLease, label: str, kind: object) -> None:
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        first = self._cg_kinds.setdefault(key, kind)
        if first != kind:
            raise RuntimeError(
                f"AITER attention: bucket {lease.bucket} label {label!r} was "
                f"captured as {first} and is now planned as {kind}; the graph "
                "would replay the wrong kernel"
            )

    def _stage_captured(
        self, label: str, rows: _Rows, lease: SlotLease, force_prefill: bool,
        row_queries: int | None = None,
    ) -> _Staged:
        num_rows = rows.qo_indptr.shape[0] - 1
        # a decode bucket always has one token per row, while one row of a
        # packed bucket can hold all of its tokens, even when the bucket has
        # as many rows as tokens and replays mixed steps (`force_prefill`)
        num_tokens = lease.bucket.num_tokens
        max_q = 1 if num_rows == num_tokens and not force_prefill else num_tokens
        if row_queries is not None and not force_prefill:
            # every row declared the same query count (a verify step)
            max_q = row_queries
            if num_rows and int(np.diff(rows.qo_indptr).max()) > row_queries:
                raise RuntimeError(
                    f"AITER attention: bucket {lease.bucket} planned a row with "
                    f"more than the {row_queries} queries its step declared"
                )
        if max_q == 1 and not single_query_rows(rows.qo_indptr):
            raise RuntimeError(
                f"AITER attention: decode bucket {lease.bucket} planned a row "
                "with more than one query token"
            )
        self._check_kind(lease, label, (num_rows, max_q))
        image = self._cg_image(lease, label, _image_numel(num_rows, self._cg_width))
        return stage_plan(
            image, rows, self._cg_width, self._pad_cols,
            max_q, self._kv_config.max_seq_len,
        )

    def _stage_eager(self, label: str, rows: _Rows, slot: int) -> _Staged:
        num_rows = rows.qo_indptr.shape[0] - 1
        counts = np.diff(rows.kv_indptr)
        width = (int(counts.max()) if num_rows else 0) + self._pad_cols
        image = self._eager_image(label, slot, _image_numel(num_rows, width))
        return stage_plan(
            image, rows, width, self._pad_cols,
            int(np.diff(rows.qo_indptr).max()) if num_rows else 1,
            int(rows.kv_lens.max()) if num_rows else 1,
        )

    def clear_preplan(self):
        if self._preplan_delegated and self._fallback is not None:
            self._fallback.clear_preplan()
        # rebind instead of clear, since a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_delegated = False
        self._preplan_plans = {}

    ### Submodule-level functionality
    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        plan = self._current_plans.get(label)
        if plan is None:
            return None
        if plan is _DELEGATED:
            return self._fallback.qo_indptr_buf(label)
        return plan.cu_seqlens_q

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str = "main"
    ) -> torch.Tensor:
        """Last token of each request's hidden states, for sampling a prefill."""
        last_token_indices = (self.qo_indptr_buf(label)[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # the layer already wrote this step's K/V to the pages (requires_kv_write)
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        plan = self._current_plans[label]
        if plan is _DELEGATED:
            return self._fallback.run(q, label, kv_cache_layer)
        return self._unified(q, plan, kv_cache_layer)


class AiterCrossManager(_UnifiedPlans, CrossAttentionManager):
    """Non-causal attention over a write-once context, on AITER's unified
    attention with one kernel row per query token (see ``per_query_rows``).

    The block table grows with the query span, which cross-attention keeps
    short (one token per request at decode). A captured step has one row per
    bucket token, the padding rows pointing at ``SINK_PAGE``.
    """

    def __init__(
        self,
        kv_cache: str,
        query_kv_cache: str | None,
        context_label: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
    ):
        check_aiter_paged(kv_config, dtype)
        self._kv_cache_name = kv_cache
        self._query_kv_cache_name = query_kv_cache
        self._context_label = context_label
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._scale = kv_config.head_dim ** -0.5
        self._pad_cols = -(-_TILE_TOKENS // kv_config.page_size)
        self._cg_width = kv_config.max_num_pages + self._pad_cols

        self._cg_images: dict[CGSlotKey, _PlanImage] = {}
        self._eager_images: dict[EagerSlotKey, _PlanImage] = {}
        self._current_plans: dict[str, _Staged] = {}
        self._preplan_plans: dict[str, _Staged] = {}
        self._preplanned = False

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager buffers are per slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "cross attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            self._current_plans = self._preplan_plans
            self._preplan_plans = {}
            self._preplanned = False
            return

        context_views = self._context_views(ctx)
        page_size = self._kv_config.page_size
        plans = {}
        for plan_label, packing in self._query_packings(step, ctx).items():
            request_rows = kernel_rows(
                self._build_indptrs(packing, context_views, plan_label), page_size,
            )
            if lease is not None:
                num_rows = lease.bucket.num_tokens
                image = self._cg_image(
                    lease, plan_label, _image_numel(num_rows, self._cg_width),
                )
                plans[plan_label] = stage_plan(
                    image, per_query_rows(request_rows, num_rows),
                    self._cg_width, self._pad_cols, 1, self._kv_config.max_seq_len,
                )
                continue
            rows = per_query_rows(request_rows)
            num_rows = rows.qo_indptr.shape[0] - 1
            width = (int(np.diff(rows.kv_indptr).max()) if num_rows else 0) + self._pad_cols
            image = self._eager_image(plan_label, ctx.slot, _image_numel(num_rows, width))
            plans[plan_label] = stage_plan(
                image, rows, width, self._pad_cols,
                1, int(rows.kv_lens.max()) if num_rows else 1,
            )
        if ctx.is_preplan:
            self._preplan_plans = plans
            self._preplanned = True
        else:
            self._current_plans = plans

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_plans = {}

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer's cross attention. ``kv_cache_layer`` is a layer of the
        context cache, and is only read."""
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        return self._unified(q, self._current_plans[label], kv_cache_layer)


def build_aiter_paged(spec, info, kv_config) -> AiterPagedAttentionManager:
    """The ``aiter`` row of ``attn.backends``."""
    return AiterPagedAttentionManager(
        kv_cache=spec.config.kv_cache,
        device=info.device,
        dtype=info.kv_dtype,
        kv_config=kv_config,
    )
