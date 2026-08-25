"""Attention as one dense FlashAttention-3 varlen pass."""

import functools
import logging
from dataclasses import dataclass, field

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import StepContext

logger = logging.getLogger(__name__)


@functools.cache
def _fa3_unavailable_reason() -> str | None:
    """verify flash attention 3 fwd kernel imports"""
    try:
        import fa3_fwd_interface  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def _warn_dense_fallback(reason: str) -> None:
    logger.warning(
        "Attention backend 'dense' requested but the fa3-fwd kernel is "
        "unavailable (%s); using the paged 'flashinfer' backend instead.",
        reason,
    )


@dataclass(frozen=True)
class DenseSegment:
    """(request, label) of dense plan; frozen prefix and fresh tokens"""
    request_id: str
    label: str
    pages: torch.Tensor  # the prefix's pages, in stream order
    prefix_len: int
    q_len: int
    generation: int  # of the stream, at plan time; see CacheStream.generation


@dataclass(frozen=True)
class DensePlan:
    """dense plan layout. once per step and read by each layer `run`"""
    segments: tuple[DenseSegment, ...]
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    causal: bool


@dataclass
class _GatheredPrefix:
    """One stream's gathered prefix, per layer, tagged with what it was read
    from so a stale gather cannot be served."""
    """gathered prefix of one stream per-layer. tagged with what read from so stale cannot serve"""
    generation: int
    length: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


class DenseAttentionManager(AttentionManager):
    """Attention as one dense FlashAttention-3 varlen pass over a contiguous
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
        kv_config: KVConfig,
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
        self.reset_default_cursors()
        assert ctx.slot_lease is None and not ctx.is_preplan, (
            "dense attention is eager-only: the prefix gather and the "
            "concatenation allocate and are shaped by the step, so there is "
            "nothing for a captured graph to replay"
        )
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"Dense Attention Manager expected plan result from {self._kv_cache_name}"
        )

        self._current_plans = {
            plan_label: self._build_plan(kv_out, step.causal)
            for plan_label, kv_out in plan_outputs.items()
        }

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
                pages=torch.tensor(
                    view.page_idxs[:n_pages], dtype=torch.long, device=self._device
                ),
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
            cu_q=torch.tensor(cu_q, dtype=torch.int32, device=self._device),
            cu_k=torch.tensor(cu_k, dtype=torch.int32, device=self._device),
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
        pages = kv_cache_layer[segment.pages]
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
        # `plan` refuses a leased slot, which covers every path that plans. A
        # piecewise region declaring `reuses_outer_plan` never plans, so its
        # capture would reach here and bake this step's gather, its shapes and
        # the address of a cached prefix into the graph; the first invalidation
        # re-gathers into a new tensor the replay does not read. Silent, so
        # raise rather than assert.
        if self._on_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dense attention cannot be captured: the prefix gather and "
                "the concatenation are shaped by the step and allocate. A "
                "piecewise region with `reuses_outer_plan` is the path that "
                "gets here — capture that region against the paged backend."
            )
        from fa3_fwd_interface import flash_attn_varlen_func

        plan = self._current_plans[label]
        q = q.to(self._dtype)
        k = k.to(self._dtype)
        v = v.to(self._dtype)

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

        out = flash_attn_varlen_func(
            q, torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0),
            plan.cu_q, plan.cu_k, plan.max_q, plan.max_k,
            # a causal plan aligns the query block to the end of the key
            # sequence, so the fresh tokens see the whole prefix; only
            # non-causal generation attention is exercised today
            causal=plan.causal,
        )
        return out[0] if isinstance(out, tuple) else out
