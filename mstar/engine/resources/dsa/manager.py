"""DeepSeek sparse attention (DSA) over the paged MLA cache: the index keys, each query's top-k
and the attention over it.

A model's FULL indexer layers hand this resource their tokens' index keys (``store_keys``) and
their queries and head weights (``select``): each query row scores its request's keys,
``sum_h w[h] * relu(q[h] . k)`` in fp32, and keeps its top ``topk`` positions. SHARED layers
reuse the last FULL layer's pick (``shared_selection``), and ``attend`` runs absorbed MLA over
just the picked latents of the MLA cache. A step's rows come from the KV plan: each request's
rows, the position they start at and its pages.

The keys live in per-request buffers that grow with the context and the selection runs per
request on the host, so the path is eager only; past ``topk`` it decodes (one row per request).
"""

from dataclasses import dataclass

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.dsa.config import DSAConfig, DSASpec, DSAStep
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager

_INITIAL_CAPACITY = 64
_MAIN = "main"


class IndexKeyStore:
    """Per-request, per-layer growing buffers of the indexer's keys, in position order."""

    def __init__(self) -> None:
        # request_id -> layer_idx -> [buffer (capacity, head_dim), filled rows]
        self._buffers: dict[str, dict[int, list]] = {}

    def append(
        self,
        request_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Append one chunk of ``(n, head_dim)`` keys at positions ``start_pos ..``."""
        per_layer = self._buffers.setdefault(request_id, {})
        num_new, head_dim = keys.shape
        entry = per_layer.get(layer_idx)
        if entry is None:
            capacity = max(_INITIAL_CAPACITY, num_new)
            entry = per_layer[layer_idx] = [keys.new_empty((capacity, head_dim)), 0]
        buffer, filled = entry
        if start_pos != filled:
            raise RuntimeError(
                f"DSA key store desync for request {request_id!r} layer {layer_idx}: "
                f"append at position {start_pos} but {filled} keys are stored"
            )
        needed = filled + num_new
        if needed > buffer.shape[0]:
            grown = buffer.new_empty((max(needed, 2 * buffer.shape[0]), head_dim))
            grown[:filled] = buffer[:filled]
            entry[0] = buffer = grown
        buffer[filled:needed] = keys.detach()
        entry[1] = needed

    def history(self, request_id: str, layer_idx: int, length: int) -> torch.Tensor:
        """The first ``length`` stored keys — a view, not a copy. Raises if
        fewer are stored (a scoring window must never exceed the appended
        history: the current chunk is appended before selection)."""
        entry = self._buffers[request_id][layer_idx]
        buffer, filled = entry
        if length > filled:
            raise RuntimeError(
                f"DSA key store for request {request_id!r} layer {layer_idx} holds "
                f"{filled} keys; selection window wants {length}"
            )
        return buffer[:length]

    def evict(self, request_id: str) -> None:
        """Drop all of a request's buffers. Idempotent — the engine may retire
        a request that never reached a DSA forward."""
        self._buffers.pop(request_id, None)

    # -- introspection (tests + leak asserts) ---------------------------------

    def tracked_requests(self) -> set[str]:
        return set(self._buffers)

    def tokens(self, request_id: str, layer_idx: int) -> int:
        """Stored rows for one (request, layer); 0 when absent."""
        entry = self._buffers.get(request_id, {}).get(layer_idx)
        return 0 if entry is None else entry[1]


def select_topk_causal(
    scores: torch.Tensor, positions: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per-row top-``topk`` key positions of causally masked ``scores``.

    ``scores`` is ``(T, num_keys)`` with every key past a row's position
    already ``-inf``; the result is ``(T, topk)`` int32, -1 padded where a
    row sees fewer than ``topk`` keys. One stable sort for the whole batch:
    no per-token topk launch, no host sync. Ties break toward the earlier
    key position, so a row's selection does not depend on how the batch
    is packed or on which device ran it (torch.topk's tie order is neither).
    """
    num_tokens, num_keys = scores.shape
    k_eff = min(topk, num_keys)
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, :k_eff]
    # the -inf tail sorts last (stable: by ascending index), so rank >= p_t+1
    # is exactly the invisible remainder
    rank = torch.arange(k_eff, device=scores.device)
    invisible = rank.unsqueeze(0) >= positions.unsqueeze(1) + 1
    selection = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=scores.device)
    selection[:, :k_eff] = order.to(torch.int32).masked_fill(invisible, -1)
    return selection


def index_topk(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor,
    positions: torch.Tensor, topk: int,
) -> torch.Tensor:
    """Top-``topk`` key positions per query row, ``(T, topk)`` int32, -1 padded.

    ``q`` ``(T, H, D)`` and ``weights`` ``(T, H)`` are the indexer's queries and head weights
    (any scales folded in), ``keys`` ``(S, D)`` a request's keys by position and ``positions``
    ``(T,)`` the rows' own: a row sees keys ``0 .. position``, itself included."""
    # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]), in fp32: per-head ReLU BEFORE the
    # weighted sum; the raw weights get no softmax/sigmoid. The engine's bf16 autocast would run
    # both einsums in bf16.
    with torch.autocast(q.device.type, enabled=False):
        dots = torch.einsum("thd,sd->ths", q.float(), keys.float()).relu()
        scores = torch.einsum("th,ths->ts", weights.float(), dots)
    key_pos = torch.arange(keys.shape[0], device=scores.device)
    scores = scores.masked_fill(key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))
    return select_topk_causal(scores, positions, topk)


@dataclass
class DSASpan:
    """One request's rows of a step: ``q_len`` rows from row ``q_start``, the first at position
    ``ctx_start``; ``pages`` its MLA cache pages, ``page_start`` its first token's offset."""

    request_id: str
    q_start: int
    q_len: int
    ctx_start: int
    pages: list[int]
    page_start: int = 0


@dataclass
class DSAPlan:
    """A label's rows this step. ``needs_selection``: some row sees more than ``topk`` keys;
    ``last_selection``: the last FULL layer's pick this forward, which SHARED layers reuse."""

    spans: list[DSASpan]
    needs_selection: bool
    last_selection: torch.Tensor | None = None


class DSAManager(Resource):
    """DSA for one node's layers over its MLA cache (``config.kv_cache``)."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config
        kv_spec = info.dependency(spec.config.kv_cache)
        return cls(spec.config, page_size=kv_spec.config.page_size, device=info.device)

    def __init__(self, config: DSAConfig, page_size: int, device: torch.device):
        self.config = config
        self.page_size = page_size
        self._device = device
        self.keys = IndexKeyStore()
        self._plans: dict[str, DSAPlan] = {}
        # each decode row's pages, one pinned H2D each
        self._page_stager = PinnedStager(torch.long)

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def remove_request(self, rid: str) -> None:
        self.keys.evict(rid)

    def reset_request(self, rid: str, free: bool = False) -> None:
        self.keys.evict(rid)

    def plan(self, step: DSAStep, ctx: StepContext) -> dict[str, DSAPlan]:
        """Each label's rows, as the KV plan placed them: a request's rows follow the one
        before it, and its first sits at its length before the step."""
        del step
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self.config.kv_cache)
        assert plan_outputs is not None, f"DSA expected a plan result from {self.config.kv_cache}"
        self._plans = {}
        for label, kv_out in plan_outputs.items():
            spans, q_start = [], 0
            for view in kv_out.views:
                spans.append(DSASpan(
                    request_id=view.request_id, q_start=q_start, q_len=view.to_compute,
                    ctx_start=view.length - view.to_compute, pages=list(view.page_idxs),
                    page_start=view.start,
                ))
                q_start += view.to_compute
            needs = any(view.length > self.config.topk for view in kv_out.views)
            self._plans[label] = DSAPlan(spans=spans, needs_selection=needs)
        return self._plans

    def needs_selection(self, label: str = _MAIN) -> bool:
        """Whether this step's rows attend a pick rather than every key."""
        return self._plans[label].needs_selection

    # Host-side per-request work (dict mutation, python loops over spans): keep dynamo out of
    # it, as the cache managers' plan / run do, so the engine's warmup torch.compile runs these
    # eagerly instead of graph-breaking token by token.
    @torch.compiler.disable
    def store_keys(self, layer_idx: int, keys: torch.Tensor, label: str = _MAIN) -> None:
        """Store the step's ``(T, D)`` index keys of one layer, each request's at its positions."""
        for span in self._plans[label].spans:
            self.keys.append(
                span.request_id, layer_idx, keys[span.q_start:span.q_start + span.q_len],
                start_pos=span.ctx_start,
            )

    @torch.compiler.disable
    def select(
        self, layer_idx: int, q: torch.Tensor, weights: torch.Tensor, label: str = _MAIN,
    ) -> torch.Tensor:
        """Each row's top ``topk`` positions of its request (``index_topk``) over this layer's
        stored keys, ``(T, topk)`` int32, -1 padded; SHARED layers reuse it this forward."""
        plan = self._plans[label]
        rows = []
        for span in plan.spans:
            rows_of = slice(span.q_start, span.q_start + span.q_len)
            keys = self.keys.history(span.request_id, layer_idx, span.ctx_start + span.q_len)
            positions = torch.arange(
                span.ctx_start, span.ctx_start + span.q_len, device=q.device)
            rows.append(index_topk(q[rows_of], weights[rows_of], keys, positions,
                                   self.config.topk))
        selection = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        plan.last_selection = selection
        return selection

    def shared_selection(self, layer_idx: int, label: str = _MAIN) -> torch.Tensor:
        """The last FULL layer's pick this forward, for a SHARED layer (IndexShare)."""
        selection = self._plans[label].last_selection
        if selection is None:
            raise RuntimeError(
                f"SHARED layer {layer_idx} needs a DSA selection but no FULL layer ran before "
                "it this forward: the model's first indexer layer must be FULL"
            )
        return selection

    @torch.compiler.disable
    def attend(
        self,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        latent_layer: torch.Tensor,
        sm_scale: float,
        label: str = _MAIN,
    ) -> torch.Tensor:
        """Absorbed MLA of each row over its picked latents, ``(T, H, kv_lora_rank)``.

        ``latent_layer`` is the MLA cache's layer view ``[pages, page_size, kv_lora_rank +
        rope]``, this step's latents already written; ``selection`` holds request-local
        positions (-1: none). Past ``topk`` a request decodes, one row per request."""
        rank = q_nope.shape[-1]  # the ckv width, after w_kc absorption
        query = torch.cat([q_nope, q_pe], dim=-1).float()  # (T, H, rank + rope)
        out = torch.empty_like(q_nope)
        for span in self._plans[label].spans:
            if span.q_len != 1:
                raise RuntimeError(
                    f"request {span.request_id!r}: sparse attention beyond topk is "
                    "decode-only (one row per request); a prefill past topk is refused "
                    "before the step"
                )
            row = selection[span.q_start]
            live = row >= 0
            slots = row.clamp_min(0).long() + span.page_start  # request-local -> page offset
            pages = self._page_stager.to_device(span.pages, latent_layer.device)
            picked = latent_layer[pages[slots // self.page_size], slots % self.page_size]
            # fp32 outside the engine's bf16 autocast, which would run both einsums in bf16
            with torch.autocast(query.device.type, enabled=False):
                scores = torch.einsum("hd,kd->hk", query[span.q_start], picked.float()) * sm_scale
                attn = scores.masked_fill(~live, float("-inf")).softmax(dim=-1)
                out[span.q_start] = torch.einsum(
                    "hk,kd->hd", attn, picked[:, :rank].float()).to(out.dtype)
        return out
