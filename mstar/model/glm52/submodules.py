"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.engine.cuda_graph_config import (
    FlashInferPackedCudaGraphConfig,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
    PiecewiseCudaGraphConfig,
)
from mstar.engine.cuda_graph_runner import BasicBatchedCudaGraphConfig
from mstar.engine.kv_store import PositionInfo
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.dsa import (
    Glm52DsaForwardContext,
    Glm52DsaKStore,
    Glm52DsaRequestSpan,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.sampling import Sampler

logger = logging.getLogger(__name__)

_MAIN = "main"
# Piecewise-graph label for the MTP decode step's trunk verify forward.
MTP_TRUNK_LABEL = "mtp_trunk"
MTP_DRAFT_LABEL = "mtp_draft"
MTP_SYNC_LABEL = "mtp_sync"
# Output/edge name for the prefill's [emitted, k drafts] bundle. Deliberately
# NOT "text_inputs": the conductor seeds persist_signals from initial_signals,
# where "text_inputs" is the PROMPT, so a persisted edge of that name is
# indistinguishable from the prompt at the prefill->decode transition — which
# fed decode the whole prompt back as its first step (measured 2026-08-10).
MTP_DRAFT_BUNDLE = "mtp_draft_bundle"


def mtp_sync_padded_layout(
    e_list: list[int], starts: list[int], k: int,
) -> tuple[list[int], list[int], list[int]]:
    """Row layout for a PADDED MTP sync pass — the arithmetic behind capturing it.

    The sync pass extends the MTP plane over the tokens just committed. Each
    request contributes ``e = n_acc + 1`` rows, which is data-dependent, so the
    pass has no fixed shape and has stayed eager — ~10 of a 36.5 ms step. But
    that 10 ms is almost entirely *launch overhead*, not compute (the module is
    one decoder layer ≈ 0.3 ms, against a 79-layer 24 ms trunk), so padding
    every request out to the maximum ``k+1`` rows costs nearly nothing and buys
    a single fixed shape per batch size — the very shape ``mtp_trunk`` already
    captures. That beats one bucket per row count, which at bs > 1 would need a
    bucket per *composition* (bs=2,k=2 → 9), not per total.

    Returns ``(positions, last_rows, rewind)`` for ``rows = k+1`` per request:

    - ``positions`` — RoPE position per row, real rows first. Request i's real
      rows carry token positions ``start-e+1 .. start``; its pad rows continue
      monotonically past ``start`` so RoPE never sees a repeated or negative
      position. Pads land at plane slots ``>= start``, which is exactly the
      transient region the draft chain overwrites next, so they are harmless —
      the same guarantee the chain's own ``k-1`` extra entries already rely on.
    - ``last_rows`` — index of each request's LAST REAL row, the one draft 1
      comes from. With padding this is ``i*rows + e - 1``, not a cumsum.
    - ``rewind`` — per-request counter correction. The runner advances every
      captured request by ``rows``; only ``e`` of those are real, so rewinding
      ``rows - e`` restores the counter to ``start``, leaving the plane holding
      exactly ``position_id_start`` entries as the eager path does.

    Causality is unaffected: real rows precede pads within a request, so a real
    row never attends to a pad.
    """
    rows = k + 1
    positions: list[int] = []
    last_rows: list[int] = []
    for i, (st, e) in enumerate(zip(starts, e_list, strict=True)):
        if not 1 <= e <= rows:
            raise ValueError(f"sync rows e={e} outside [1, {rows}] for request {i}")
        positions.extend(range(st - e + 1, st + 1))
        positions.extend(range(st + 1, st + 1 + rows - e))
        last_rows.append(i * rows + e - 1)
    return positions, last_rows, [rows - e for e in e_list]


@dataclass(kw_only=True)
class Glm52MtpTrunkGraphConfig(PiecewiseCudaGraphConfig):
    """PACKED piecewise config with exactly one (bs, [k+1]*bs) bucket per
    capture batch size.

    Every MTP decode step feeds exactly k+1 rows per request (last emitted
    token + k drafts), so the generic ``PiecewisePackedConfig`` bs x
    token-bucket cross product would enumerate shapes that can never occur.
    PACKED (not BATCHED) because the trunk is row-packed — replay must
    slice outputs to the real token count and pad absent requests with
    zero-length plan rows rather than whole capture-length rows.
    """
    rows_per_request: int

    def get_config_type(self) -> PiecewiseConfigType:
        return PiecewiseConfigType.PACKED

    def get_capture_shapes(
        self, batch_sizes: list[int],
    ) -> list[PiecewiseCaptureShape]:
        return [
            PiecewiseCaptureShape(
                bs=bs,
                seq_lens=[self.rows_per_request] * bs,
                total_tokens=self.rows_per_request * bs,
            )
            for bs in batch_sizes
        ]


class Glm52LLMSubmodule(ARNodeSubmodule):
    """Embed + 78 decoder layers + lm_head, one fat TP node."""

    def __init__(self, language_model: nn.Module, config: Glm52ModelConfig):
        super().__init__()
        self.language_model = language_model  # Glm52ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        self._load_heartbeat_stop = None
        # DSA indexer k-cache (dsa.py): per-request bf16 index keys, appended
        # by FULL layers each forward when dsa_long_context is on. Owned here
        # so the engine's request lifecycle covers it — ``cleanup_request``
        # below evicts on retirement.
        self._dsa_k_store = Glm52DsaKStore()
        # M3 per-request draft-loop state (mtp_num_draft_tokens > 0 only):
        # total emitted tokens (incl. the prefill-emitted one — max_tokens
        # counts it, the M1 off-by-one lesson) and the stop parameters
        # stashed at prepare_inputs time so the batched step can truncate
        # emission without engine round-trips. All evicted in
        # ``cleanup_request``.
        self._mtp_emitted: dict[str, int] = {}
        self._mtp_max_tokens: dict[str, int] = {}
        self._mtp_ignore_eos: dict[str, bool] = {}
        # Which trunk stream the MTP plane pairs drafts against — see
        # ``_mtp_pair_rows``. Default keeps today's pre-final-norm behaviour;
        # MSTAR_GLM52_MTP_PAIR_POSTNORM=1 selects vLLM's convention for the A/B.
        self._mtp_pair_postnorm = (
            os.environ.get("MSTAR_GLM52_MTP_PAIR_POSTNORM", "0") == "1"
        )
        # Capture the decode sync pass as a padded (bs, k+1) piecewise graph.
        # DEFAULT OFF: measured 2026-08-10 at 33.02 tok/s with p1 acceptance
        # 0.18 vs the eager path's 49.65 / 0.76 — the padded replay is
        # producing a wrong draft 1 somewhere the CPU seam test cannot see
        # (its stub runner plans and runs eagerly on the real handle; the
        # real one replays a captured graph through static buffers and
        # aliased dummy states). An unvalidated capture must not be the
        # default path: the whole point of this switch is that a graph which
        # lowers acceptance looks exactly like a modelling problem.
        self._mtp_capture_sync = (
            os.environ.get("MSTAR_GLM52_MTP_CAPTURE_SYNC", "0") == "1"
        )
        # One-shot flag: warn the first time an MTP decode trunk runs eager
        # (no captured piecewise bucket) — the 2026-08-09 bench showed that
        # regression is 13x and silence lets it masquerade as "MTP is slow".
        self._mtp_trunk_eager_warned = False
        self._mtp_draft_eager_warned = False
        self._mtp_sync_eager_warned = False
        # Acceptance instrumentation (global, not per-request): raw emitted
        # tokens (n_accepted + 1, pre-truncation) and request-step count.
        self._mtp_stat_emitted = 0
        self._mtp_stat_steps = 0
        self._mtp_stat_logged = 0
        # Histogram of n_accepted per decode step (bins 0..k). The aggregate
        # rate can't separate "first draft mediocre" from "chained drafts
        # collapse" — the conditional per-position profile can.
        self._mtp_stat_acc_hist = [0] * (config.mtp_num_draft_tokens + 1)

    def set_load_heartbeat_stop(self, stop) -> None:
        """Adopt the load-time GPU liveness tick; stopped on first forward."""
        self._load_heartbeat_stop = stop

    def _stop_load_heartbeat(self) -> None:
        if self._load_heartbeat_stop is not None:
            self._load_heartbeat_stop.set()
            self._load_heartbeat_stop = None

    def cleanup_request(self, request_id: str):
        """Evict the request's DSA k-history alongside the base per-request
        state. ``KVCacheEngine.remove_request`` calls this for every managed
        submodule (the contract ``test_kv_cache_engine_cleanup.py`` pins);
        skipping it would leak ~5.4 KB/token/request of index keys per rank
        forever."""
        self._dsa_k_store.evict(request_id)
        self._mtp_emitted.pop(request_id, None)
        self._mtp_max_tokens.pop(request_id, None)
        self._mtp_ignore_eos.pop(request_id, None)
        super().cleanup_request(request_id)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the fused fp8 kernel."""
        from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

        lm = getattr(self, "language_model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm52SparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        """The reference MoE dispatch paths (fp8-resident per-hit-expert
        loop; naive bf16 loop under TP) use .nonzero()/host loops — illegal
        under CUDA-graph stream capture. Capturing would fail for every
        bucket, leave runner.graphs empty, and serve eagerly anyway;
        registering no configs makes eager-only serving explicit. The fused
        fused_experts_fp8 path IS capture-safe: when the loaded MoE blocks
        resolved to it (moe_quant_kernel triton/auto-on-cuda), graphs come
        back. Called post-load, so the resolved flag is authoritative.
        Shared by the full-forward and MTP-trunk piecewise config gates —
        the trunk forward contains the same MoE dispatch."""
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        return fp8_reference or naive_tp

    def _build_prefill_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros((num_tokens,), dtype=torch.long, device=device),
            "position_ids": torch.arange(num_tokens, dtype=torch.long, device=device),
        }

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts.

        get_submodule already established final per-param dtypes (bf16
        compute, fp32 block scales + router bias, uint8 fp8 bytes). The
        engine manager's blanket ``submodule.to(device, autocast_dtype)``
        would re-narrow the fp32 quantization params that
        ``restore_fp32_params`` widened — silently corrupting every routed-
        expert dequant and the top-8 selection — so dtype is dropped here.
        """
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm52LLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig | FlashInferPackedCudaGraphConfig]:
        if self.config.dsa_long_context:
            # DSA maintenance is host-side per-request work (k-store appends,
            # per-request selection + gather loops): a captured decode would
            # replay only the recorded kernels and silently skip index
            # upkeep, corrupting every later selection. Eager-only until the
            # fp8 paged k-pool + captured-scatter follow-up.
            return []
        if self._moe_capture_blocked(tp_world_size):
            return []
        if self.config.mtp_num_draft_tokens > 0:
            # The MTP step's host phases — greedy verify (a device→host
            # sync), KV rewind, per-request emission bookkeeping, and the
            # draft loop's in-forward re-plans — are not stream-capturable,
            # and the packed prefill capture path never runs preprocess, so
            # full-forward captures with MTP on crash during warmup (the
            # seq_lens assert). Measured 2026-08-09: all 296 captures
            # failed and the whole decode silently served eager, 3.39 vs
            # 43.09 tok/s. The capturable heavy half — the trunk verify
            # forward over (bs, k+1) rows — registers as a piecewise graph
            # instead (get_piecewise_cuda_graph_configs); prefill and the
            # draft iterations stay eager per the config's declared scope.
            logger.info(
                "Glm52LLMSubmodule: MTP k=%d — skipping full-forward CUDA "
                "graphs (host-side verify/rewind is uncapturable); the trunk "
                "verify forward is captured piecewise, prefill + draft "
                "iterations run eager.",
                self.config.mtp_num_draft_tokens,
            )
            return []
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        prefill_packed = {
            num_tokens: self._build_prefill_packed(num_tokens, device)
            for num_tokens in prefill_buckets
        }
        # MSTAR_GLM52_GRAPH_COMPILE=0: capture the eager forward instead of
        # the torch.compile'd one. Escape hatch for the Inductor-subprocess
        # Triton crash (per_token_group_quant_fp8_kernel dies with
        # "PassManager::run failed" in make_llir under the subprocess compile
        # pool, 2026-08-07 — the same kernel compiles fine in-process) that
        # failed all 296 captures and silently degraded the fast config to
        # eager. Pure stream capture still buys the launch-overhead win;
        # Inductor fusion returns when the toolchain bug is resolved.
        # Env-gated per MSTAR_PRE_PLAN_SPEC precedent (worker.py).
        graph_compile = os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1"
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="decode",
                requires_cfg=False,
                labels=[_MAIN],
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                compile=graph_compile,
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill",
                replay_graph_walks=["prefill"],
                packed_seq_len_to_inputs=prefill_packed,
                requires_cfg=False,
                labels=[_MAIN],
                compile=graph_compile,
                causal_attention=True,
                capture_batch_sizes=prefill_batch_sizes,
            ),
        ]

    MTP_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def get_piecewise_cuda_graph_configs(
        self,
        device: torch.device,
        autocast_dtype: torch.dtype,
        tp_world_size: int = 1,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """Three piecewise graphs for the MTP step. ``mtp_trunk``: the trunk
        verify forward — embed + 78 layers + lm_head over the packed
        (bs, k+1) rows. ``mtp_draft`` (k >= 2 only): ONE chain iteration of
        the draft loop — fuse(embed(draft), prev_hidden) through the
        layer-78 module + head argmax, 1 row per request — replayed k-1
        times per step. ``mtp_sync``: the decode sync pass PADDED to k+1
        rows per request (mtp_sync_padded_layout), sharing the trunk's
        capture shape — pads cost ~nothing (the 10 ms eager sync is launch
        overhead, not compute) and buy the fixed shape that makes it
        capturable at all. The step's remaining host phases (greedy verify,
        rewind, the whole-prompt prefill sync) stay eager in
        ``_forward_batched_mtp``. Same MSTAR_GLM52_GRAPH_COMPILE gate as
        the full-forward captures so the kernel stack matches the k=0 fast
        config."""
        k = self.config.mtp_num_draft_tokens
        if (
            k <= 0
            or self.config.dsa_long_context
            or self._moe_capture_blocked(tp_world_size)
        ):
            return {}
        rows = k + 1

        def make_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
                "position_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
            }

        def make_draft_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "draft_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
                "prev_hidden": torch.zeros(
                    shape.total_tokens, self.config.hidden_size,
                    dtype=autocast_dtype, device=device),
                "position_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
            }

        def make_sync_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "sync_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
                "pair_hidden": torch.zeros(
                    shape.total_tokens, self.config.hidden_size,
                    dtype=autocast_dtype, device=device),
                "position_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
            }

        configs: dict[str, PiecewiseCudaGraphConfig] = {
            MTP_TRUNK_LABEL: Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_trunk_captured,
                make_static_inputs=make_static_inputs,
                plan_fn=self._mtp_trunk_plan,
                uses_kv_cache=True,
                cache_labels=[_MAIN],
                capture_batch_sizes=list(self.MTP_CAPTURE_BATCH_SIZES),
                compile=os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
            )
        }
        if k >= 2:
            configs[MTP_DRAFT_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=1,
                capture_fn=self._mtp_draft_captured,
                make_static_inputs=make_draft_static_inputs,
                plan_fn=self._mtp_draft_plan,
                uses_kv_cache=True,
                cache_labels=[_MAIN],
                capture_batch_sizes=list(self.MTP_CAPTURE_BATCH_SIZES),
                compile=os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
            )
        if self._mtp_capture_sync:
            configs[MTP_SYNC_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_sync_captured,
                make_static_inputs=make_sync_static_inputs,
                plan_fn=self._mtp_sync_plan,
                uses_kv_cache=True,
                cache_labels=[_MAIN],
                capture_batch_sizes=list(self.MTP_CAPTURE_BATCH_SIZES),
                compile=os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
            )
        return configs

    def _mtp_trunk_captured(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm: BatchedCacheManager | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Captured region for the MTP decode step (PiecewiseCudaGraphRunner
        contract: read static buffers, return name->tensor). The in-forward
        ``advance_seq_lens`` at the end of the trunk loop is host-only code
        — it runs at capture but not at replay, where the runner's own
        post-replay advance takes over (same +k+1 per request)."""
        static_cm.set_active_label(_MAIN)
        hidden, prenorm = self._hidden(
            static_inputs["input_ids"], static_inputs["position_ids"], static_cm,
            with_prenorm=True)
        return {
            "hidden": hidden,
            "prenorm": prenorm,
            "logits": self.lm_head(hidden),
        }

    def _mtp_trunk_plan(
        self, cache_manager: BatchedCacheManager, shape: PiecewiseCaptureShape,
    ) -> None:
        """Outside-the-graph plan for capture and every replay: attention +
        rope on the trunk label, mirroring ``preprocess`` (which skips its
        own plan when the trunk is about to replay — this one, against the
        runner's persistent wrappers, is the live plan). At replay the
        runner has already aliased the real request states onto the dummy
        slots, so position reads see real counters."""
        cache_manager.set_active_label(_MAIN)
        cache_manager.plan_attention(
            seq_lens=shape.seq_lens, is_causal=True, label=_MAIN)
        cache_manager.plan_rope(
            seq_lens=shape.seq_lens, pos_ids=None, label=_MAIN)

    def _mtp_draft_captured(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm: BatchedCacheManager | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Captured region for one draft-chain iteration: fuse the previous
        draft's embedding with the chained raw hidden, run the layer-78
        module, take the head argmax. Output names deliberately match the
        input names so each replay's outputs feed the next replay's
        copy-in unchanged."""
        static_cm.set_active_label(_MAIN)
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(static_inputs["draft_ids"]),
            static_inputs["prev_hidden"],
            static_cm,
            static_inputs["position_ids"],
        )
        return {
            "draft_ids": self.lm_head(h_head).argmax(dim=-1),
            "prev_hidden": h_raw,
        }

    def _mtp_draft_plan(
        self, cache_manager: BatchedCacheManager, shape: PiecewiseCaptureShape,
    ) -> None:
        """Plan one chain iteration on the MTP plane: 1 row per request,
        RoPE at counter+1 — the plane stores the entry for the token at
        stream position p in slot p-1, so the row written at the current
        counter carries the NEXT stream position (the same shift-by-one as
        the eager loop's ``st + it``). Derivable entirely from the aliased
        request states, for capture (dummy counters) and replay (real
        counters) alike; zero-length padding rows contribute no tokens."""
        cache_manager.set_active_label(_MAIN)
        cache_manager.set_layer_idx(self.config.num_hidden_layers)
        cache_manager.plan_attention(
            seq_lens=shape.seq_lens, is_causal=True, label=_MAIN)
        pos = [
            cache_manager._get_state(rid, _MAIN).position_id_start + 1
            for rid, sl in zip(
                cache_manager.request_ids, shape.seq_lens, strict=True)
            if sl > 0
        ]
        cache_manager.plan_rope(
            seq_lens=shape.seq_lens,
            pos_ids=torch.tensor(
                pos, dtype=torch.long, device=self.get_device()),
            label=_MAIN)

    def _mtp_sync_captured(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm: BatchedCacheManager | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Captured region for the PADDED sync pass: fuse the committed
        tokens' embeddings with their paired trunk rows and run the
        layer-78 module over k+1 rows per request — real rows first, pads
        after (``mtp_sync_padded_layout`` owns the arithmetic; pads write
        transient plane entries the chain overwrites next). Draft 1's head
        gather stays OUTSIDE the graph: which row it reads (the last REAL
        row) is data-dependent, and the eager (bs,)-row gather + head GEMM
        is exactly what the eager sync already paid."""
        static_cm.set_active_label(_MAIN)
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(static_inputs["sync_ids"]),
            static_inputs["pair_hidden"],
            static_cm,
            static_inputs["position_ids"],
        )
        return {"h_head": h_head, "h_raw": h_raw}

    def _mtp_sync_plan(
        self, cache_manager: BatchedCacheManager, shape: PiecewiseCaptureShape,
    ) -> None:
        """Plan the padded sync pass on the plane: k+1 rows per present
        request (zero-length rows for padding slots). The caller has
        already rewound the counter by e, so row j's RoPE position is
        counter+1+j — one contiguous run per request: real rows land on
        their token's true position and pads continue monotonically past
        it. That contiguity is the whole point of the padded layout: the
        plan derives positions from the aliased counters alone, with no
        knowledge of the data-dependent e, so capture (dummy counters) and
        replay (real counters) both plan correctly."""
        cache_manager.set_active_label(_MAIN)
        cache_manager.set_layer_idx(self.config.num_hidden_layers)
        cache_manager.plan_attention(
            seq_lens=shape.seq_lens, is_causal=True, label=_MAIN)
        pos: list[int] = []
        for rid, sl in zip(
            cache_manager.request_ids, shape.seq_lens, strict=True,
        ):
            if sl > 0:
                start = cache_manager._get_state(rid, _MAIN).position_id_start
                pos.extend(range(start + 1, start + 1 + sl))
        cache_manager.plan_rope(
            seq_lens=shape.seq_lens,
            pos_ids=torch.tensor(
                pos, dtype=torch.long, device=self.get_device()),
            label=_MAIN)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        if self.config.mtp_num_draft_tokens > 0:
            rid = fwd_info.request_id
            sampling = fwd_info.sampling_config["LLM"]
            # v1 is greedy-only: decode drafts and verification bypass the
            # engine sampler, so temperature would be silently ignored and a
            # repetition penalty moves even the greedy argmax (prefill would
            # apply it, verify would not — a mixed stream). Refuse per
            # request rather than serve a distribution the client didn't ask
            # for; the engine drops only this rid (kv_cache_engine catches
            # per-request prepare_inputs failures).
            if sampling.temperature != 0 or sampling.repetition_penalty != 1:
                raise RuntimeError(
                    f"request {rid}: MTP speculative decoding is greedy-only "
                    f"(v1) but the request asks for "
                    f"temperature={sampling.temperature}, "
                    f"repetition_penalty={sampling.repetition_penalty}. "
                    "Decode tokens are raw argmax and would silently ignore "
                    "both. Send temperature=0 without a penalty, or serve a "
                    "k=0 config."
                )
            self._mtp_max_tokens[rid] = fwd_info.max_tokens
            self._mtp_ignore_eos[rid] = sampling.ignore_eos
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]

        # M3 trunk replay decision, made ONCE here so plan and forward agree:
        # when the MTP decode trunk will replay its piecewise graph, the
        # graph's own plan (on its persistent wrappers, from the same aliased
        # request states) is the live one — the eager plan below would be
        # dead work on the step-critical path. getattr: older engine-input
        # stubs (tests) predate the piecewise_runners field.
        trunk_runner = None
        draft_runner = None
        sync_runner = None
        if self.config.mtp_num_draft_tokens > 0:
            runners = getattr(engine_inputs, "piecewise_runners", None) or {}
            if graph_walk == "decode":
                candidate = runners.get(MTP_TRUNK_LABEL)
                if candidate is not None and candidate.can_run(
                    len(inputs), sum(seq_lens)
                ):
                    trunk_runner = candidate
                # The padded sync pass is decode-only (prefill's sync spans
                # the whole prompt, outside the k+1-row family): k+1 rows
                # per request regardless of how many were accepted.
                rows = self.config.mtp_num_draft_tokens + 1
                scand = runners.get(MTP_SYNC_LABEL)
                if scand is not None and scand.can_run(
                    len(inputs), rows * len(inputs)
                ):
                    sync_runner = scand
            # The draft chain runs after decode AND prefill (both draft):
            # 1 row per request, so bs is the token count.
            dcand = runners.get(MTP_DRAFT_LABEL)
            if dcand is not None and dcand.can_run(len(inputs), len(inputs)):
                draft_runner = dcand

        cache_manager.set_active_label(_MAIN)
        if trunk_runner is None:
            cache_manager.plan_attention(
                seq_lens=seq_lens, is_causal=True, label=_MAIN)
            cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label=_MAIN)

        device = self.get_device()
        # With dsa_long_context off, dense MLA equals the reference
        # computation ONLY while every attended context fits in the top-k
        # window (index_topk = 2048) — beyond it the ~20 FULL-indexer layers
        # would silently diverge, so refuse rather than serve off-spec
        # logits. With the flag on, the DSA engine path serves beyond topk
        # and the cap moves to the configured serving window.
        long_context = self.config.dsa_long_context
        limit = self.config.max_seq_len if long_context else self.config.index_topk
        topk = self.config.index_topk
        pos_ids_list: list[int] = []
        spans: list[Glm52DsaRequestSpan] = []
        needs_selection = False
        q_start = 0
        for rid, sl in zip(cache_manager.request_ids, seq_lens, strict=True):
            state = cache_manager._get_state(rid, _MAIN)
            start = state.position_id_start
            if start + sl > limit:
                raise RuntimeError(
                    f"request {rid}: context {start + sl} exceeds {limit}, "
                    + (
                        "the configured max_seq_len serving window."
                        if long_context
                        else "the regime where dense MLA is exactly GLM-5.2's "
                        "DSA computation. Long context needs the Phase C "
                        "indexer engine path (dsa_long_context=True)."
                    )
                )
            if long_context:
                if start + sl > topk:
                    if sl > 1:
                        raise RuntimeError(
                            f"request {rid}: prefill context {start + sl} "
                            f"exceeds index_topk={topk}. Sparse attention "
                            "beyond topk is decode-only in v1 — long-prompt "
                            "sparse prefill (per-token windows inside the "
                            "chunk) is the marked follow-up."
                        )
                    needs_selection = True
                # plan_attention already allocated this chunk's pages, so the
                # snapshot covers every position the sparse path touches.
                spans.append(Glm52DsaRequestSpan(
                    request_id=rid, q_start=q_start, q_len=sl,
                    ctx_start=start, page_indices=list(state.page_indices),
                ))
                q_start += sl
            pos_ids_list.extend(range(start, start + sl))
        position_ids = torch.tensor(pos_ids_list, dtype=torch.long, device=device)

        seq_len_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
        return {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
            "position_ids": position_ids,
            # Eager prefill fallback: the mla_absorb FlashInfer wrapper only
            # materializes qo_indptr_buf under CUDA-graph capture, and the
            # reference-dispatch configs register no graphs.
            "last_token_indices": seq_len_t.cumsum(0) - 1,
            # Per-request row counts for the M3 step (verify slicing).
            "seq_lens": list(seq_lens),
            # Non-None only when this decode step's trunk replays its
            # piecewise graph (decision above; forward must not re-decide).
            "mtp_trunk_runner": trunk_runner,
            # Non-None when the draft-chain iterations can replay theirs.
            "mtp_draft_runner": draft_runner,
            # Non-None when the decode sync pass can replay padded.
            "mtp_sync_runner": sync_runner,
            "dsa_ctx": Glm52DsaForwardContext(
                spans=spans, k_store=self._dsa_k_store,
                needs_selection=needs_selection,
            ) if long_context else None,
        }

    def _mtp_pair_rows(
        self, normed: torch.Tensor, prenorm: torch.Tensor
    ) -> torch.Tensor:
        """The trunk stream the MTP plane pairs drafts against — an OPEN
        question, env-switchable so the box can settle it in one A/B.

        ``hnorm`` is a learned RMSNorm over the trunk hidden, so feeding it the
        wrong stream is a systematic distribution shift that depresses every
        draft position. Two candidate conventions:

        - **pre-final-norm** (default today, commit 49315a44) — the raw
          residual stream out of layer 77, before ``model.norm``.
        - **post-final-norm** (``MSTAR_GLM52_MTP_PAIR_POSTNORM=1``) — what
          vLLM feeds its drafter.

        Why this is open rather than settled [2026-08-10]: 49315a44 switched to
        pre-norm citing 0.00 acceptance for post-norm over 3584 request-steps —
        but the very next commit, 474a95e9, identifies the 0.00-acceptance ROOT
        CAUSE as the TP fast read plan never loading the MTP layer, i.e.
        drafting from uninitialized weights (``Glm52ForCausalLM.load_weights``
        now refuses to serve on exactly that condition). The post-norm arm was
        never re-measured against loaded weights, so its only evidence is
        explained by a different bug.

        Meanwhile vLLM, on the identical checkpoint and the same single layer,
        scores p1=0.866 / p2=0.614 against M*'s 0.77 / 0.34, and it pairs
        POST-final-norm: ``GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)``, whose
        ``forward`` returns ``self.norm(...)`` (``deepseek_v2.py:1339``), and it
        does not implement the ``get_mtp_target_hidden_states`` override that
        would swap in a pre-norm residual (DeepSeek-V4 only). Note the CHAIN
        convention already matches vLLM — both thread the raw pre-shared_head
        block output — so the trunk pairing is the sole divergence.
        """
        return normed if self._mtp_pair_postnorm else prenorm

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        with_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.language_model.model(
            input_ids, cache_handle, position_ids, dsa_ctx=dsa_ctx,
            return_prenorm=with_prenorm)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        self._stop_load_heartbeat()
        cache_handle = engine_inputs.cache_manager
        hidden = self._hidden(
            input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"))
        logits = self.lm_head(hidden[-1:])
        return {"logits": [logits]}

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        if self.config.mtp_num_draft_tokens > 0:
            return self._forward_batched_mtp(
                graph_walk, engine_inputs, input_ids, position_ids, **kwargs
            )
        self._stop_load_heartbeat()
        cache_handle = engine_inputs.cache_manager
        sampler = engine_inputs.sampler
        cache_handle.set_active_label(_MAIN)

        hidden = self._hidden(
            input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"))

        if graph_walk == "prefill":
            qo_indptr_buf = cache_handle.get_qo_indptr_buf(_MAIN)
            if qo_indptr_buf is not None:  # CUDA-graph path: static buffer
                last_token_indices = (qo_indptr_buf[1:] - 1).long()
            else:  # eager path: computed by preprocess from real seq_lens
                last_token_indices = kwargs.get("last_token_indices")
                assert last_token_indices is not None, (
                    "eager prefill forward_batched needs last_token_indices "
                    "from preprocess when no CUDA-graph qo_indptr buffer exists"
                )
            hidden = hidden.index_select(0, last_token_indices)
        elif graph_walk != "decode":
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

        logits = self.lm_head(hidden)  # (bs, vocab)
        request_ids = cache_handle.request_ids
        new_tokens = self._sample(sampler, request_ids, logits)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }

    # ------------------------------------------------------------------
    # M3: the MTP draft loop. One step = verify the previous k drafts in a
    # single target forward (k+1 tokens/request), emit accepted + 1, rewind
    # the rejected KV tail, then sync + draft the next k with the layer-78
    # module. Drafts ride the walk's text_inputs loop-back edge together
    # with the last emitted token, so every decode step is uniformly
    # (k+1)-in / (accepted+1)-out and TP followers rebuild identical
    # batches from replicated edges. Greedy-verify keeps the emitted
    # stream bit-identical to non-speculative decoding at temp 0 (v1 is
    # greedy-only; drafts and verification bypass the engine sampler).
    # ------------------------------------------------------------------

    def _forward_batched_mtp(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        from mstar.model.glm52.components.mtp import mtp_greedy_verify

        self._stop_load_heartbeat()
        cache_handle = engine_inputs.cache_manager
        cache_handle.set_active_label(_MAIN)
        seq_lens = kwargs.get("seq_lens")
        assert seq_lens is not None, "MTP step needs seq_lens from preprocess"
        request_ids = list(cache_handle.request_ids)

        row_starts = [0]
        for sl in seq_lens:
            row_starts.append(row_starts[-1] + sl)

        if graph_walk == "prefill":
            # Prefill stays eager (v1 scope): its tail needs the engine
            # sampler plus the same uncapturable sync/draft phases.
            hidden, prenorm = self._hidden(
                input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"),
                with_prenorm=True)
            # Emit the prefill token through the engine sampler, exactly as
            # flag-off does — bit-parity for the first emitted token.
            qo_indptr_buf = cache_handle.get_qo_indptr_buf(_MAIN)
            if qo_indptr_buf is not None:
                last_token_indices = (qo_indptr_buf[1:] - 1).long()
            else:
                last_token_indices = kwargs.get("last_token_indices")
                assert last_token_indices is not None
            last_hidden = hidden.index_select(0, last_token_indices)
            new_tokens = self._sample(
                engine_inputs.sampler, request_ids, self.lm_head(last_hidden))
            # MTP-plane sync over the whole prompt: entry for token t_p
            # pairs (embed(t_p), h_{p-1}); the prompt's first token has no
            # predecessor and is never an entry. The last entry — the
            # prefill-emitted token paired with the prompt-final hidden —
            # yields draft 1.
            sync_tokens, pair_hiddens = [], []
            for i, rid in enumerate(request_ids):
                r = slice(row_starts[i], row_starts[i + 1])
                self._mtp_emitted[rid] = 1
                sync_tokens.append(torch.cat(
                    [input_ids[r][1:], new_tokens[i:i + 1]]))
                pair_hiddens.append(self._mtp_pair_rows(hidden, prenorm)[r])
            drafts = self._mtp_sync_and_draft(
                cache_handle, sync_tokens, pair_hiddens,
                draft_runner=kwargs.get("mtp_draft_runner"))
            return {
                rid: {
                    "new_token": [new_tokens[i:i + 1]],
                    # Under MTP_DRAFT_BUNDLE, never "text_inputs" — see the
                    # constant. Consumed only when the prefill-drafts edge is
                    # declared; otherwise unrouted and dropped, exactly as
                    # before.
                    MTP_DRAFT_BUNDLE: [
                        torch.cat([new_tokens[i:i + 1], drafts[i]])],
                }
                for i, rid in enumerate(request_ids)
            }

        if graph_walk != "decode":
            raise ValueError(
                f"Batched forward not supported for graph walk: {graph_walk!r}")

        trunk_runner = kwargs.get("mtp_trunk_runner")
        if trunk_runner is not None:
            # Replay the captured trunk (embed + layers + lm_head over the
            # packed (bs, k+1) rows). The runner plans on its persistent
            # wrappers from the aliased real states, replays, and advances
            # — the same net bookkeeping as the eager trunk's in-forward
            # advance. Outputs come back cloned and sliced to the real
            # token rows. preprocess already skipped its eager plan for
            # this step (the decision is made once, there).
            replay = trunk_runner.run(
                static_inputs={
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                },
                request_ids=request_ids,
                seq_lens=list(seq_lens),
            )
            hidden = replay["hidden"]
            prenorm = replay["prenorm"]
            logits = replay["logits"]
        else:
            self._warn_mtp_trunk_eager_once(len(request_ids), sum(seq_lens))
            hidden, prenorm = self._hidden(
                input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"),
                with_prenorm=True)
            logits = self.lm_head(hidden)  # (sum(k+1), vocab)
        results: dict[str, NameToTensorList] = {}
        rewinds: list[int] = []
        sync_tokens, pair_hiddens = [], []
        for i, rid in enumerate(request_ids):
            r = slice(row_starts[i], row_starts[i + 1])
            m = seq_lens[i]
            r_inputs = input_ids[r]
            drafts_in = r_inputs[1:]
            target_argmax = logits[r].argmax(dim=-1)
            n_acc, bonus = mtp_greedy_verify(drafts_in, target_argmax)
            # Raw (pre-truncation) emission is the draft-quality signal.
            self._mtp_stat_steps += 1
            self._mtp_stat_emitted += n_acc + 1
            self._mtp_stat_acc_hist[n_acc] += 1
            emitted = torch.cat([drafts_in[:n_acc], bonus.reshape(1)])
            # Truncate: max_tokens budget first, then first stop id. EOS is
            # always the LAST element after truncation, which is the
            # contract check_stop relies on.
            budget = self._mtp_max_tokens[rid] - self._mtp_emitted[rid]
            e = min(emitted.shape[0], max(budget, 1))
            if not self._mtp_ignore_eos[rid]:
                for j in range(e):
                    if int(emitted[j]) in self.config.eos_token_ids:
                        e = j + 1
                        break
            emitted = emitted[:e]
            self._mtp_emitted[rid] += e
            # m tokens appended KV this forward; the committed prefix is
            # input[0] plus the e-1 now-emitted drafts. The bonus was never
            # processed.
            rewinds.append(m - e)
            sync_tokens.append(emitted)
            pair_hiddens.append(self._mtp_pair_rows(hidden, prenorm)[r][:e])
            results[rid] = {"new_token": [emitted]}
        self._maybe_log_mtp_acceptance()
        cache_handle.rewind_seq_lens(rewinds)
        sync_runner = kwargs.get("mtp_sync_runner")
        if sync_runner is None and self._mtp_capture_sync:
            self._warn_mtp_sync_eager_once(len(request_ids))
        drafts = self._mtp_sync_and_draft(
            cache_handle, sync_tokens, pair_hiddens,
            draft_runner=kwargs.get("mtp_draft_runner"),
            sync_runner=sync_runner)
        for i, rid in enumerate(request_ids):
            emitted = results[rid]["new_token"][0]
            results[rid]["text_inputs"] = [
                torch.cat([emitted[-1:], drafts[i]])]
        return results

    def _mtp_sync_and_draft(
        self,
        cache_handle: BatchedCacheManager,
        sync_tokens: list[torch.Tensor],
        pair_hiddens: list[torch.Tensor],
        draft_runner=None,
        sync_runner=None,
    ) -> list[torch.Tensor]:
        """Extend the MTP plane over the newly committed tokens, then draft
        k tokens autoregressively. Returns per-request (k,) draft tensors.

        The MTP plane (layer index ``num_hidden_layers``) shares the trunk's
        page table and position counter, SHIFTED BY ONE: the entry for the
        token at stream position p — ``fuse(embed(t_p), h_{p-1})`` — lives
        at plane position p-1, so after the sync pass the plane holds
        exactly ``position_id_start`` entries, aligned with the trunk. RoPE
        uses the token's true position (the explicit position_ids arg).
        Draft-iteration entries beyond the counter are transient: the
        counter is rolled back at the end, and the next step's writes
        overwrite them in place (paged layout, no data movement).
        """
        k = self.config.mtp_num_draft_tokens
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        request_ids = list(cache_handle.request_ids)
        num = len(request_ids)
        mtp_layer = self.config.num_hidden_layers
        device = pair_hiddens[0].device
        e_list = [t.shape[0] for t in sync_tokens]
        starts = [
            cache_handle._get_state(rid, _MAIN).position_id_start
            for rid in request_ids
        ]

        # Sync pass (+ draft 1 from its last row): plane positions
        # start-e .. start-1, token positions start-e+1 .. start.
        cache_handle.rewind_seq_lens(e_list)
        if sync_runner is not None:
            # PADDED replay: k+1 rows per request, the trunk's own capture
            # shape. Real rows first, pads after; pads write transient
            # plane entries at slots >= start that the chain (and the next
            # step's writes) overwrite before any read. Decode-only by
            # construction — e = emitted <= k+1 — and the guard is loud
            # because a violated bound here would otherwise surface as
            # nothing but lower acceptance.
            rows = k + 1
            assert all(e <= rows for e in e_list), (
                f"padded sync got rows {e_list} outside [1, {rows}] — the "
                "padded family is decode-only")
            pos_l, last_l, over_advance = mtp_sync_padded_layout(
                e_list, starts, k)
            sync_ids = torch.zeros(
                num * rows, dtype=torch.long, device=device)
            pair_h = torch.zeros(
                (num * rows, pair_hiddens[0].shape[-1]),
                dtype=pair_hiddens[0].dtype, device=device)
            for i, (t, h) in enumerate(
                zip(sync_tokens, pair_hiddens, strict=True)
            ):
                sync_ids[i * rows:i * rows + t.shape[0]] = t
                pair_h[i * rows:i * rows + h.shape[0]] = h
            out = sync_runner.run(
                static_inputs={
                    "sync_ids": sync_ids,
                    "pair_hidden": pair_h,
                    "position_ids": torch.tensor(
                        pos_l, dtype=torch.long, device=device),
                },
                request_ids=request_ids,
                seq_lens=[rows] * num,
            )
            # The runner advanced `rows` per request; only e were real.
            cache_handle.rewind_seq_lens(over_advance)
            h_head, h_raw = out["h_head"], out["h_raw"]
            last_rows = torch.tensor(
                last_l, dtype=torch.long, device=device)
        else:
            cache_handle.set_layer_idx(mtp_layer)
            cache_handle.plan_attention(
                seq_lens=e_list, is_causal=True, label=_MAIN)
            pos_list: list[int] = []
            for st, e in zip(starts, e_list, strict=True):
                pos_list.extend(range(st - e + 1, st + 1))
            positions = torch.tensor(
                pos_list, dtype=torch.long, device=device)
            cache_handle.plan_rope(
                seq_lens=e_list, pos_ids=positions, label=_MAIN)
            h_head, h_raw = mtp(
                embed(torch.cat(sync_tokens)), torch.cat(pair_hiddens),
                cache_handle, positions,
            )
            cache_handle.advance_seq_lens()
            # Packed (unpadded) layout: last real row via cumsum.
            last_rows = torch.tensor(
                e_list, dtype=torch.long, device=device).cumsum(0) - 1

        # Head reads the shared_head-normed rows; the CHAIN threads the raw
        # layer output (hnorm re-norms it next iteration — same convention
        # as the trunk pairing).
        prev_h = h_raw.index_select(0, last_rows)      # (B, hid) chain
        prev_d = self.lm_head(
            h_head.index_select(0, last_rows)).argmax(dim=-1)  # (B,) draft 1
        draft_cols = [prev_d]
        ones = [1] * num
        if k > 1 and draft_runner is None:
            self._warn_mtp_draft_eager_once(num)
        for it in range(1, k):
            positions = torch.tensor(
                [st + it for st in starts], dtype=torch.long, device=device)
            if draft_runner is not None:
                # Replay the captured chain iteration. The runner plans on
                # its persistent wrappers from the aliased states (rope at
                # counter+1 == st+it, matching ``positions``), replays, and
                # advances +1 per request — the same bookkeeping as the
                # eager body, so the final k-1 rewind below is unchanged.
                out = draft_runner.run(
                    static_inputs={
                        "draft_ids": prev_d,
                        "prev_hidden": prev_h,
                        "position_ids": positions,
                    },
                    request_ids=request_ids,
                    seq_lens=ones,
                )
                prev_d = out["draft_ids"]
                prev_h = out["prev_hidden"]
            else:
                cache_handle.set_layer_idx(mtp_layer)
                cache_handle.plan_attention(
                    seq_lens=ones, is_causal=True, label=_MAIN)
                cache_handle.plan_rope(
                    seq_lens=ones, pos_ids=positions, label=_MAIN)
                it_head, prev_h = mtp(
                    embed(prev_d), prev_h, cache_handle, positions)
                cache_handle.advance_seq_lens()
                prev_d = self.lm_head(it_head).argmax(dim=-1)
            draft_cols.append(prev_d)
        if k > 1:
            cache_handle.rewind_seq_lens([k - 1] * num)
        stacked = torch.stack(draft_cols, dim=1)       # (B, k)
        return [stacked[i] for i in range(num)]

    def _warn_mtp_trunk_eager_once(self, bs: int, num_rows: int) -> None:
        """The 2026-08-09 lesson: an MTP decode whose trunk silently runs
        eager is a 13x regression that looks like "MTP is slow". Say it
        once, loudly, with the likely causes."""
        if self._mtp_trunk_eager_warned:
            return
        self._mtp_trunk_eager_warned = True
        logger.warning(
            "MTP decode trunk running EAGER (bs=%d, %d rows): no piecewise "
            "CUDA graph bucket available — capture failed at warmup, the "
            "MoE resolved to an uncapturable dispatch path, or bs exceeds "
            "the captured sizes %s. Expect roughly reference-pace decode.",
            bs, num_rows, self.MTP_CAPTURE_BATCH_SIZES,
        )

    def _warn_mtp_draft_eager_once(self, bs: int) -> None:
        """Same silence-is-a-regression lesson for the chain: each eager
        iteration costs ~7 ms of launch overhead vs ~1-2 replayed, and at
        k>=3 that dominates the MTP step."""
        if self._mtp_draft_eager_warned:
            return
        self._mtp_draft_eager_warned = True
        logger.warning(
            "MTP draft chain running EAGER (bs=%d): no mtp_draft piecewise "
            "bucket available — capture failed at warmup or bs exceeds the "
            "captured sizes %s. Each chain iteration pays eager launch "
            "overhead.",
            bs, self.MTP_CAPTURE_BATCH_SIZES,
        )

    def _warn_mtp_sync_eager_once(self, bs: int) -> None:
        """Same silence-is-a-regression lesson for the sync pass: eager it
        is ~10 ms of a ~37 ms step — launch overhead, not compute."""
        if self._mtp_sync_eager_warned:
            return
        self._mtp_sync_eager_warned = True
        logger.warning(
            "MTP decode sync pass running EAGER (bs=%d): no mtp_sync "
            "piecewise bucket available — capture failed at warmup or bs "
            "exceeds the captured sizes %s. Expect ~10 ms of avoidable "
            "launch overhead per decode step.",
            bs, self.MTP_CAPTURE_BATCH_SIZES,
        )

    _MTP_STAT_LOG_EVERY = 512  # request-steps between acceptance log lines

    def _maybe_log_mtp_acceptance(self) -> None:
        if self._mtp_stat_steps - self._mtp_stat_logged < self._MTP_STAT_LOG_EVERY:
            return
        self._mtp_stat_logged = self._mtp_stat_steps
        k = self.config.mtp_num_draft_tokens
        mean_emitted = self._mtp_stat_emitted / self._mtp_stat_steps
        logger.info(
            "MTP acceptance: %.2f emitted/step (ceiling %d, plain decode "
            "would be 1.00) — draft acceptance rate %.2f over %d "
            "request-steps.",
            mean_emitted, k + 1, (mean_emitted - 1.0) / k if k else 0.0,
            self._mtp_stat_steps,
        )
        # Conditional per-position profile: p_i = P(draft i accepted | drafts
        # 1..i-1 accepted). Greedy verify accepts prefixes, so "reached
        # position i" = n_acc >= i. A flat profile means draft quality is
        # uniform (domain-limited); a falling one means the chained
        # iterations degrade (loop bug or compounding drift).
        reached = [
            sum(self._mtp_stat_acc_hist[i:]) for i in range(k + 1)
        ]  # reached[0] = all steps
        cond = [
            f"{reached[i] / reached[i - 1]:.2f}" if reached[i - 1] else "-"
            for i in range(1, k + 1)
        ]
        # The pairing convention rides along with the numbers on purpose: this
        # profile is the A/B's readout, and a profile whose arm you have to
        # infer from launch env is a profile you cannot trust six hours later.
        logger.info(
            "MTP acceptance by position: n_acc histogram %s, conditional "
            "accept per position %s [trunk pairing: %s]",
            self._mtp_stat_acc_hist, " ".join(cond),
            "POST-final-norm (vLLM convention)" if self._mtp_pair_postnorm
            else "pre-final-norm (default)",
        )

    @staticmethod
    def _sample(
        sampler: Sampler, request_ids: list[str], logits: torch.Tensor,
    ) -> torch.Tensor:
        return sampler.sample(request_ids, logits, apply_penalty=True)

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        if "text_inputs" in outputs:
            # M3 step: the forward already assembled the loop-back input
            # ([last emitted, k drafts]); new_token carries only the
            # verified emission.
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        if self.config.mtp_num_draft_tokens > 0:
            # Multi-token emission: in-step truncation guarantees a stop id
            # can only be the LAST element; totals live in the per-request
            # counter (loop iters no longer count tokens).
            tokens = outputs["new_token"][0]
            last = int(tokens[-1])
            is_eos = last in self.config.eos_token_ids
            ignore_eos = request_info.sampling_config["LLM"].ignore_eos
            generated = self._mtp_emitted.get(request_id, 0)
            if (not ignore_eos and is_eos) or generated >= request_info.max_tokens:
                return {"decode_loop"}
            return set()
        token = outputs["new_token"][0].item()
        # GLM-5.2 defines three stop ids: <|endoftext|>, <|user|>, <|observation|>.
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.sampling_config["LLM"].ignore_eos
        # Total generated = 1 prefill-emitted token + (iters + 1) decode
        # tokens. Counting only decode iters against max_tokens made every
        # length-capped completion one token long — measured in the M1 diff
        # (m = v + 1 on all 20 prompts vs vLLM's max_tokens semantics).
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
