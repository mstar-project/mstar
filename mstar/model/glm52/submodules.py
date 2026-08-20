"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
import os
import time
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
    PiecewisePackedConfig,
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
from mstar.utils.pinned_staging import pinned, to_device_async
from mstar.utils.sampling import Sampler

logger = logging.getLogger(__name__)

_MAIN = "main"
# Piecewise-graph label for the MTP decode step's trunk verify forward.
MTP_TRUNK_LABEL = "mtp_trunk"
MTP_DRAFT_LABEL = "mtp_draft"
MTP_SYNC_LABEL = "mtp_sync"
MTP_PREFILL_LABEL = "mtp_prefill"
MTP_DRAFT_PHASE_LABEL = "mtp_draft_phase"
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


class _MtpStepTimer:
    """nsys-lite for one MTP decode step, gated by MSTAR_GLM52_MTP_STEP_TIMING=N.

    On every N-th step, records a CUDA event + host timestamp at each phase
    boundary (trunk replay, verify, sync replay, each chain replay, tail) and,
    at the next sampled step, reads them back: the GPU column is the device
    timeline between consecutive events (includes any idle the GPU spent
    waiting for the host to enqueue), the host column is the wall the host
    spent enqueueing that phase. GPU >> host means the phase is GPU-bound;
    GPU ≈ host means the host is the bottleneck. Costs one event sync per
    sampled step; zero cost when off.
    """

    def __init__(self, every: int):
        self.every = every
        self.step = 0
        self._marks: list[tuple[str, torch.cuda.Event, float]] = []
        self._pending: list[tuple[str, torch.cuda.Event, float]] | None = None
        self.active = False

    def begin(self) -> None:
        self.step += 1
        self.active = self.every > 0 and self.step % self.every == 0
        if self.active:
            self._marks = []
            self.mark("start")

    def mark(self, name: str) -> None:
        if not self.active:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._marks.append((name, ev, time.perf_counter()))

    def end(self) -> None:
        if not self.active:
            return
        self.mark("end")
        self._pending, self._marks = self._marks, []
        self.active = False

    def report(self, log) -> None:
        """Called at the START of a step (before new work): the previous
        sample's events are long complete, so reading them is free."""
        if not self._pending:
            return
        marks, self._pending = self._pending, None
        if not marks[-1][1].query():
            marks[-1][1].synchronize()
        parts = []
        gpu_total = marks[0][1].elapsed_time(marks[-1][1])
        host_total = (marks[-1][2] - marks[0][2]) * 1e3
        for (_n0, e0, h0), (n1, e1, h1) in zip(marks, marks[1:], strict=False):
            parts.append(f"{n1} {e0.elapsed_time(e1):.2f}|{(h1 - h0) * 1e3:.2f}")
        log.info(
            "MTP step timing (GPU ms | host ms per phase, step %d): %s ; total %.2f|%.2f",
            self.step, ", ".join(parts), gpu_total, host_total,
        )


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
        # ``_mtp_pair_rows``. DEFAULT ON [2026-08-11]: post-final-norm is vLLM's
        # convention; it recovers p1/p2 to 0.89/0.74 (from pre-norm's 0.77/0.33)
        # and, combined with capture below, measures 66.97 tok/s at k=3, 3264
        # bit-exact. Set MSTAR_GLM52_MTP_PAIR_POSTNORM=0 to restore pre-norm
        # pairing (whose only known cost was an eager-sync FP near-tie forking
        # ~0.25% of the stream; the captured path holds bit-identity).
        self._mtp_pair_postnorm = (
            os.environ.get("MSTAR_GLM52_MTP_PAIR_POSTNORM", "1") == "1"
        )
        # Capture the decode sync pass as a padded (bs, k+1) piecewise graph.
        # DEFAULT ON [2026-08-11]: measured clean at 53.58 tok/s, 3264 bit-exact,
        # 0 eager (arm C, TP8), and 66.97 combined with post-norm at k=3. The
        # 08-10 33.02/0.18 scare was the text_inputs bug measured jointly, not
        # this — the padded replay is bit-identical, pinned by
        # test_sync_capture_matches_eager_bit_identically and confirmed at TP8.
        # Set MSTAR_GLM52_MTP_CAPTURE_SYNC=0 to fall back to the eager sync pass
        # — but with post-norm also default-on, that alone lands on post-norm's
        # eager-sync ~0.25% fork; recovering strict bit-identity to plain decode
        # needs MSTAR_GLM52_MTP_PAIR_POSTNORM=0 as well.
        # bs>1 is covered by the reduced-dims GPU test only, not yet at scale.
        self._mtp_capture_sync = (
            os.environ.get("MSTAR_GLM52_MTP_CAPTURE_SYNC", "1") == "1"
        )
        # Capture the MTP PREFILL trunk (embed + 78 layers over the packed
        # prompt) as a piecewise graph over the same token buckets the k=0
        # config captures. With MTP on, get_cuda_graph_configs returns no
        # full-forward graphs — the whole prefill ran eager, and that is the
        # +248 ms TTFT (305 vs k=0's 57 ms) [measured 08-09/08-18]. The
        # sample, the whole-prompt plane sync and the draft chain stay
        # outside the graph, exactly as the decode step keeps verify outside
        # the trunk graph. Set MSTAR_GLM52_MTP_CAPTURE_PREFILL=0 for the eager
        # prefill (escape hatch; per-bucket capture failures also fall back to
        # eager on their own).
        self._mtp_capture_prefill = (
            os.environ.get("MSTAR_GLM52_MTP_CAPTURE_PREFILL", "1") == "1"
        )
        # The whole decode DRAFT PHASE as ONE graph: padded sync pass, draft-1
        # head, and the k-1 chain iterations, with k FlashInfer plan slots
        # planned before a single replay. Arm G's step timer (08-19) showed the
        # three-replay version host-bound at ~5 ms/step (sync 1.6, chain 2.1 +
        # 1.3 ms of host per piecewise run()); one run() is ~1.2 ms. Requires
        # sync capture. MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH=0 restores the
        # three-graph path (kept as the fallback for missing buckets anyway).
        self._mtp_draft_phase_graph = (
            os.environ.get("MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH", "1") == "1"
        )
        # Hoist the e-independent half of the draft phase above the verify
        # readback: full-row sync inputs (rows < e are exactly the emitted
        # tokens; rows >= e are rejected continuations landing on the same
        # transient pad slots the zeros used), the contiguous positions
        # range(P0+1, P0+k+2) (identical to mtp_sync_padded_layout's — its two
        # ranges concatenate), and the e-independent slot-0 FlashInfer plan.
        # The host does ~0.6-0.9 ms of that work during the ~20 ms it spends
        # blocked in the verify .tolist() while the GPU runs the trunk,
        # instead of stalling the GPU for it afterwards. Bit-exact by
        # construction; DEFAULT OFF until a box arm measures it (lane rule).
        self._mtp_phase_prepare = (
            os.environ.get("MSTAR_GLM52_MTP_PHASE_PREPARE", "0") == "1"
        )
        # GLM-5.2 applies RoPE explicitly (Glm52RotaryEmbedding(position_ids),
        # components/attention.py:185/250) and never calls
        # ``cache_handle.apply_rope`` — the only consumer of what plan_rope
        # stages (cache_manager.py:549 asserts ps.pos_ids inside apply_rope).
        # Every GLM plan_rope call is therefore dead work: a pinned host
        # build + async H2D per plan (3-4 per MTP step, 2 of them on the
        # post-verify critical path). MSTAR_GLM52_PLAN_ROPE=0 skips them;
        # default 1 keeps today's behavior until a box arm confirms
        # bit-identity (the GPU rung checks it first).
        self._glm_plan_rope = (
            os.environ.get("MSTAR_GLM52_PLAN_ROPE", "1") == "1"
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
        # nsys-lite (see _MtpStepTimer): MSTAR_GLM52_MTP_STEP_TIMING=200 logs
        # the per-phase GPU|host split of every 200th decode step.
        self._mtp_timer = _MtpStepTimer(
            int(os.environ.get("MSTAR_GLM52_MTP_STEP_TIMING", "0") or "0"))

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
        if self._mtp_capture_sync and self._mtp_draft_phase_graph and k >= 1:
            def make_phase_static_inputs(
                shape: PiecewiseCaptureShape,
            ) -> dict[str, torch.Tensor]:
                bs = shape.bs
                return {
                    "sync_ids": torch.zeros(
                        shape.total_tokens, dtype=torch.long, device=device),
                    "pair_hidden": torch.zeros(
                        shape.total_tokens, self.config.hidden_size,
                        dtype=autocast_dtype, device=device),
                    "sync_position_ids": torch.zeros(
                        shape.total_tokens, dtype=torch.long, device=device),
                    "last_rows": torch.zeros(bs, dtype=torch.long, device=device),
                    # One (bs,) input per chain iteration — NOT one flat
                    # (k-1)*bs buffer: the runner pads the real batch to the
                    # bucket's bs by copying the caller's tensor into buf[:n]
                    # and zeroing the tail, so per-request rows must be
                    # separate inputs to stay aligned when num < bs.
                    **{
                        f"chain_pos_{it}": torch.zeros(bs, dtype=torch.long, device=device)
                        for it in range(1, k)
                    },
                }

            configs[MTP_DRAFT_PHASE_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_draft_phase_captured,
                make_static_inputs=make_phase_static_inputs,
                plan_fn=self._mtp_draft_phase_plan,
                uses_kv_cache=True,
                cache_labels=[_MAIN],
                capture_batch_sizes=list(self.MTP_CAPTURE_BATCH_SIZES),
                compile=os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
                # The plan_fn does the counter bookkeeping (it must position the
                # aliased states per plan slot); the caller rewinds k-1 after.
                advance_seq_lens=False,
                plans_per_label=k,
            )
        if self._mtp_capture_prefill:
            # Same token buckets and batch sizes as the k=0 packed prefill
            # graphs (get_cuda_graph_configs), so MTP-on TTFT lands where
            # MTP-off's does. Static outputs are the full-row hidden and
            # prenorm streams: the whole-prompt plane sync needs every row.
            configs[MTP_PREFILL_LABEL] = PiecewisePackedConfig(
                total_tokens=list(
                    self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS),
                capture_fn=self._mtp_prefill_captured,
                make_static_inputs=make_static_inputs,
                plan_fn=self._mtp_prefill_plan,
                uses_kv_cache=True,
                cache_labels=[_MAIN],
                capture_batch_sizes=list(
                    self.config.prefill_capture_batch_sizes
                    or self.PREFILL_CAPTURE_BATCH_SIZES),
                compile=os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
                # One 512 MiB FlashInfer workspace for all 30 prefill buckets,
                # not 30: arm L (08-19) measured +17 GB/rank from the per-bucket
                # default, leaving ~10 GB headroom on an H200. One prefill
                # replays per step, so sharing is safe.
                share_workspace_across_buckets=True,
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
        self._plan_rope(cache_manager, 
            seq_lens=shape.seq_lens, pos_ids=None, label=_MAIN)

    def _mtp_draft_phase_captured(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm: BatchedCacheManager | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Captured region for the whole decode draft phase (one graph):
        padded sync pass on plan slot 0 → gather each request's last real
        row → draft 1 = head argmax → for it in 1..k-1: fuse(embed(draft),
        raw hidden) through the plane on plan slot it → head argmax. All
        data-dependent ints (which row is real, the chain's kv lengths and
        positions) arrive as static inputs / plan slots, planned before the
        replay by ``_mtp_draft_phase_plan``. ``select_plan_slot`` is host-only:
        at replay the graph simply reads each slot's static buffers."""
        k = self.config.mtp_num_draft_tokens
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        static_cm.set_active_label(_MAIN)
        static_cm.set_layer_idx(self.config.num_hidden_layers)
        static_cm.select_plan_slot(_MAIN, 0)
        h_head, h_raw = mtp(
            embed(static_inputs["sync_ids"]),
            static_inputs["pair_hidden"],
            static_cm,
            static_inputs["sync_position_ids"],
        )
        last = static_inputs["last_rows"]
        prev_h = h_raw.index_select(0, last)
        prev_d = self.lm_head(h_head.index_select(0, last)).argmax(dim=-1)
        cols = [prev_d]
        for it in range(1, k):
            static_cm.select_plan_slot(_MAIN, it)
            pos = static_inputs[f"chain_pos_{it}"]
            it_head, prev_h = mtp(embed(prev_d), prev_h, static_cm, pos)
            prev_d = self.lm_head(it_head).argmax(dim=-1)
            cols.append(prev_d)
        static_cm.select_plan_slot(_MAIN, 0)
        return {"drafts": torch.stack(cols, dim=1)}  # (bs, k)

    def _plan_rope(self, cache_manager, **kw) -> None:
        """GLM's rope plans are dead unless something regrows an
        ``apply_rope`` consumer — see the MSTAR_GLM52_PLAN_ROPE note in
        ``__init__``. Default on (unchanged); flip off in a measured arm."""
        if getattr(self, "_glm_plan_rope", True):
            cache_manager.plan_rope(**kw)

    def _plan_sync_slot(
        self,
        cache_manager: BatchedCacheManager,
        shape: PiecewiseCaptureShape,
    ) -> None:
        """Slot 0 of the draft-phase graph: the padded sync pass, k+1 rows per
        present request at the current counter P, positions P+1..P+k+1. Reads
        nothing data-dependent — callable before the verify readback."""
        cache_manager.select_plan_slot(_MAIN, 0)
        cache_manager.plan_attention(
            seq_lens=shape.seq_lens, is_causal=True, label=_MAIN)
        pos: list[int] = []
        for rid, sl in zip(cache_manager.request_ids, shape.seq_lens, strict=True):
            if sl > 0:
                start = cache_manager._get_state(rid, _MAIN).position_id_start
                pos.extend(range(start + 1, start + 1 + sl))
        self._plan_rope(cache_manager, seq_lens=shape.seq_lens, pos_ids=pos, label=_MAIN)

    def _plan_chain_slots(
        self, cache_manager: BatchedCacheManager, present: list[bool],
    ) -> None:
        """Slots 1..k-1: one row per present request, planned at the current
        counter (P+e at entry), advancing one per iteration. Ends with slot 0
        selected so the next trunk plan lands where it always has."""
        k = self.config.mtp_num_draft_tokens
        ones = [1 if p else 0 for p in present]
        for it in range(1, k):
            cache_manager.select_plan_slot(_MAIN, it)
            cache_manager.plan_attention(seq_lens=ones, is_causal=True, label=_MAIN)
            pos_it = [
                cache_manager._get_state(rid, _MAIN).position_id_start + 1
                for rid, sl in zip(cache_manager.request_ids, ones, strict=True)
                if sl > 0
            ]
            self._plan_rope(cache_manager, seq_lens=ones, pos_ids=pos_it, label=_MAIN)
            cache_manager.advance_seq_lens()
        cache_manager.select_plan_slot(_MAIN, 0)

    def _mtp_draft_phase_plan(
        self,
        cache_manager: BatchedCacheManager,
        shape: PiecewiseCaptureShape,
        e_list: list[int] | None = None,
        phase: str | None = None,
    ) -> None:
        """Plan the draft-phase graph's slots before its single replay.

        ``phase=None`` (capture, warmup, and the un-hoisted replay): all k
        slots in one call. Slot 0 is the padded sync pass exactly as
        ``_mtp_sync_plan`` (k+1 rows per present request, counter at P: the
        caller has rewound e). Then the aliased states are advanced by e (the
        sync consumed e REAL rows) and slot ``it`` (1..k-1) plans one row per
        request at kv length P+e+it-1 with RoPE at position start+1 — exactly
        what ``_mtp_draft_plan`` plans for the it-th replay today. The states
        end at P+e+(k-1); the caller rewinds k-1, mirroring the three-graph
        path. ``e_list`` comes from ``run(plan_ctx=...)``; capture/warmup pass
        none and plan e=1.

        ``phase="prepare"`` (from ``prepare()``, before the verify readback):
        slot 0 only. Counters sit at P+rows (the trunk's post-replay advance);
        rewind rows, plan, and restore them EXACTLY — by snapshot, so a
        mid-plan AllocationFailedError cannot strand the live aliased
        counters (this lane's audit risk #1). e is neither known nor needed.

        ``phase="finish"`` (the paired ``run()``): slots 1..k-1 only, from the
        post-verify counter P+e — the caller must NOT have rewound e_list.
        Ends at P+e+(k-1) like the full plan; the caller rewinds k-1."""
        k = self.config.mtp_num_draft_tokens
        rows = k + 1
        cache_manager.set_active_label(_MAIN)
        cache_manager.set_layer_idx(self.config.num_hidden_layers)
        present = [sl > 0 for sl in shape.seq_lens]

        if phase == "prepare":
            snap = [
                cache_manager._get_state(rid, _MAIN)
                for rid, p in zip(cache_manager.request_ids, present, strict=True)
                if p
            ]
            saved = [(st.seq_len, st.position_id_start) for st in snap]
            cache_manager.rewind_seq_lens([rows if p else 0 for p in present])
            try:
                # leaves slot 0 selected, exactly as the one-shot plan does
                self._plan_sync_slot(cache_manager, shape)
            finally:
                for st, (sl_, ps_) in zip(snap, saved, strict=True):
                    st.seq_len, st.position_id_start = sl_, ps_
            return

        if phase == "finish":
            self._plan_chain_slots(cache_manager, present)
            return

        if e_list is None:
            e_list = [1] * sum(present)
        e_iter = iter(e_list)
        e_by_slot = [next(e_iter) if p else 0 for p in present]
        # Slot 0: padded sync (k+1 rows), positions start+1 .. start+k+1.
        self._plan_sync_slot(cache_manager, shape)
        # The sync consumed e real rows: advance by rows, rewind rows-e.
        cache_manager.advance_seq_lens()
        cache_manager.rewind_seq_lens(
            [rows - e if p else 0 for p, e in zip(present, e_by_slot, strict=True)])
        self._plan_chain_slots(cache_manager, present)

    def _mtp_prefill_captured(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm: BatchedCacheManager | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Captured region for the MTP prefill trunk: embed + layers over the
        packed prompt rows, returning both streams for every row (the last
        rows feed the sample, all rows feed the plane sync). No lm_head
        inside — prefill needs logits for the last row per request only,
        and a (tokens, vocab) static output would be 300 MB per bucket."""
        static_cm.set_active_label(_MAIN)
        hidden, prenorm = self._hidden(
            static_inputs["input_ids"], static_inputs["position_ids"], static_cm,
            with_prenorm=True)
        return {"hidden": hidden, "prenorm": prenorm}

    def _mtp_prefill_plan(
        self, cache_manager: BatchedCacheManager, shape: PiecewiseCaptureShape,
    ) -> None:
        """Plan the packed prefill on the trunk label — the runner hands the
        real per-request lengths (plus zero-length padding rows) at replay,
        mirroring ``preprocess``, which skips its own plan for this step."""
        cache_manager.set_active_label(_MAIN)
        cache_manager.plan_attention(
            seq_lens=shape.seq_lens, is_causal=True, label=_MAIN)
        self._plan_rope(cache_manager, 
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
        # A host list: plan_rope stages it through pinned memory and copies
        # async. ``torch.tensor(pos, device=cuda)`` here was a pageable H2D
        # that drained the stream on every chain iteration.
        self._plan_rope(cache_manager, seq_lens=shape.seq_lens, pos_ids=pos, label=_MAIN)

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
        self._plan_rope(cache_manager, seq_lens=shape.seq_lens, pos_ids=pos, label=_MAIN)

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
        prefill_runner = None
        phase_runner = None
        if self.config.mtp_num_draft_tokens > 0:
            runners = getattr(engine_inputs, "piecewise_runners", None) or {}
            if graph_walk == "prefill":
                # The captured prefill trunk (PACKED: smallest token bucket
                # >= the real total, real per-request lengths at plan time).
                pcand = runners.get(MTP_PREFILL_LABEL)
                if pcand is not None and pcand.can_run(
                    len(inputs), sum(seq_lens)
                ):
                    prefill_runner = pcand
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
                pcand2 = runners.get(MTP_DRAFT_PHASE_LABEL)
                if pcand2 is not None and pcand2.can_run(
                    len(inputs), rows * len(inputs)
                ):
                    phase_runner = pcand2
            # The draft chain runs after decode AND prefill (both draft):
            # 1 row per request, so bs is the token count.
            dcand = runners.get(MTP_DRAFT_LABEL)
            if dcand is not None and dcand.can_run(len(inputs), len(inputs)):
                draft_runner = dcand

        cache_manager.set_active_label(_MAIN)
        if trunk_runner is None and prefill_runner is None:
            cache_manager.plan_attention(
                seq_lens=seq_lens, is_causal=True, label=_MAIN)
            self._plan_rope(cache_manager, seq_lens=seq_lens, pos_ids=None, label=_MAIN)

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
        # Async H2D through pinned staging: a pageable ``torch.tensor(...,
        # device=cuda)`` here drains the stream before the step even starts,
        # which defeats any plan/schedule overlap with the previous step.
        position_ids = to_device_async(pos_ids_list, torch.long, device)

        seq_len_t = to_device_async(seq_lens, torch.long, device)
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
            # Non-None when this prefill's trunk replays its packed graph.
            "mtp_prefill_runner": prefill_runner,
            # Non-None when the whole decode draft phase replays as ONE graph
            # (takes precedence over sync+draft runners in the decode step).
            "mtp_draft_phase_runner": phase_runner,
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
        from mstar.model.glm52.components.mtp import mtp_greedy_verify_host

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
            prefill_runner = kwargs.get("mtp_prefill_runner")
            if prefill_runner is not None:
                # Replay the captured prefill trunk over the packed prompt
                # rows (embed + layers; the tail — sample, whole-prompt plane
                # sync, draft chain — stays eager below, exactly as the
                # decode step keeps verify outside its trunk graph). The
                # runner plans the real lengths on its persistent wrappers,
                # replays, and advances; preprocess skipped its eager plan.
                # Per-request row boundaries come from preprocess (real
                # seq_lens): the runner's static_cm, not cache_handle, holds
                # this step's plan, so get_qo_indptr_buf would be stale here.
                replay = prefill_runner.run(
                    static_inputs={
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                    },
                    request_ids=request_ids,
                    seq_lens=list(seq_lens),
                )
                # Views, not clones: both are consumed inside this step
                # (index_select + the eager plane sync) before this runner
                # replays again.
                hidden = replay.get_view("hidden")
                prenorm = replay.get_view("prenorm")
                last_token_indices = kwargs.get("last_token_indices")
                assert last_token_indices is not None
            else:
                # Eager prefill (no captured bucket for this shape, or
                # MSTAR_GLM52_MTP_CAPTURE_PREFILL=0).
                hidden, prenorm = self._hidden(
                    input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"),
                    with_prenorm=True)
                qo_indptr_buf = cache_handle.get_qo_indptr_buf(_MAIN)
                if qo_indptr_buf is not None:
                    last_token_indices = (qo_indptr_buf[1:] - 1).long()
                else:
                    last_token_indices = kwargs.get("last_token_indices")
                    assert last_token_indices is not None
            # Emit the prefill token through the engine sampler, exactly as
            # flag-off does — bit-parity for the first emitted token.
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

        timer = self._mtp_timer
        timer.report(logger)
        timer.begin()
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
            # Views into the trunk graph's static outputs (no clone launches):
            # logits go straight into argmax below, hidden/prenorm rows are
            # copied into the sync pass this same step — all before the next
            # trunk replay.
            hidden = replay.get_view("hidden")
            prenorm = replay.get_view("prenorm")
            logits = replay.get_view("logits")
        else:
            self._warn_mtp_trunk_eager_once(len(request_ids), sum(seq_lens))
            hidden, prenorm = self._hidden(
                input_ids, position_ids, cache_handle, kwargs.get("dsa_ctx"),
                with_prenorm=True)
            logits = self.lm_head(hidden)  # (sum(k+1), vocab)
        results: dict[str, NameToTensorList] = {}
        rewinds: list[int] = []
        sync_tokens, pair_hiddens = [], []
        # ONE device->host round trip for the whole verify. The per-request
        # form (``mtp_greedy_verify`` on device slices, then ``int(tok)`` per
        # emitted token for the EOS scan) was 2 + e syncs per request per
        # step; each one also drains whatever the GPU still has queued. The
        # target argmax over all rows and the drafts (already on device from
        # last step's chain) travel together, and everything after this
        # line is host arithmetic. Emitted tokens are a VIEW of the target
        # argmax: greedy verify accepts draft j iff it equals target[j], so
        # ``drafts[:n_acc] + [target[n_acc]] == target[:n_acc + 1]`` — same
        # values as ``mtp_greedy_verify`` returned, no gather needed.
        target_argmax_all = logits.argmax(dim=-1)  # (sum(k+1),)
        total_rows = row_starts[-1]
        timer.mark("trunk")
        phase_runner = kwargs.get("mtp_draft_phase_runner")
        rows_uniform = all(
            sl == self.config.mtp_num_draft_tokens + 1 for sl in seq_lens)
        prepared = False
        pair_rows = None
        if phase_runner is not None and self._mtp_phase_prepare and rows_uniform:
            # Hoist the e-independent half of the draft phase into the window
            # where the host otherwise sits blocked in the verify readback
            # below (the GPU is still running the trunk):
            #   - full-row sync inputs: rows j < e of request i ARE the
            #     emitted tokens (emitted is a view of the target argmax);
            #     rows >= e are rejected continuations landing on the same
            #     transient pad slots the zero pads used — causality keeps
            #     real rows from attending to them, and the chain/next step
            #     overwrites them in place;
            #   - positions: mtp_sync_padded_layout's two ranges concatenate
            #     to range(P0+1, P0+rows+1) — contiguous, e-independent, and
            #     IDENTICAL to the post-verify value (pinned by
            #     test_glm52_mtp_phase_prepare);
            #   - the slot-0 FlashInfer plan (padded sync, k+1 rows at P0).
            # Only last_rows, chain_pos_{it} and the k-1 chain plans depend
            # on e and stay after the verify.
            pair_rows = self._mtp_pair_rows(hidden, prenorm)
            pos_full: list[int] = []
            for i, rid in enumerate(request_ids):
                p_now = cache_handle._get_state(rid, _MAIN).position_id_start
                p0 = p_now - seq_lens[i]
                pos_full.extend(range(p0 + 1, p0 + 1 + seq_lens[i]))
            phase_runner.prepare(
                static_inputs={
                    "sync_ids": target_argmax_all,
                    "pair_hidden": pair_rows,
                    "sync_position_ids": pinned(pos_full, torch.long),
                },
                request_ids=request_ids,
                seq_lens=list(seq_lens),
                plan_ctx={"phase": "prepare"},
            )
            prepared = True
            timer.mark("prepare")
        try:
            host = torch.cat([input_ids, target_argmax_all]).tolist()
            timer.mark("verify_d2h")
            inputs_h, target_h = host[:total_rows], host[total_rows:]
            if pair_rows is None:
                pair_rows = self._mtp_pair_rows(hidden, prenorm)
            eos_ids = self.config.eos_token_ids
            for i, rid in enumerate(request_ids):
                lo, hi = row_starts[i], row_starts[i + 1]
                m = seq_lens[i]
                n_acc = mtp_greedy_verify_host(
                    inputs_h[lo + 1:hi], target_h[lo:hi])
                # Raw (pre-truncation) emission is the draft-quality signal.
                self._mtp_stat_steps += 1
                self._mtp_stat_emitted += n_acc + 1
                self._mtp_stat_acc_hist[n_acc] += 1
                # Truncate: max_tokens budget first, then first stop id. EOS is
                # always the LAST element after truncation, which is the
                # contract check_stop relies on.
                budget = self._mtp_max_tokens[rid] - self._mtp_emitted[rid]
                e = min(n_acc + 1, max(budget, 1))
                if not self._mtp_ignore_eos[rid]:
                    for j in range(e):
                        if target_h[lo + j] in eos_ids:
                            e = j + 1
                            break
                emitted = target_argmax_all[lo:lo + e]
                self._mtp_emitted[rid] += e
                # m tokens appended KV this forward; the committed prefix is
                # input[0] plus the e-1 now-emitted drafts. The bonus was never
                # processed.
                rewinds.append(m - e)
                sync_tokens.append(emitted)
                pair_hiddens.append(pair_rows[lo:lo + e])
                results[rid] = {"new_token": [emitted]}
            self._maybe_log_mtp_acceptance()
            cache_handle.rewind_seq_lens(rewinds)
            timer.mark("verify_host")
            sync_runner = kwargs.get("mtp_sync_runner")
            if sync_runner is None and self._mtp_capture_sync:
                self._warn_mtp_sync_eager_once(len(request_ids))
            drafts = self._mtp_sync_and_draft(
                cache_handle, sync_tokens, pair_hiddens,
                draft_runner=kwargs.get("mtp_draft_runner"),
                sync_runner=sync_runner,
                phase_runner=phase_runner,
                prepared=prepared)
        except Exception:
            if prepared:
                # A dummy slot left aliased to a live request would let a
                # later replay's padding-slot reset free live pages — restore
                # before propagating (run()'s own finally covers everything
                # from its call onward; abort twice is a harmless no-op).
                phase_runner.abort_prepare(request_ids, list(seq_lens))
            raise
        for i, rid in enumerate(request_ids):
            emitted = results[rid]["new_token"][0]
            results[rid]["text_inputs"] = [
                torch.cat([emitted[-1:], drafts[i]])]
        timer.mark("tail")
        timer.end()
        return results

    def _mtp_sync_and_draft(
        self,
        cache_handle: BatchedCacheManager,
        sync_tokens: list[torch.Tensor],
        pair_hiddens: list[torch.Tensor],
        draft_runner=None,
        sync_runner=None,
        phase_runner=None,
        prepared: bool = False,
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

        assert not prepared or phase_runner is not None, (
            "prepared=True is only meaningful for the one-graph draft phase")
        # Sync pass (+ draft 1 from its last row): plane positions
        # start-e .. start-1, token positions start-e+1 .. start. When the
        # phase runner was PREPARED before the verify, slot 0 is already
        # planned at P0 and the counters must stay at P0+e — no rewind.
        if not prepared:
            cache_handle.rewind_seq_lens(e_list)
        if phase_runner is not None:
            # ONE graph for sync + draft-1 head + the k-1 chain iterations.
            # Inputs mirror the padded sync (real rows first, pads after) plus
            # the host-known gather/positions the graph cannot derive; the
            # runner's plan_fn plans all k slots from e_list, then one replay.
            rows = k + 1
            assert all(e <= rows for e in e_list), (
                f"draft-phase graph got rows {e_list} outside [1, {rows}]")
            if prepared:
                # sync_ids / pair_hidden / sync_position_ids and the slot-0
                # plan went in through prepare() before the verify readback;
                # only the e-dependent leftovers travel here.
                phase_inputs = {
                    "last_rows": pinned(
                        [i * rows + e - 1 for i, e in enumerate(e_list)],
                        torch.long),
                }
                plan_ctx = {"e_list": list(e_list), "phase": "finish"}
            else:
                pos_l, last_l, _over = mtp_sync_padded_layout(e_list, starts, k)
                sync_ids = torch.zeros(num * rows, dtype=torch.long, device=device)
                pair_h = torch.zeros(
                    (num * rows, pair_hiddens[0].shape[-1]),
                    dtype=pair_hiddens[0].dtype, device=device)
                for i, (t, h) in enumerate(zip(sync_tokens, pair_hiddens, strict=True)):
                    sync_ids[i * rows:i * rows + t.shape[0]] = t
                    pair_h[i * rows:i * rows + h.shape[0]] = h
                phase_inputs = {
                    "sync_ids": sync_ids,
                    "pair_hidden": pair_h,
                    "sync_position_ids": pinned(pos_l, torch.long),
                    "last_rows": pinned(last_l, torch.long),
                }
                plan_ctx = {"e_list": list(e_list)}
            for it in range(1, k):
                phase_inputs[f"chain_pos_{it}"] = pinned(
                    [st + it for st in starts], torch.long)
            out = phase_runner.run(
                static_inputs=phase_inputs,
                request_ids=request_ids,
                seq_lens=[rows] * num,
                plan_ctx=plan_ctx,
            )
            # plan_fn left the counters at start+(k-1) (P+e+k-1); the chain
            # entries beyond the counter are transient, as in the 3-graph path.
            if k > 1:
                cache_handle.rewind_seq_lens([k - 1] * num)
            self._mtp_timer.mark("draft_phase")
            drafts = out["drafts"]  # (num, k), owned
            return [drafts[i] for i in range(num)]
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
            # Host-known ints go to the device through PINNED staging
            # (mstar/utils/pinned_staging.py): the runner's static-input
            # copy is non_blocking, so nothing here waits for the trunk to
            # finish — the sync pass is queued right behind it.
            out = sync_runner.run(
                static_inputs={
                    "sync_ids": sync_ids,
                    "pair_hidden": pair_h,
                    "position_ids": pinned(pos_l, torch.long),
                },
                request_ids=request_ids,
                seq_lens=[rows] * num,
            )
            # The runner advanced `rows` per request; only e were real.
            cache_handle.rewind_seq_lens(over_advance)
            # Views: index_select'd immediately below.
            h_head, h_raw = out.get_view("h_head"), out.get_view("h_raw")
            last_rows = to_device_async(last_l, torch.long, device)
        else:
            cache_handle.set_layer_idx(mtp_layer)
            cache_handle.plan_attention(
                seq_lens=e_list, is_causal=True, label=_MAIN)
            pos_list: list[int] = []
            for st, e in zip(starts, e_list, strict=True):
                pos_list.extend(range(st - e + 1, st + 1))
            positions = to_device_async(pos_list, torch.long, device)
            self._plan_rope(cache_handle, 
                seq_lens=e_list, pos_ids=positions, label=_MAIN)
            h_head, h_raw = mtp(
                embed(torch.cat(sync_tokens)), torch.cat(pair_hiddens),
                cache_handle, positions,
            )
            cache_handle.advance_seq_lens()
            # Packed (unpadded) layout: last real row via cumsum (on host).
            last_l_eager: list[int] = []
            acc = 0
            for e in e_list:
                acc += e
                last_l_eager.append(acc - 1)
            last_rows = to_device_async(last_l_eager, torch.long, device)

        self._mtp_timer.mark("sync")
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
            pos_it = [st + it for st in starts]
            if draft_runner is not None:
                # Replay the captured chain iteration. The runner plans on
                # its persistent wrappers from the aliased states (rope at
                # counter+1 == st+it, matching ``positions``), replays, and
                # advances +1 per request — the same bookkeeping as the
                # eager body, so the final k-1 rewind below is unchanged.
                # Nothing in this iteration blocks on the previous one: the
                # positions are pinned host memory (async copy-in), the plan
                # inside run() stages its indices the same way, and the
                # wrapper's plan fence only waits for the PREVIOUS plan's
                # DMAs (i.e. for the GPU to reach the previous replay), so
                # the host stays one replay ahead of the device.
                out = draft_runner.run(
                    static_inputs={
                        "draft_ids": prev_d,
                        "prev_hidden": prev_h,
                        "position_ids": pinned(pos_it, torch.long),
                    },
                    request_ids=request_ids,
                    seq_lens=ones,
                )
                # draft_ids must be an owned copy (it is kept in draft_cols
                # across later replays that overwrite the static output);
                # prev_hidden is only copied into the next replay's static
                # input, so a view is enough.
                prev_d = out["draft_ids"]
                prev_h = out.get_view("prev_hidden")
            else:
                positions = to_device_async(pos_it, torch.long, device)
                cache_handle.set_layer_idx(mtp_layer)
                cache_handle.plan_attention(
                    seq_lens=ones, is_causal=True, label=_MAIN)
                self._plan_rope(cache_handle, 
                    seq_lens=ones, pos_ids=positions, label=_MAIN)
                it_head, prev_h = mtp(
                    embed(prev_d), prev_h, cache_handle, positions)
                cache_handle.advance_seq_lens()
                prev_d = self.lm_head(it_head).argmax(dim=-1)
            draft_cols.append(prev_d)
            self._mtp_timer.mark(f"chain{it}")
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
