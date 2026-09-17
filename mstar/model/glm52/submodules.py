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
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
    PiecewiseCudaGraphConfig,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    MlaAttentionStep,
    MlaSubPlan,
    SamplerStep,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.model.glm52.config import (
    ATTN_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
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

logger = logging.getLogger(__name__)

_MAIN = "main"
# Piecewise-graph labels of the MTP step.
MTP_TRUNK_LABEL = "mtp_trunk"
MTP_DRAFT_LABEL = "mtp_draft"
MTP_SYNC_LABEL = "mtp_sync"
MTP_PREFILL_LABEL = "mtp_prefill"
MTP_DRAFT_PHASE_LABEL = "mtp_draft_phase"
# Output/edge name for the prefill's [emitted, k drafts] bundle. Deliberately
# NOT "text_inputs": the conductor seeds persist_signals from initial_signals,
# where "text_inputs" is the PROMPT, so a persisted edge of that name would
# feed decode the whole prompt back as its first step.
MTP_DRAFT_BUNDLE = "mtp_draft_bundle"


def mtp_sync_padded_layout(
    e_list: list[int], starts: list[int], k: int,
) -> tuple[list[int], list[int], list[int]]:
    """Row layout for a PADDED MTP sync pass."""
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
    """PACKED piecewise config with exactly one (bs, [rows]*bs) bucket per
    capture batch size: an MTP step feeds a fixed row count per request, so
    the generic bs x token-bucket cross product would enumerate shapes that
    never occur. PACKED because replay slices outputs to the real rows and
    pads absent requests with zero-length plan rows.
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

    def replay_seq_lens(
        self, shape: PiecewiseCaptureShape, seq_lens: list[int] | None, real_bs: int,
    ) -> list[int]:
        if seq_lens is None:
            raise ValueError("MTP piecewise replay requires seq_lens")
        return list(seq_lens) + [0] * (shape.bs - real_bs)


class _MtpStepTimer:
    """nsys-lite for one MTP decode step, gated by MSTAR_GLM52_MTP_STEP_TIMING=N."""

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
        # Under MTP the forward is a host-side driver (verify readback, per
        # request bookkeeping, region replays, rewinds) whose heavy halves are
        # captured graphs compiled at capture; compiling the driver itself
        # gains nothing and lets dynamo inline plan-time host code (the
        # "Only CPU tensors can be pinned" failure of 2026-08-28).
        self.disable_torch_compile = config.mtp_num_draft_tokens > 0
        # DSA indexer k-cache (dsa.py): per-request index keys, appended by
        # FULL layers each forward when dsa_long_context is on; evicted in
        # cleanup_request.
        self._dsa_k_store = Glm52DsaKStore()
        # MTP per-request state: total emitted tokens (incl. the prefill-
        # emitted one — max_tokens counts it) and the stop parameters stashed
        # at prepare_inputs time, so the batched step truncates emission
        # without engine round trips.
        self._mtp_emitted: dict[str, int] = {}
        self._mtp_max_tokens: dict[str, int] = {}
        self._mtp_ignore_eos: dict[str, bool] = {}
        # Which trunk stream the MTP plane pairs drafts against (see
        # _mtp_pair_rows): post-final-norm (vLLM's convention) by default.
        self._mtp_pair_postnorm = (
            os.environ.get("MSTAR_GLM52_MTP_PAIR_POSTNORM", "1") == "1"
        )
        # Capture the decode sync pass as a padded (bs, k+1) piecewise graph.
        self._mtp_capture_sync = (
            os.environ.get("MSTAR_GLM52_MTP_CAPTURE_SYNC", "1") == "1"
        )
        # Capture the MTP prefill trunk (embed + layers over the packed
        # prompt) over the same token buckets the k=0 config captures; the
        # sample, whole-prompt plane sync and draft chain stay outside.
        self._mtp_capture_prefill = (
            os.environ.get("MSTAR_GLM52_MTP_CAPTURE_PREFILL", "1") == "1"
        )
        # The whole decode draft phase as ONE graph (padded sync pass,
        # draft-1 head, k-1 chain iterations, k attention sub-plans planned
        # before one replay). Requires sync capture; =0 restores the
        # three-graph path, which stays the fallback for missing buckets.
        self._mtp_draft_phase_graph = (
            os.environ.get("MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH", "1") == "1"
        )
        # Hoist the accepted-count-independent half of the draft phase above
        # the verify readback: the full-row sync inputs (rows < e are the
        # emitted tokens, rows >= e rejected continuations on transient
        # slots), the contiguous positions P0+1..P0+k+1, and sub-plan 0's
        # attention plan go in through runner.stage() while the host would
        # otherwise sit in the .tolist(); only last_rows, chain_pos_* and the
        # k-1 chain sub-plans wait for e. Bit-exact by construction; default
        # off until a TP8 arm measures it.
        self._mtp_phase_prepare = (
            os.environ.get("MSTAR_GLM52_MTP_PHASE_PREPARE", "0") == "1"
        )
        # One-shot warnings: an MTP step whose graphs silently run eager is a
        # 13x regression that looks like "MTP is slow".
        self._mtp_trunk_eager_warned = False
        self._mtp_draft_eager_warned = False
        self._mtp_sync_eager_warned = False
        # Acceptance instrumentation: raw emitted tokens (n_accepted + 1,
        # pre-truncation), request-step count, n_accepted histogram.
        self._mtp_stat_emitted = 0
        self._mtp_stat_steps = 0
        self._mtp_stat_logged = 0
        self._mtp_stat_acc_hist = [0] * (config.mtp_num_draft_tokens + 1)
        self._mtp_timer = _MtpStepTimer(
            int(os.environ.get("MSTAR_GLM52_MTP_STEP_TIMING", "0") or "0"))

    def set_load_heartbeat_stop(self, stop) -> None:
        """Adopt the load-time GPU liveness tick; stopped before capture."""
        self._load_heartbeat_stop = stop

    def _stop_load_heartbeat(self) -> None:
        # getattr: the graph-config getters call this first, and CPU tests
        # construct partially-initialized submodules that never ran __init__
        stop = getattr(self, "_load_heartbeat_stop", None)
        if stop is not None:
            stop.set()
            self._load_heartbeat_stop = None

    def cleanup_request(self, request_id: str):
        self._dsa_k_store.evict(request_id)
        self._mtp_emitted.pop(request_id, None)
        self._mtp_max_tokens.pop(request_id, None)
        self._mtp_ignore_eos.pop(request_id, None)
        super().cleanup_request(request_id)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    MTP_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    # ── resources ──

    def _kv(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[KV_RESOURCE]

    def _attn(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[ATTN_RESOURCE]

    def _attn_step(
        self, sub_plans: tuple[MlaSubPlan, ...] | None = None,
        first_sub_plan: int = 0, num_sub_plans: int | None = None,
    ):
        if self.config.mla_absorb:
            return MlaAttentionStep(
                causal=True, sub_plans=sub_plans,
                first_sub_plan=first_sub_plan, num_sub_plans=num_sub_plans,
            )
        if sub_plans is not None:
            raise RuntimeError("sub-plans need the absorbed MLA attention resource")
        return AttentionStep(causal=True)

    def _kv_attn_step(
        self, request_ids: list[str], spans: list[int],
        sub_plans: tuple[MlaSubPlan, ...] | None = None, commit: bool = True,
        first_sub_plan: int = 0, num_sub_plans: int | None = None,
    ) -> SubmoduleStep:
        """A step over the trunk stream: KV pages for ``spans`` plus the
        attention plan(s). ``commit=False`` for a pass whose rows are
        transient (the draft phase)."""
        return SubmoduleStep(
            segments=[
                Segment(rid, _MAIN, span)
                for rid, span in zip(request_ids, spans, strict=True)
            ],
            steps={
                KV_RESOURCE: KVStep(commit=commit),
                ATTN_RESOURCE: self._attn_step(sub_plans, first_sub_plan, num_sub_plans),
            },
        )

    # Host-side bookkeeping reached from inside the compiled forward: kept
    # out of dynamo, which once inlined a plan's pinned staging into the
    # graph ("Only CPU tensors can be pinned", 8/20 requests failed).
    @torch.compiler.disable
    def _eager_step(
        self, engine_inputs: ModelInputsFromEngine, request_ids: list[str],
        spans: list[int], sub_plans: tuple[MlaSubPlan, ...] | None = None,
        commit: bool = True,
    ) -> tuple[SubmoduleStep, StepContext]:
        """Admit + plan a KV/attention step by hand, for an eager pass this submodule
        drives itself (an MTP plane pass with no captured bucket).
        """
        step = self._kv_attn_step(request_ids, spans, sub_plans, commit)
        ctx = StepContext(
            request_ids=tuple(request_ids), graph_walk="decode", slot=0, capture=False,
        )
        step.set_ctx(ctx)
        kv, attn = self._kv(engine_inputs), self._attn(engine_inputs)
        outcome = kv.admit(step.get(KV_RESOURCE), ctx)
        if not outcome.ok:
            raise RuntimeError(f"KV admit failed: {outcome.reason.message}")
        ctx.plan_results[KV_RESOURCE] = kv.plan(step.get(KV_RESOURCE), ctx)
        ctx.plan_results[ATTN_RESOURCE] = attn.plan(step.get(ATTN_RESOURCE), ctx)
        return step, ctx

    @torch.compiler.disable
    def _commit_eager(
        self, engine_inputs: ModelInputsFromEngine, step: SubmoduleStep, ctx: StepContext,
    ) -> None:
        self._kv(engine_inputs).commit(step.get(KV_RESOURCE), ctx)
        self._attn(engine_inputs).commit(step.get(ATTN_RESOURCE), ctx)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep | None:
        prefill_tokens = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            }
        sampler = SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens)
        segments = [
            Segment(rid, _MAIN, inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        if self.config.mtp_num_draft_tokens > 0:
            # The MTP regions declare, plan and commit their own KV and
            # attention work per replay (eager fallbacks through
            # _eager_step); the outer step only samples the prefill token.
            if graph_walk != "prefill":
                return None
            return SubmoduleStep(segments=segments, steps={SAMPLER_RESOURCE: sampler})
        return SubmoduleStep(
            segments=segments,
            steps={
                KV_RESOURCE: KVStep(),
                ATTN_RESOURCE: self._attn_step(),
                SAMPLER_RESOURCE: sampler,
            },
        )

    # ── capture configs ──

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
        """The reference MoE dispatch paths (.nonzero() / host loops) are not
        stream-capturable; only the fused fp8 path is.
        """
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        return fp8_reference or naive_tp

    def _compile_flags(self) -> dict[str, Any]:
        # MSTAR_GLM52_GRAPH_COMPILE=0 captures the eager forward (escape hatch
        # for an Inductor toolchain crash); the mode is the cuBLAS one
        # (cuda_graph_runner.resolve_compile_mode: 90.03 -> 96.97 tok/s TP8).
        return {
            "compile": os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
            "compile_mode": "default",
        }

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts (per-param dtypes
        are fixed at load — fp32 block scales + router bias, uint8 fp8)."""
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
    ) -> list[CudaGraphConfig]:
        self._stop_load_heartbeat()
        if self.config.dsa_long_context:
            # DSA maintenance is host-side per-request work; a captured
            # decode would skip index upkeep. Eager-only.
            return []
        if self._moe_capture_blocked(tp_world_size):
            return []
        if self.config.mtp_num_draft_tokens > 0:
            # The MTP step's host phases (verify readback, rewind, emission
            # bookkeeping) are not capturable; its heavy halves are piecewise
            # graphs instead (get_piecewise_cuda_graph_configs).
            logger.info(
                "Glm52LLMSubmodule: MTP k=%d — no full-forward CUDA graphs; the "
                "trunk verify, draft phase and prefill trunk capture piecewise.",
                self.config.mtp_num_draft_tokens,
            )
            return []
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        flags = self._compile_flags()
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                **flags,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill",
                capture_token_lengths=list(prefill_buckets),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                ),
                capture_batch_sizes=list(prefill_batch_sizes),
                **flags,
            ),
        ]

    def get_piecewise_cuda_graph_configs(
        self,
        device: torch.device,
        autocast_dtype: torch.dtype,
        tp_world_size: int = 1,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """The MTP step's graphs. ``mtp_trunk``: the verify forward (embed + layers +
        lm_head over the packed (bs, k+1) rows).
        """
        self._stop_load_heartbeat()
        k = self.config.mtp_num_draft_tokens
        if (
            k <= 0
            or self.config.dsa_long_context
            or self._moe_capture_blocked(tp_world_size)
        ):
            return {}
        rows = k + 1
        flags = self._compile_flags()
        batch_sizes = list(self.MTP_CAPTURE_BATCH_SIZES)

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
                "position_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
            }

        def make_draft_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {
                "draft_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
                "prev_hidden": torch.zeros(
                    shape.total_tokens, self.config.hidden_size,
                    dtype=autocast_dtype, device=device),
                "position_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
            }

        def make_sync_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {
                "sync_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
                "pair_hidden": torch.zeros(
                    shape.total_tokens, self.config.hidden_size,
                    dtype=autocast_dtype, device=device),
                "position_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
            }

        def make_phase_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            bs = shape.bs
            return {
                "sync_ids": torch.zeros(shape.total_tokens, dtype=torch.long, device=device),
                "pair_hidden": torch.zeros(
                    shape.total_tokens, self.config.hidden_size,
                    dtype=autocast_dtype, device=device),
                "sync_position_ids": torch.zeros(
                    shape.total_tokens, dtype=torch.long, device=device),
                "last_rows": torch.zeros(bs, dtype=torch.long, device=device),
                # one (bs,) input per chain iteration, not one flat buffer:
                # the runner pads each input's leading dim separately
                **{
                    f"chain_pos_{it}": torch.zeros(bs, dtype=torch.long, device=device)
                    for it in range(1, k)
                },
            }

        configs: dict[str, PiecewiseCudaGraphConfig] = {
            MTP_TRUNK_LABEL: Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_trunk_captured,
                make_static_inputs=make_static_inputs,
                declare_step=self._trunk_region_step,
                capture_batch_sizes=batch_sizes,
                **flags,
            )
        }
        if k >= 2:
            configs[MTP_DRAFT_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=1,
                capture_fn=self._mtp_draft_captured,
                make_static_inputs=make_draft_static_inputs,
                declare_step=self._chain_region_step,
                capture_batch_sizes=batch_sizes,
                **flags,
            )
        if self._mtp_capture_sync:
            configs[MTP_SYNC_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_sync_captured,
                make_static_inputs=make_sync_static_inputs,
                declare_step=self._trunk_region_step,
                capture_batch_sizes=batch_sizes,
                **flags,
            )
        if self._mtp_capture_sync and self._mtp_draft_phase_graph and self.config.mla_absorb:
            configs[MTP_DRAFT_PHASE_LABEL] = Glm52MtpTrunkGraphConfig(
                rows_per_request=rows,
                capture_fn=self._mtp_draft_phase_captured,
                make_static_inputs=make_phase_static_inputs,
                declare_step=self._draft_phase_region_step,
                capture_batch_sizes=batch_sizes,
                **flags,
            )
        if self._mtp_capture_prefill:
            configs[MTP_PREFILL_LABEL] = PiecewisePackedConfig(
                total_tokens=list(
                    self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS),
                capture_fn=self._mtp_prefill_captured,
                make_static_inputs=make_static_inputs,
                declare_step=self._trunk_region_step,
                capture_batch_sizes=list(
                    self.config.prefill_capture_batch_sizes
                    or self.PREFILL_CAPTURE_BATCH_SIZES),
                **flags,
            )
        return configs

    # ── region step declarations (capture and every replay) ──

    def _trunk_region_step(
        self, request_ids: list[str], seq_lens: list[int],
    ) -> SubmoduleStep:
        """One pass appending ``seq_lens`` rows per request to the trunk
        stream and committing them (trunk verify, sync pass, prefill)."""
        return self._kv_attn_step(request_ids, seq_lens)

    def _chain_region_step(
        self, request_ids: list[str], seq_lens: list[int],
    ) -> SubmoduleStep:
        """One draft-chain iteration: one row per present request, committed
        (+1); the caller rewinds the k-1 transient rows at the end."""
        return self._kv_attn_step(request_ids, seq_lens)

    def _draft_phase_region_step(
        self, request_ids: list[str], seq_lens: list[int],
        e_list: list[int] | None = None, phase: str | None = None,
    ) -> SubmoduleStep:
        """The draft-phase graph's step: with the stream at P0+e (the trunk's
        k+1 rows committed, k+1-e rewound), sub-plan 0 is the padded sync
        pass — k+1 rows at P0 attending P0+k+1 — and sub-plan ``it`` (1..k-1)
        one chain row attending P0+e+it. Nothing commits: every row past
        P0+e is transient and the next trunk step overwrites it. The segment
        span covers the deepest sub-plan so admit reserves its pages.
        ``e_list`` arrives per replay (``run(step_kwargs=)``); capture and
        warmup have fresh streams at 0 and plan e=0 (P0 = 0), which gives
        every sub-plan a valid shape.
        """
        k = self.config.mtp_num_draft_tokens
        rows = k + 1
        kv = self._kv()
        present = [sl > 0 for sl in seq_lens]
        stored = [kv.stored_len(rid) if p else 0 for rid, p in zip(request_ids, present, strict=True)]
        if phase == "prepare":
            # the trunk's k+1 rows are committed: the sync pass attends
            # exactly the stream as it stands
            sub_plans = (MlaSubPlan(
                q_lens=tuple(rows if p else 0 for p in present),
                kv_lens=tuple(s if p else 0 for s, p in zip(stored, present, strict=True)),
            ),)
            return self._kv_attn_step(
                request_ids, [0] * len(present), sub_plans, commit=False,
                first_sub_plan=0, num_sub_plans=k,
            )
        if e_list is None:
            e_list = [0] * sum(present)
        e_iter = iter(e_list)
        e_by_row = [next(e_iter) if p else 0 for p in present]
        chain = [MlaSubPlan(
            q_lens=tuple(1 if p else 0 for p in present),
            kv_lens=tuple(s + it if p else 0 for s, p in zip(stored, present, strict=True)),
        ) for it in range(1, k)]
        if phase == "finish":
            spans = [k - 1 if p else 0 for p in present]
            return self._kv_attn_step(
                request_ids, spans, tuple(chain), commit=False,
                first_sub_plan=1, num_sub_plans=k,
            )
        p0 = [s - e for s, e in zip(stored, e_by_row, strict=True)]
        spans = [max(rows - e, k - 1) if p else 0 for p, e in zip(present, e_by_row, strict=True)]
        sub_plans = [MlaSubPlan(
            q_lens=tuple(rows if p else 0 for p in present),
            kv_lens=tuple(p + rows if pr else 0 for p, pr in zip(p0, present, strict=True)),
        ), *chain]
        return self._kv_attn_step(request_ids, spans, tuple(sub_plans), commit=False)

    # ── captured regions ──

    def _mtp_trunk_captured(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The MTP decode trunk: embed + layers + lm_head over the packed
        (bs, k+1) rows, from the runner's static buffers."""
        hidden, prenorm = self._hidden(
            call.static_inputs["input_ids"], call.static_inputs["position_ids"],
            with_prenorm=True)
        return {"hidden": hidden, "prenorm": prenorm, "logits": self.lm_head(hidden)}

    def _mtp_prefill_captured(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The MTP prefill trunk over the packed prompt: both streams for
        every row (last rows feed the sample, all rows the plane sync). No
        lm_head inside — a (tokens, vocab) static output would be 300 MB."""
        hidden, prenorm = self._hidden(
            call.static_inputs["input_ids"], call.static_inputs["position_ids"],
            with_prenorm=True)
        return {"hidden": hidden, "prenorm": prenorm}

    def _mtp_draft_captured(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """One draft-chain iteration: fuse the previous draft's embedding with
        the chained raw hidden through the layer-78 module, head argmax.
        Output names match the input names so each replay feeds the next."""
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(call.static_inputs["draft_ids"]),
            call.static_inputs["prev_hidden"],
            call.static_inputs["position_ids"],
        )
        return {"draft_ids": self.lm_head(h_head).argmax(dim=-1), "prev_hidden": h_raw}

    def _mtp_sync_captured(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The PADDED sync pass: the committed tokens' embeddings fused with their paired
        trunk rows through the layer-78 module, k+1 rows per request (real first, pads
        after).
        """
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(call.static_inputs["sync_ids"]),
            call.static_inputs["pair_hidden"],
            call.static_inputs["position_ids"],
        )
        return {"h_head": h_head, "h_raw": h_raw}

    def _mtp_draft_phase_captured(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The whole draft phase: padded sync pass on sub-plan 0, gather each request's
        last real row, draft 1 = head argmax, then k-1 chain iterations on sub-plans
        1..k-1.
        """
        k = self.config.mtp_num_draft_tokens
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        attn = call.resources[ATTN_RESOURCE]
        static = call.static_inputs
        attn.select_plan_slot(0)
        h_head, h_raw = mtp(
            embed(static["sync_ids"]), static["pair_hidden"], static["sync_position_ids"],
        )
        last = static["last_rows"]
        prev_h = h_raw.index_select(0, last)
        prev_d = self.lm_head(h_head.index_select(0, last)).argmax(dim=-1)
        cols = [prev_d]
        for it in range(1, k):
            attn.select_plan_slot(it)
            it_head, prev_h = mtp(embed(prev_d), prev_h, static[f"chain_pos_{it}"])
            prev_d = self.lm_head(it_head).argmax(dim=-1)
            cols.append(prev_d)
        attn.select_plan_slot(0)
        return {"drafts": torch.stack(cols, dim=1)}  # (bs, k)

    # ── per-step contract ──

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        if self.config.mtp_num_draft_tokens > 0:
            rid = fwd_info.request_id
            sampling = fwd_info.resource_configs[SAMPLER_RESOURCE]
            # Greedy-only: decode drafts and verification bypass the sampler,
            # so a temperature would be silently ignored and a repetition
            # penalty would move even the prefill argmax. Refuse this rid.
            if sampling.temperature != 0 or sampling.repetition_penalty != 1:
                raise RuntimeError(
                    f"request {rid}: MTP speculative decoding is greedy-only "
                    f"but the request asks for temperature={sampling.temperature}, "
                    f"repetition_penalty={sampling.repetition_penalty}. Send "
                    "temperature=0 without a penalty, or serve a k=0 config."
                )
            self._mtp_max_tokens[rid] = fwd_info.max_tokens
            self._mtp_ignore_eos[rid] = sampling.ignore_eos
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
        )

    def _runner_for(self, engine_inputs, label: str, bs: int, tokens: int):
        runner = (getattr(engine_inputs, "piecewise_runners", None) or {}).get(label)
        if runner is not None and runner.can_run(bs, tokens):
            return runner
        return None

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        kv = self._kv(engine_inputs)
        request_ids = list(engine_inputs.request_ids)
        seq_lens = [inp.input_seq_len for inp in inputs]
        bs = len(inputs)
        k = self.config.mtp_num_draft_tokens

        # Which MTP regions replay this step, decided once so the forward
        # never re-decides: a region without a bucket for this shape runs
        # eager through _eager_step.
        runners: dict[str, Any] = {}
        if k > 0:
            rows = k + 1
            if graph_walk == "prefill":
                runners["prefill"] = self._runner_for(
                    engine_inputs, MTP_PREFILL_LABEL, bs, sum(seq_lens))
            elif graph_walk == "decode":
                runners["trunk"] = self._runner_for(
                    engine_inputs, MTP_TRUNK_LABEL, bs, sum(seq_lens))
                runners["sync"] = self._runner_for(
                    engine_inputs, MTP_SYNC_LABEL, bs, rows * bs)
                runners["phase"] = self._runner_for(
                    engine_inputs, MTP_DRAFT_PHASE_LABEL, bs, rows * bs)
            runners["draft"] = self._runner_for(engine_inputs, MTP_DRAFT_LABEL, bs, bs)

        device = self.get_device()
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window.
        long_context = self.config.dsa_long_context
        limit = self.config.max_seq_len if long_context else self.config.index_topk
        topk = self.config.index_topk
        pos_ids_list: list[int] = []
        spans: list[Glm52DsaRequestSpan] = []
        needs_selection = False
        q_start = 0
        page_tables = self._attn(engine_inputs).page_tables() if long_context else None
        for i, (rid, sl) in enumerate(zip(request_ids, seq_lens, strict=True)):
            start = kv.stored_len(rid)
            if start + sl > limit:
                raise RuntimeError(
                    f"request {rid}: context {start + sl} exceeds {limit}, "
                    + (
                        "the configured max_seq_len serving window."
                        if long_context
                        else "the regime where dense MLA is exactly GLM-5.2's "
                        "DSA computation. Long context needs dsa_long_context=True."
                    )
                )
            if long_context:
                if start + sl > topk:
                    if sl > 1:
                        raise RuntimeError(
                            f"request {rid}: prefill context {start + sl} exceeds "
                            f"index_topk={topk}; sparse attention beyond topk is "
                            "decode-only."
                        )
                    needs_selection = True
                spans.append(Glm52DsaRequestSpan(
                    request_id=rid, q_start=q_start, q_len=sl,
                    ctx_start=start, page_indices=list(page_tables[i]),
                ))
                q_start += sl
            pos_ids_list.extend(range(start, start + sl))
        # pinned staging: a pageable torch.tensor(..., device=cuda) here
        # drains the stream before the step even starts
        position_ids = to_device_async(pos_ids_list, torch.long, device)
        seq_len_t = to_device_async(seq_lens, torch.long, device)
        return {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
            "position_ids": position_ids,
            # eager prefill: the last row per request, when no plan buffer
            # carries qo_indptr
            "last_token_indices": seq_len_t.cumsum(0) - 1,
            "seq_lens": list(seq_lens),
            "mtp_runners": runners,
            "dsa_ctx": Glm52DsaForwardContext(
                spans=spans, k_store=self._dsa_k_store,
                needs_selection=needs_selection,
            ) if long_context else None,
        }

    def _mtp_pair_rows(self, normed: torch.Tensor, prenorm: torch.Tensor) -> torch.Tensor:
        """The trunk stream the MTP plane pairs drafts against: post-final-
        norm (vLLM's convention: p1/p2 0.89/0.74 vs pre-norm's 0.77/0.33
        on this checkpoint) unless MSTAR_GLM52_MTP_PAIR_POSTNORM=0."""
        return normed if self._mtp_pair_postnorm else prenorm

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        with_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.language_model.model(
            input_ids, position_ids, dsa_ctx=dsa_ctx, return_prenorm=with_prenorm)

    def _last_rows(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, kwargs: dict,
    ) -> torch.Tensor:
        """Each request's last row of a packed prefill: from the plan's
        qo_indptr buffer where the attention resource has one (captured, or
        the MLA resource), else from preprocess's real lengths."""
        qo_indptr = self._attn(engine_inputs).qo_indptr_buf(_MAIN)
        if qo_indptr is not None:
            return hidden.index_select(0, (qo_indptr[1:] - 1).long())
        last = kwargs.get("last_token_indices")
        assert last is not None, "eager prefill needs last_token_indices from preprocess"
        return hidden.index_select(0, last)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        self._stop_load_heartbeat()
        hidden = self._hidden(input_ids, position_ids, kwargs.get("dsa_ctx"))
        return {"logits": [self.lm_head(hidden[-1:])]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
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
        hidden = self._hidden(input_ids, position_ids, kwargs.get("dsa_ctx"))
        if graph_walk == "prefill":
            hidden = self._last_rows(engine_inputs, hidden, kwargs)
        elif graph_walk != "decode":
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")
        logits = self.lm_head(hidden)  # (bs, vocab)
        request_ids = list(engine_inputs.request_ids)
        new_tokens = engine_inputs.resources[SAMPLER_RESOURCE].sample(
            request_ids, logits=logits)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }

    # ── MTP: verify, rewind, sync, draft ──

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
        kv = self._kv(engine_inputs)
        seq_lens = kwargs.get("seq_lens")
        assert seq_lens is not None, "MTP step needs seq_lens from preprocess"
        runners = kwargs.get("mtp_runners") or {}
        request_ids = list(engine_inputs.request_ids)
        num = len(request_ids)
        row_starts = [0]
        for sl in seq_lens:
            row_starts.append(row_starts[-1] + sl)

        if graph_walk == "prefill":
            prefill_runner = runners.get("prefill")
            if prefill_runner is not None:
                # the runner plans the real lengths, replays and commits; the
                # sample, whole-prompt plane sync and draft chain stay eager
                replay = prefill_runner.run(
                    static_inputs={"input_ids": input_ids, "position_ids": position_ids},
                    request_ids=request_ids, seq_lens=list(seq_lens),
                )
                hidden = replay.get_view("hidden")
                prenorm = replay.get_view("prenorm")
            else:
                step, ctx = self._eager_step(engine_inputs, request_ids, list(seq_lens))
                hidden, prenorm = self._hidden(
                    input_ids, position_ids, kwargs.get("dsa_ctx"), with_prenorm=True)
                self._commit_eager(engine_inputs, step, ctx)
            last_hidden = self._last_rows(engine_inputs, hidden, kwargs)
            # the prefill token goes through the sampler, as flag-off does
            new_tokens = engine_inputs.resources[SAMPLER_RESOURCE].sample(
                request_ids, logits=self.lm_head(last_hidden))
            # MTP-plane sync over the whole prompt: entry for token t_p pairs
            # (embed(t_p), h_{p-1}); the last entry — the emitted token
            # paired with the prompt-final hidden — yields draft 1
            sync_tokens, pair_hiddens = [], []
            pair_rows = self._mtp_pair_rows(hidden, prenorm)
            for i, rid in enumerate(request_ids):
                r = slice(row_starts[i], row_starts[i + 1])
                self._mtp_emitted[rid] = 1
                sync_tokens.append(torch.cat([input_ids[r][1:], new_tokens[i:i + 1]]))
                pair_hiddens.append(pair_rows[r])
            drafts = self._mtp_sync_and_draft(
                engine_inputs, sync_tokens, pair_hiddens, draft_runner=runners.get("draft"))
            return {
                rid: {
                    "new_token": [new_tokens[i:i + 1]],
                    # consumed only when the prefill-drafts edge is declared
                    MTP_DRAFT_BUNDLE: [torch.cat([new_tokens[i:i + 1], drafts[i]])],
                }
                for i, rid in enumerate(request_ids)
            }

        if graph_walk != "decode":
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

        timer = self._mtp_timer
        timer.report(logger)
        timer.begin()
        trunk_runner = runners.get("trunk")
        if trunk_runner is not None:
            replay = trunk_runner.run(
                static_inputs={"input_ids": input_ids, "position_ids": position_ids},
                request_ids=request_ids, seq_lens=list(seq_lens),
            )
            # views into the static outputs: all consumed before the next replay
            hidden = replay.get_view("hidden")
            prenorm = replay.get_view("prenorm")
            logits = replay.get_view("logits")
        else:
            self._warn_mtp_trunk_eager_once(num, sum(seq_lens))
            step, ctx = self._eager_step(engine_inputs, request_ids, list(seq_lens))
            hidden, prenorm = self._hidden(
                input_ids, position_ids, kwargs.get("dsa_ctx"), with_prenorm=True)
            logits = self.lm_head(hidden)  # (sum(k+1), vocab)
            self._commit_eager(engine_inputs, step, ctx)
        # ONE device->host round trip for the whole verify: the target argmax
        # over all rows and the drafts travel together; everything after is
        # host arithmetic. Emitted tokens are a view of the target argmax
        # (greedy verify accepts draft j iff it equals target[j]).
        target_argmax_all = logits.argmax(dim=-1)  # (sum(k+1),)
        total_rows = row_starts[-1]
        timer.mark("trunk")
        phase_runner = runners.get("phase")
        rows = self.config.mtp_num_draft_tokens + 1
        prepared = False
        pair_rows = self._mtp_pair_rows(hidden, prenorm)
        if (
            phase_runner is not None and self._mtp_phase_prepare
            and all(sl == rows for sl in seq_lens)
        ):
            # the e-independent half of the draft phase, while the GPU is
            # still running the trunk (see _mtp_phase_prepare)
            pos_full: list[int] = []
            for rid, sl in zip(request_ids, seq_lens, strict=True):
                p0 = kv.stored_len(rid) - sl
                pos_full.extend(range(p0 + 1, p0 + 1 + sl))
            phase_runner.stage(
                static_inputs={
                    "sync_ids": target_argmax_all,
                    "pair_hidden": pair_rows,
                    "sync_position_ids": pinned(pos_full, torch.long),
                },
                request_ids=request_ids, seq_lens=list(seq_lens),
                step_kwargs={"phase": "prepare"},
            )
            prepared = True
            timer.mark("prepare")
        host = torch.cat([input_ids, target_argmax_all]).tolist()
        timer.mark("verify_d2h")
        inputs_h, target_h = host[:total_rows], host[total_rows:]
        eos_ids = self.config.eos_token_ids
        results: dict[str, NameToTensorList] = {}
        sync_tokens, pair_hiddens = [], []
        for i, rid in enumerate(request_ids):
            lo, hi = row_starts[i], row_starts[i + 1]
            m = seq_lens[i]
            n_acc = mtp_greedy_verify_host(inputs_h[lo + 1:hi], target_h[lo:hi])
            self._mtp_stat_steps += 1
            self._mtp_stat_emitted += n_acc + 1
            self._mtp_stat_acc_hist[n_acc] += 1
            # truncate: max_tokens budget first, then the first stop id; EOS
            # is always the LAST element, which check_stop relies on
            budget = self._mtp_max_tokens[rid] - self._mtp_emitted[rid]
            e = min(n_acc + 1, max(budget, 1))
            if not self._mtp_ignore_eos[rid]:
                for j in range(e):
                    if target_h[lo + j] in eos_ids:
                        e = j + 1
                        break
            emitted = target_argmax_all[lo:lo + e]
            self._mtp_emitted[rid] += e
            # m rows were committed; keep input[0] plus the e-1 accepted
            # drafts, drop the rest (the bonus row was never processed)
            kv.rewind(rid, m - e)
            sync_tokens.append(emitted)
            pair_hiddens.append(pair_rows[lo:lo + e])
            results[rid] = {"new_token": [emitted]}
        self._maybe_log_mtp_acceptance()
        timer.mark("verify_host")
        sync_runner = runners.get("sync")
        if sync_runner is None and self._mtp_capture_sync:
            self._warn_mtp_sync_eager_once(num)
        drafts = self._mtp_sync_and_draft(
            engine_inputs, sync_tokens, pair_hiddens,
            draft_runner=runners.get("draft"), sync_runner=sync_runner,
            phase_runner=phase_runner, prepared=prepared,
        )
        for i, rid in enumerate(request_ids):
            emitted = results[rid]["new_token"][0]
            results[rid]["text_inputs"] = [torch.cat([emitted[-1:], drafts[i]])]
        timer.mark("tail")
        timer.end()
        return results

    def _mtp_sync_and_draft(
        self,
        engine_inputs: ModelInputsFromEngine,
        sync_tokens: list[torch.Tensor],
        pair_hiddens: list[torch.Tensor],
        draft_runner=None,
        sync_runner=None,
        phase_runner=None,
        prepared: bool = False,
    ) -> list[torch.Tensor]:
        """Extend the MTP plane over the newly committed tokens, then draft
        k tokens autoregressively. Returns per-request (k,) draft tensors.
        """
        k = self.config.mtp_num_draft_tokens
        kv = self._kv(engine_inputs)
        mtp = self.language_model.mtp
        embed = self.language_model.model.embed_tokens
        request_ids = list(engine_inputs.request_ids)
        num = len(request_ids)
        device = pair_hiddens[0].device
        e_list = [t.shape[0] for t in sync_tokens]
        starts = [kv.stored_len(rid) for rid in request_ids]
        rows = k + 1

        assert not prepared or phase_runner is not None
        if phase_runner is not None:
            # ONE graph for sync + draft-1 head + k-1 chain iterations; the
            # region's step declares k sub-plans from e_list (P0 = start - e)
            assert all(e <= rows for e in e_list), (
                f"draft-phase graph got rows {e_list} outside [1, {rows}]")
            pos_l, last_l, _ = mtp_sync_padded_layout(e_list, starts, k)
            if prepared:
                # sync inputs and sub-plan 0 went in through stage() before
                # the readback; only the e-dependent leftovers travel here
                phase_inputs = {"last_rows": pinned(last_l, torch.long)}
                step_kwargs = {"e_list": list(e_list), "phase": "finish"}
            else:
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
                step_kwargs = {"e_list": list(e_list)}
            for it in range(1, k):
                phase_inputs[f"chain_pos_{it}"] = pinned([st + it for st in starts], torch.long)
            out = phase_runner.run(
                static_inputs=phase_inputs, request_ids=request_ids,
                seq_lens=[rows] * num, step_kwargs=step_kwargs,
            )
            self._mtp_timer.mark("draft_phase")
            drafts = out["drafts"]  # (num, k), owned
            return [drafts[i] for i in range(num)]

        # Sync pass (+ draft 1 from its last row): plane slots start-e ..
        for rid, e in zip(request_ids, e_list, strict=True):
            kv.rewind(rid, e)
        if sync_runner is not None:
            # PADDED replay: k+1 rows per request; only e are real, and the
            # runner commits rows, so rewind rows - e after
            assert all(e <= rows for e in e_list), (
                f"padded sync got rows {e_list} outside [1, {rows}]")
            pos_l, last_l, over_advance = mtp_sync_padded_layout(e_list, starts, k)
            sync_ids = torch.zeros(num * rows, dtype=torch.long, device=device)
            pair_h = torch.zeros(
                (num * rows, pair_hiddens[0].shape[-1]),
                dtype=pair_hiddens[0].dtype, device=device)
            for i, (t, h) in enumerate(zip(sync_tokens, pair_hiddens, strict=True)):
                sync_ids[i * rows:i * rows + t.shape[0]] = t
                pair_h[i * rows:i * rows + h.shape[0]] = h
            out = sync_runner.run(
                static_inputs={
                    "sync_ids": sync_ids,
                    "pair_hidden": pair_h,
                    "position_ids": pinned(pos_l, torch.long),
                },
                request_ids=request_ids, seq_lens=[rows] * num,
            )
            for rid, over in zip(request_ids, over_advance, strict=True):
                kv.rewind(rid, over)
            h_head, h_raw = out.get_view("h_head"), out.get_view("h_raw")
            last_rows = to_device_async(last_l, torch.long, device)
        else:
            pos_list: list[int] = []
            for st, e in zip(starts, e_list, strict=True):
                pos_list.extend(range(st - e + 1, st + 1))
            positions = to_device_async(pos_list, torch.long, device)
            step, ctx = self._eager_step(engine_inputs, request_ids, list(e_list))
            h_head, h_raw = mtp(embed(torch.cat(sync_tokens)), torch.cat(pair_hiddens), positions)
            self._commit_eager(engine_inputs, step, ctx)
            last_l_eager: list[int] = []
            acc = 0
            for e in e_list:
                acc += e
                last_l_eager.append(acc - 1)
            last_rows = to_device_async(last_l_eager, torch.long, device)

        self._mtp_timer.mark("sync")
        # the head reads the shared_head-normed rows; the chain threads the
        # raw layer output (hnorm re-norms it next iteration)
        prev_h = h_raw.index_select(0, last_rows)      # (B, hid)
        prev_d = self.lm_head(h_head.index_select(0, last_rows)).argmax(dim=-1)  # (B,)
        draft_cols = [prev_d]
        ones = [1] * num
        if k > 1 and draft_runner is None:
            self._warn_mtp_draft_eager_once(num)
        for it in range(1, k):
            pos_it = [st + it for st in starts]
            if draft_runner is not None:
                # the runner plans one row at the current length (RoPE at
                # start+it), replays and commits +1; the host stays one
                # replay ahead (pinned positions, host-side plan)
                out = draft_runner.run(
                    static_inputs={
                        "draft_ids": prev_d,
                        "prev_hidden": prev_h,
                        "position_ids": pinned(pos_it, torch.long),
                    },
                    request_ids=request_ids, seq_lens=ones,
                )
                # draft_ids must be owned (kept across later replays that
                # overwrite the static output); prev_hidden is copied into
                # the next replay's input, so a view is enough
                prev_d = out["draft_ids"]
                prev_h = out.get_view("prev_hidden")
            else:
                positions = to_device_async(pos_it, torch.long, device)
                step, ctx = self._eager_step(engine_inputs, request_ids, ones)
                it_head, prev_h = mtp(embed(prev_d), prev_h, positions)
                self._commit_eager(engine_inputs, step, ctx)
                prev_d = self.lm_head(it_head).argmax(dim=-1)
            draft_cols.append(prev_d)
            self._mtp_timer.mark(f"chain{it}")
        if k > 1:
            for rid in request_ids:
                kv.rewind(rid, k - 1)
        stacked = torch.stack(draft_cols, dim=1)       # (B, k)
        return [stacked[i] for i in range(num)]

    def _warn_mtp_trunk_eager_once(self, bs: int, num_rows: int) -> None:
        if self._mtp_trunk_eager_warned:
            return
        self._mtp_trunk_eager_warned = True
        logger.warning(
            "MTP decode trunk running EAGER (bs=%d, %d rows): no piecewise CUDA "
            "graph bucket — capture failed at warmup, the MoE resolved to an "
            "uncapturable dispatch, or bs exceeds the captured sizes %s.",
            bs, num_rows, self.MTP_CAPTURE_BATCH_SIZES,
        )

    def _warn_mtp_draft_eager_once(self, bs: int) -> None:
        if self._mtp_draft_eager_warned:
            return
        self._mtp_draft_eager_warned = True
        logger.warning(
            "MTP draft chain running EAGER (bs=%d): no mtp_draft piecewise bucket; "
            "each chain iteration pays eager launch overhead.", bs,
        )

    def _warn_mtp_sync_eager_once(self, bs: int) -> None:
        if self._mtp_sync_eager_warned:
            return
        self._mtp_sync_eager_warned = True
        logger.warning(
            "MTP decode sync pass running EAGER (bs=%d): no mtp_sync piecewise "
            "bucket; ~10 ms of avoidable launch overhead per decode step.", bs,
        )

    _MTP_STAT_LOG_EVERY = 512  # request-steps between acceptance log lines

    def _maybe_log_mtp_acceptance(self) -> None:
        if self._mtp_stat_steps - self._mtp_stat_logged < self._MTP_STAT_LOG_EVERY:
            return
        self._mtp_stat_logged = self._mtp_stat_steps
        k = self.config.mtp_num_draft_tokens
        mean_emitted = self._mtp_stat_emitted / self._mtp_stat_steps
        logger.info(
            "MTP acceptance: %.2f emitted/step (ceiling %d, plain decode would be "
            "1.00) — draft acceptance rate %.2f over %d request-steps.",
            mean_emitted, k + 1, (mean_emitted - 1.0) / k if k else 0.0,
            self._mtp_stat_steps,
        )
        # conditional per-position profile: p_i = P(draft i accepted |
        # drafts 1..i-1 accepted); a falling profile means the chain degrades
        reached = [sum(self._mtp_stat_acc_hist[i:]) for i in range(k + 1)]
        cond = [
            f"{reached[i] / reached[i - 1]:.2f}" if reached[i - 1] else "-"
            for i in range(1, k + 1)
        ]
        logger.info(
            "MTP acceptance by position: n_acc histogram %s, conditional accept "
            "per position %s [trunk pairing: %s]",
            self._mtp_stat_acc_hist, " ".join(cond),
            "POST-final-norm (vLLM convention)" if self._mtp_pair_postnorm
            else "pre-final-norm (default)",
        )

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        if "text_inputs" in outputs:
            # MTP step: the forward assembled the loop-back input
            # ([last emitted, k drafts]); new_token is the verified emission
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        ignore_eos = request_info.resource_configs[SAMPLER_RESOURCE].ignore_eos
        if self.config.mtp_num_draft_tokens > 0:
            # multi-token emission: in-step truncation guarantees a stop id
            # can only be the LAST element; totals live in the per-request
            # counter (loop iters no longer count tokens)
            tokens = outputs["new_token"][0]
            last = int(tokens[-1])
            is_eos = last in self.config.eos_token_ids
            generated = self._mtp_emitted.get(request_id, 0)
            if (not ignore_eos and is_eos) or generated >= request_info.max_tokens:
                return {"decode_loop"}
            return set()
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        # total generated = 1 prefill-emitted token + (iters + 1) decode
        # tokens (vLLM's max_tokens semantics)
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
