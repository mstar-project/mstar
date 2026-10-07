"""AR submodule for the GLM-5.3-Flash text backbone on the resource-pool engine."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
)
from mstar.engine.cuda_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.attn import sparse_mla
from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.linear_attn.kda import SpecBlocks
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.attention import Glm5NextMLAAttention
from mstar.model.glm5_next.config import (
    ATTN,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    MTP_DRAFT,
    MTP_HIDDEN,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.glm5_next.dsa import DsaState, Glm5NextDsaContext, attn_lens
from mstar.model.glm5_next.kda import TorchKDAKernels
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.pinned_staging import to_device_async

logger = logging.getLogger(__name__)

# Captured prefill shapes when the config names none (prefill_graphs on). Buckets start
# above the MoE decode path's 64 tokens: the fused runner routes padding rows safely.
# 128-token steps to 1024, then 64 up to one request's max, so a long prompt replays at most
# 63 padding tokens; then 256 to 4096, so a step of several prompts pads at most 255.
PREFILL_TOKEN_BUCKETS = (*range(128, 1024, 128), *range(1024, 2049, 64), *range(2304, 4097, 256))
PREFILL_CAPTURE_BATCH_SIZES = (1, 2, 4)

# The rope slot of a rejected draft row: against a query rope part of ones, a -6.4e5 score term.
HOLE_KPE = -1.0e4
_VERDICT_TIMEOUT_S = 120.0


def _fused_kda(config: Glm5NextModelConfig) -> bool:
    """The KDA layers run the fused kernels, whose prefill is the only capturable one."""
    return (fused_decode._ENABLED and kda_triton._HAS_TRITON
            and kda_triton.supported(config.linear_head_dim))


def _real_row_values(last: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """``values`` with each padding row's entry replaced by that of the row it repeats.

    A padding row repeats the previous row's last index (``_last_rows``), so a scatter at
    ``last`` writes two values to one position; this makes them the same one. Device ops
    only, so a captured replay runs it too.
    """
    n = last.shape[0]
    starts = torch.ones(n, dtype=torch.bool, device=last.device)
    starts[1:] = last[1:] != last[:-1]
    rows = torch.where(starts, torch.arange(n, device=last.device), 0)
    return values.index_select(0, torch.cummax(rows, 0).values)


class Glm5NextLLMSubmodule(ARNodeSubmodule):
    """Embed + 45 hybrid decoder layers + lm_head, one fat TP node, and with MTP the
    layer-45 draft module."""

    # The eager prefill forward hosts the KDA resource (@torch.compiler.disable
    # inside a compiled outer frame) and the fp8 reference MoE loop; the
    # post-capture torch.compile of the eager forwards is untested for it.
    disable_torch_compile: bool = True
    # Weights load in bf16; the KDA conv and delta rule, the router and mHC run
    # in fp32 on purpose, which the engine's bf16 autocast would undo.
    disable_autocast: bool = True

    def __init__(self, language_model: nn.Module, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.language_model = language_model  # Glm5NextForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        # MTP: a decode step verifies the last token and k drafts per request
        self.mtp_k = config.mtp_num_draft_tokens
        self._mtp_seq = 0
        # accepted-draft counts since the last log line: a histogram over 0..k
        self._mtp_hist = [0] * (self.mtp_k + 1)
        self._mtp_logged = time.monotonic()
        # capture slot (None: eager) -> the pinned verdict table and step number the host reads
        self._mailboxes: dict[int | None, tuple[torch.Tensor, torch.Tensor]] = {}
        # dsa_long_context: the forward's DSA context, which the MLA layers read, and under
        # CUDA graphs one sparse-attention plan per (capture slot, rows)
        self._dsa = DsaState()
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        # per (walk, capture slot): the workspace and index buffer its plans share
        self._sparse_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if config.dsa_long_context:
            for module in language_model.modules():
                if isinstance(module, Glm5NextMLAAttention):
                    module.dsa_state = self._dsa

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    def bind_node_resources(self, resources: dict) -> None:
        if KV_CACHE in resources:
            self.config.check_long_context(resources[KV_CACHE].kv_cache.page_size)
        super().bind_node_resources(resources)
        # The fused kernels where they run; the torch reference elsewhere.
        kda = resources.get(KDA)
        if kda is not None and (kda.kernels is None or not _fused_kda(self.config)):
            kda.set_kernels(TorchKDAKernels())

    # -- CUDA-graph configs ----------------------------------------------

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the clamped fused fp8
        kernel (``moe_quant_kernel`` "triton"/"auto" on CUDA); False on the
        "reference" default, whose per-hit-expert loop hosts ``.nonzero()``
        and cannot be captured.
        """
        from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock

        lm = getattr(self, "language_model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm5NextSparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        return fp8_reference or naive_tp

    def _kda_max_slots(self) -> int | None:
        """Requests the KDA pool can hold at once: its slots but the sink."""
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        return None if pool is None else pool.config.usable_slots

    def max_batch_size(self, graph_walk: str) -> int | None:
        # A step can hold at most one slot per row of real state; the
        # scheduler splits anything larger rather than admit-failing it.
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        if pool is None:
            return None
        # A prefill takes only the slots free now, and waits while none are, so a
        # TP follower refusing one has diverged from rank 0 (the engine raises).
        # With prefill graphs, no more rows than one takes: an eager prefill's
        # last layer compiles the MoE decode kernels for its row count mid-serve.
        if graph_walk == "prefill":
            rows = self._captured_prefill_rows()
            free = pool.num_free_slots
            return free if rows is None else min(free, rows)
        return pool.config.usable_slots

    def _captured_prefill_rows(self) -> int | None:
        """The most rows a captured prefill takes, or None while prefill runs eager."""
        if not self.config.prefill_graphs or not _fused_kda(self.config):
            return None
        # declared no prefill graph (reference MoE, the MLA fallback): an eager prefill
        # held to 4 rows took 16 steps for a burst of 64 prompts
        if getattr(self, "_prefill_captured", True) is False:
            return None
        return max(self.config.prefill_capture_batch_sizes or PREFILL_CAPTURE_BATCH_SIZES)

    def max_step_tokens(self, graph_walk: str) -> int | None:
        cap = self.config.prefill_max_step_tokens
        if graph_walk != "prefill" or cap is None:
            return None
        if cap == "auto":
            return max(self.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS)
        return cap

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        if self._moe_capture_blocked(tp_world_size):
            logger.warning(
                "glm5_next: reference MoE dispatch active; decode runs eager "
                "(set moe_quant_kernel=auto for the capturable fused kernel)"
            )
            self._prefill_captured = False
            return []
        # The MLA fallback bakes page indices into its plan, so a captured
        # replay would attend the capture's pages (the resource refuses).
        attn = self.node_resources.get(ATTN)
        if device.type == "cuda" and attn is not None and not attn.uses_kernel:
            logger.warning(
                "glm5_next: MLA is on the SDPA fallback, which cannot be "
                "captured; decode runs eager"
            )
            self._prefill_captured = False
            return []
        max_slots = self._kda_max_slots()
        batch_sizes = [
            b for b in DEFAULT_CAPTURE_BATCH_SIZES
            if max_slots is None or b <= max_slots
        ]
        # Graphs capture the eager forward: a Triton crash in Inductor's compile
        # subprocess would fail every capture and fall back to eager silently.
        configs: list[CudaGraphConfig] = [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=self.mtp_block,
                ),
                capture_batch_sizes=batch_sizes,
                compile=False,
            ),
        ]
        if self.config.prefill_graphs and not _fused_kda(self.config):
            # The reference KDA prefill loops over spans on the host, so a
            # captured replay would run the capture's dummy spans.
            logger.warning("glm5_next: fused KDA prefill off; prefill runs eager")
        elif self.config.prefill_graphs:
            # One graph per (rows, token bucket); a batch past every bucket
            # runs eager rather than being split.
            configs.append(PackedCudaGraphConfig(
                capture_graph_walk="prefill",
                capture_token_lengths=list(
                    self.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS),
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                ),
                capture_batch_sizes=[
                    b for b in (self.config.prefill_capture_batch_sizes
                                or PREFILL_CAPTURE_BATCH_SIZES)
                    if max_slots is None or b <= max_slots
                ],
                caps_eager_batch_size=False,
                compile=False,
            ))
        self._prefill_captured = any(c.capture_graph_walk == "prefill" for c in configs)
        return configs

    # -- dtype discipline -------------------------------------------------

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm5NextLLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    # -- engine seam: prepare -> declare -> preprocess -> forward ---------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        if self.mtp_k > 0:
            self._check_mtp_sampling(fwd_info)
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                seq_len = self.mtp_block
        # Without dsa_long_context the serving regime is ctx <= index_topk,
        # where dense MLA computes exactly what GLM-5.3-Flash's DSA would —
        # refuse beyond it rather than serve off-spec logits. Raising here
        # fails only this request (the engine registers the failure and
        # drops the rid from the batch).
        limit = self.config.context_limit
        # the handle, which the batch, check_stop and cleanup_request key by
        state = self.request_state(fwd_info.rid_handle)
        committed = state.get("context", 0)
        real = committed
        if self.mtp_k > 0 and graph_walk == "decode":
            # holes are cache rows, not context: the model sees the prompt and the
            # tokens emitted so far, the last of which this step takes in
            real = state.get("prompt_len", 0) + state.get("mtp_generated", 1) - 1
        if real + seq_len > limit or committed + seq_len > self.config.kv_rows:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {real + seq_len} "
                f"({committed + seq_len} cache rows) exceeds "
                f"{'max_seq_len' if self.config.dsa_long_context else 'index_topk'}={limit} "
                f"or {self.config.kv_rows} rows; without dsa_long_context the limit is "
                "index_topk, where dense MLA is exactly GLM-5.3-Flash's DSA."
            )
        if graph_walk == "prefill":
            state.add("prompt_len", committed + seq_len)
            if self.mtp_k > 0:
                state.add_all(mtp_generated=1, mtp_pending=None)
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len)

    def _check_mtp_sampling(self, fwd_info: CurrentForwardPassInfo) -> None:
        """MTP verifies by argmax: refuse what it would silently ignore."""
        cfg = fwd_info.resource_configs.get(SAMPLER) if fwd_info.resource_configs else None
        if cfg is None:
            return
        # not `or`: an explicit repetition_penalty of 0 must not read as unset
        temperature = 0.0 if cfg.temperature is None else cfg.temperature
        penalty = 1.0 if cfg.repetition_penalty is None else cfg.repetition_penalty
        if temperature > 0.0 or penalty != 1.0:
            raise RuntimeError(
                f"request {fwd_info.request_id}: MTP drafting is greedy only "
                f"(temperature={cfg.temperature}, "
                f"repetition_penalty={cfg.repetition_penalty}); serve it with "
                "mtp_num_draft_tokens: 0"
            )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        prefill_tokens: dict[str, torch.Tensor] = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            }
        steps = {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(
                apply_penalty=True, prefill_tracked_tokens=prefill_tokens,
            ),
            KDA_STATE: RecurrentStep(),
            KDA: LinearAttnStep(),
        }
        if self.mtp_k > 0 and graph_walk == "decode":
            # the verify and the drafts are argmaxes in the forward; the KDA layers first
            # replay what the last step accepted
            del steps[SAMPLER]
            steps[KDA] = LinearAttnStep(speculative=True)
        if self.config.dsa_long_context:
            # every MLA layer attends sparse (dsa.py): no dense plan over the whole context
            del steps[ATTN]
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # The per-replay-varying inputs. Everything else the forward reads
        # is a resource plan (slot index, KV write slots, planned kernels).
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        if graph_walk == "prefill" and engine_inputs.captured:
            out["last_rows"] = self._last_rows(inputs)
        if self.mtp_k > 0 and graph_walk == "prefill":
            out["mtp_keep"] = self._mtp_keep(engine_inputs, inputs, out["input_ids"].shape[0])
        if self.mtp_k > 0 and graph_walk == "decode":
            out["mtp_seq"] = self._mtp_stage_seq(engine_inputs)
        if self.config.dsa_long_context:
            out.update(self._dsa_inputs(engine_inputs))
        return out

    def _dsa_inputs(self, engine_inputs: ModelInputsFromEngine) -> dict[str, Any]:
        """The step's DSA state from the KV plan, as preprocess outputs. The tensors are top
        level, so a CUDA graph interns them and each replay stages the step's values: per row
        its position and request, per request its first entry in the flat page list. Under a
        graph the score width spans the serving window and the sparse plan is the slot's own,
        re-planned here every step; the host half rides in ``dsa_meta``."""
        step_ctx = engine_inputs.step.ctx
        graph = bool(engine_inputs.captured)
        walk = step_ctx.graph_walk
        kv_out = step_ctx.plan_results[KV_CACHE][LABEL]
        page_size = engine_inputs.resources[KV_CACHE].kv_cache.page_size
        cfg = self.config
        pos: list[int] = []
        row_req: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, view in enumerate(kv_out.views):
            spans.append((len(pos), view.to_compute, i))
            pos.extend(range(view.length - view.to_compute, view.length))
            row_req.extend([i] * view.to_compute)
        n_real = len(pos)
        rows = n_real
        if graph and step_ctx.slot_lease is not None:
            # a packed prefill bucket's padding rows: position 0 of the first request. Staged, so
            # they don't keep what a larger bucket left in the shared static buffers
            rows = step_ctx.slot_lease.bucket.num_tokens
            pos += [0] * (rows - n_real)
            row_req += [0] * (rows - n_real)
        indptr = kv_out.cpu_indptrs.paged_kv_indptr
        parts = [torch.tensor(pos + row_req, dtype=torch.int32), indptr[:-1].to(torch.int32),
                 kv_out.cpu_indptrs.paged_kv_indices.to(torch.int32)]
        if step_ctx.capture:
            # the static buffer's size: every row's request at the serving window (a replay
            # stages just its own pages)
            pad = len(kv_out.views) * -(-cfg.kv_rows // page_size) - parts[-1].numel()
            parts.append(torch.zeros(pad, dtype=torch.int32))
        device = self.lm_head.weight.device
        n_pos, n_req = len(pos), indptr.numel() - 1
        staged = to_device_async(torch.cat(parts), torch.int32, device)
        lens = attn_lens(pos, cfg.index_topk, cfg.index_kpool)  # a padding row's is 1
        # score width: the serving window's pools for a decode graph; a captured prefill holds
        # whole prompts no longer than its bucket
        max_pools = (max(pos) + 1) // cfg.index_kpool if pos else 1
        sparse_plan = None
        if graph:
            if walk == "prefill":
                if pos and max(pos) >= rows:
                    raise RuntimeError(f"a captured prefill row sits at {max(pos)}, past its "
                                       f"{rows}-token bucket")
                max_pools = rows // cfg.index_kpool
            else:
                max_pools = cfg.kv_rows // cfg.index_kpool
        if graph and device.type == "cuda":
            width = cfg.index_topk + cfg.index_kpool - 1
            key = (walk, step_ctx.slot, rows)
            sparse_plan = self._sparse_plans.get(key)
            if sparse_plan is None:
                buffers = self._sparse_buffers.get(key[:2])
                if buffers is None or buffers[1].numel() < rows * width:
                    # captures run largest first, so the first is the slot's size
                    buffers = self._sparse_buffers[key[:2]] = (
                        torch.empty(sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8, device=device),
                        torch.zeros(rows * width, dtype=torch.int32, device=device))
                sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                    rows, width, buffers[0], buffers[1])
            attn = next(m for m in self.language_model.modules()
                        if isinstance(m, Glm5NextMLAAttention))
            sparse_plan.plan(lens, attn.num_heads, cfg.kv_lora_rank, cfg.mla_cache_kpe,
                             attn.softmax_scale)
        return {
            "dsa_pos": staged[:n_pos],
            "dsa_row_req": staged[n_pos:2 * n_pos],
            "dsa_page_start": staged[2 * n_pos:2 * n_pos + n_req],
            "dsa_pages": staged[2 * n_pos + n_req:],
            "dsa_meta": dict(
                host_pos=None if graph else pos[:n_real],
                spans=None if graph else spans,
                host_page_start=None if graph else indptr[:-1].tolist(),
                max_pools=max(max_pools, 1),
                page_size=page_size, sparse_plan=sparse_plan,
            ),
        }

    def _dsa_context(self, kwargs: dict[str, Any]) -> Glm5NextDsaContext | None:
        meta = kwargs.get("dsa_meta")
        if meta is None:
            return None
        return Glm5NextDsaContext(
            pos=kwargs["dsa_pos"], row_req=kwargs["dsa_row_req"], pages=kwargs["dsa_pages"],
            page_start=kwargs["dsa_page_start"], host_pos=meta["host_pos"],
            spans=meta["spans"], host_page_start=meta["host_page_start"],
            max_pools=meta["max_pools"], page_size=meta["page_size"],
            topk=self.config.index_topk, kpool=self.config.index_kpool,
            sparse_plan=meta["sparse_plan"])

    def _mtp_keep(
        self, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs], num_tokens: int,
    ) -> torch.Tensor:
        """1 per prefill token, 0 where a request's token sits at position 0: the MTP pass
        zeroes its embedding there, as vLLM does."""
        keep, start = [1.0] * num_tokens, 0
        for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True):
            state = self.request_states.get(rid)
            fresh = state is None or state.get("context", 0) == 0
            if fresh and inp.input_seq_len > 0 and start < num_tokens:
                keep[start] = 0.0
            start += inp.input_seq_len
        return to_device_async(keep, self.lm_head.weight.dtype, self.lm_head.weight.device)

    def _mailbox_key(self, engine_inputs: ModelInputsFromEngine) -> int | None:
        """A captured step's capture slot, None eager: each has its own mailbox, which a
        graph writes at a fixed address."""
        lease = engine_inputs.step.ctx.slot_lease if engine_inputs.step is not None else None
        if not engine_inputs.captured or lease is None or lease.bucket is None:
            return None
        return lease.slot

    def _mtp_stage_seq(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        """Number this verify step and note where each real request's verdict lands."""
        key = self._mailbox_key(engine_inputs)
        if key not in self._mailboxes:
            pin = self.lm_head.weight.device.type == "cuda"
            rows = engine_inputs.resources[KDA_STATE].config.max_slots
            self._mailboxes[key] = (
                torch.zeros(rows, self.mtp_block + 1, dtype=torch.long, pin_memory=pin),
                torch.full((1,), -1, dtype=torch.long, pin_memory=pin),
            )
        self._mtp_seq += 1
        seq = self._mtp_seq
        ctx = engine_inputs.step.ctx if engine_inputs.step is not None else None
        if ctx is None or not ctx.capture:
            # the real rows only: a padded replay's dummy rows have states too
            real = set(ctx.request_ids if ctx is not None else engine_inputs.per_request_info)
            for row, rid in enumerate(engine_inputs.request_ids):
                if rid in real:
                    self.request_state(rid).add("mtp_pending", (seq, key, row))
        return to_device_async([seq], torch.long, self.lm_head.weight.device)

    def _last_rows(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        """Each row's last token, which a captured prefill reads at a fixed
        address: the rows it samples."""
        last, end = [], 0
        for inp in inputs:
            end += inp.input_seq_len
            last.append(max(end - 1, 0))  # a padding row repeats the one before
        return to_device_async(last, torch.int32, self.lm_head.weight.device)

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        dsa_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self._dsa.ctx = self._dsa_context(dsa_inputs or {})
        try:
            return self._forward_inner(graph_walk, engine_inputs, input_ids, last_rows, mtp_keep)
        finally:
            self._dsa.ctx = None

    def _forward_inner(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None,
        mtp_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        attn = engine_inputs.resources[ATTN]
        sampler = engine_inputs.resources[SAMPLER]
        if self.config.dsa_long_context and graph_walk == "prefill" and not engine_inputs.captured:
            return self._prefill_windows(engine_inputs, input_ids, sampler)
        last = None  # prefill samples each row's last token: the model computes only those
        if last_rows is not None:
            last = last_rows
        elif graph_walk == "prefill":
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
            last = attn.select_last_hidden(positions, LABEL)
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        if self.mtp_k > 0 and graph_walk == "prefill":
            # the MTP pass reads the trunk's hidden state at every prompt row
            hidden = self.language_model.model(input_ids)
            last = last.long()
            new_tokens = sampler.sample(
                engine_inputs.request_ids, logits=self.lm_head(hidden.index_select(0, last)))
            self._mtp_prefill(engine_inputs, input_ids, hidden, last, new_tokens, mtp_keep)
            return new_tokens
        hidden = self.language_model.model(input_ids, rows=last)
        logits = self.lm_head(hidden)  # (rows, vocab)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def _prefill_windows(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, sampler,
    ) -> torch.Tensor:
        """dsa_long_context's prefill: the step's tokens in windows of prefill_window_tokens,
        each through every layer before the next (its KV rows and KDA state land first), so
        activations stay bounded however long the prompt. Samples each request's last row."""
        ctx = self._dsa.ctx
        kv, kda = engine_inputs.resources[KV_CACHE], engine_inputs.resources[KDA]
        size = self.config.prefill_window_tokens
        total = input_ids.shape[0]
        last = [r0 + n - 1 for r0, n, _ in ctx.spans]
        if total <= size:  # one window: the step's own plans (one-token prompts plan as decode)
            rows = to_device_async(last, torch.long, input_ids.device)
            hidden = self.language_model.model(input_ids, rows=rows)
            return sampler.sample(engine_inputs.request_ids, logits=self.lm_head(hidden))
        hidden: list[torch.Tensor | None] = [None] * len(last)
        try:
            for start in range(0, total, size):
                end = min(start + size, total)
                pieces = [(max(r0, start), min(r0 + n, end), req)
                          for r0, n, req in ctx.spans if r0 < end and r0 + n > start]
                ends = [(i, r - start) for i, r in enumerate(last) if start <= r < end]
                rows = to_device_async([r for _, r in ends] or [end - start - 1], torch.long,
                                       input_ids.device)
                self._dsa.ctx = ctx.window(start, end, pieces)
                with kv.token_window(start, end, LABEL), kda.token_window(
                        [(req, b - a) for a, b, req in pieces], LABEL):
                    out = self.language_model.model(input_ids[start:end], rows=rows)
                for k, (i, _) in enumerate(ends):
                    hidden[i] = out[k]
        finally:
            self._dsa.ctx = ctx
        return sampler.sample(engine_inputs.request_ids, logits=self.lm_head(torch.stack(hidden)))

    # -- MTP device side ---------------------------------------------------

    def _broadcast(self, t: torch.Tensor) -> torch.Tensor:
        """Rank 0's copy: every rank must verify the same ids and keep the same tail."""
        return self.lm_head.comm_group.broadcast(t, src=0)

    def _argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """The LM head's argmax without gathering the logits: each rank's best over its
        contiguous vocab shard, then the best of those. A tie goes to the lower rank, so the
        lower id, as with a full argmax."""
        head = self.lm_head
        local = F.linear(hidden, head.weight)
        idx = local.argmax(-1)
        if head.tp_size == 1:
            return idx
        val = local.gather(-1, idx[:, None]).float()
        idx = idx[:, None] + head.tp_rank * head.output_size_per_partition
        vals = head.comm_group.all_gather(val, dim=-1)
        ids = head.comm_group.all_gather(idx, dim=-1)
        return ids.gather(-1, vals.argmax(-1, keepdim=True)).squeeze(-1)

    def _draft_tokens(self, hidden: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP state ``hidden``."""
        del prev  # the tests' oracle drafts read it
        return self._argmax(hidden)

    def _mtp_seed(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, prev: torch.Tensor,
    ) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev`` (the step's next input)."""
        pool = engine_inputs.resources[KDA_STATE]
        slots = engine_inputs.resources[KDA].current_plan().slot_ids.long()
        seed = pool.block(MTP_HIDDEN)
        seed.index_copy_(0, slots, hidden.to(seed.dtype))
        pool.block(MTP_DRAFT).index_copy_(0, slots, self._draft_tokens(hidden, prev)[:, None])

    def _mtp_prefill(self, engine_inputs, input_ids, hidden, last, new_tokens, mtp_keep) -> None:
        """The MTP pass over the prompt: row i pairs the trunk's state at token i with the
        embedding of token i + 1 (each request's last row: its sampled token). It fills the
        MTP layer's KV plane and drafts d1 from each request's last row."""
        lm = self.language_model
        # roll, not a shifted copy into empty memory: a captured bucket's last slot is
        # padding, and every slot must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, _real_row_values(last, new_tokens).to(nxt.dtype))
        embeds = lm.model.embed_tokens(nxt) * mtp_keep[:, None]
        self._mtp_seed(engine_inputs, lm.mtp(embeds, hidden).index_select(0, last), new_tokens)

    def _mtp_decode(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, mtp_seq: torch.Tensor,
    ) -> torch.Tensor:
        """One verify step: draft k tokens, verify ``[last | drafts]`` with the trunk, then run
        the MTP layer over the block to seed the next step. The verdict (the target's argmax
        per block row, then the accepted count) goes to the host's mailbox; returns the next
        input token per row."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        pool, kda = engine_inputs.resources[KDA_STATE], engine_inputs.resources[KDA]
        slots = kda.current_plan().slot_ids.long()
        n = slots.shape[0]

        # Drafts: d1 was seeded by the last step; each further one is an MTP pass whose
        # query sits at its row of this step's block.
        h = pool.block(MTP_HIDDEN).index_select(0, slots)
        drafts = [pool.block(MTP_DRAFT).index_select(0, slots)[:, 0]]
        attn = lm.mtp.transformer_layer.self_attn
        latents = h.new_zeros(n, block, attn.kv_lora_rank + attn.mla_cache_kpe)
        for row in range(k - 1):
            h = lm.mtp(lm.model.embed_tokens(drafts[-1]), h, row=row, latents=latents)
            drafts.append(self._draft_tokens(h, drafts[-1]))
        drafts = self._broadcast(torch.stack(drafts, 1))

        # Verify: greedy, the longest prefix of drafts the target agrees with.
        ids = torch.cat([input_ids.view(n, 1), drafts], 1).view(-1)
        hidden = lm.model(ids)
        target = self._argmax(hidden).view(n, block)
        accepted = (drafts == target[:, :k]).long().cumprod(1).sum(1)
        verdict = self._broadcast(torch.cat([target, accepted[:, None]], 1))
        target, accepted = verdict[:, :block], verdict[:, block]
        kda.set_prefix_len(SpecBlocks.of(pool, 0), accepted)

        # The host's copy: the verdict first, then its step number.
        verdicts, seq = self._mailboxes[self._mailbox_key(engine_inputs)]
        verdicts[:n].copy_(verdict, non_blocking=True)
        seq.copy_(mtp_seq, non_blocking=True)

        # Seed the next step: the MTP pass over the block, with the target's argmax as each
        # row's next token (up to the accepted row it equals the drafts), read at the last
        # accepted row.
        rows = torch.arange(n, device=slots.device)
        next_input = target[rows, accepted]
        h_block = lm.mtp(lm.model.embed_tokens(target.reshape(-1)), hidden)
        self._mtp_seed(engine_inputs, h_block.view(n, block, -1)[rows, accepted], next_input)

        keep = torch.arange(block, device=slots.device)[None, :] <= accepted[:, None]
        self._mask_holes(engine_inputs, keep)
        return next_input

    def _mask_holes(self, engine_inputs: ModelInputsFromEngine, keep: torch.Tensor) -> None:
        """The block's rejected rows stay in the cache, so the next step can be planned
        before this one's verdict exists: a very negative rope slot in every MLA plane takes
        them out of every later softmax (queries carry a rope part of ones)."""
        kv = engine_inputs.resources[KV_CACHE]
        pages, offsets = kv.write_slots(LABEL)
        rope = torch.where(keep.reshape(-1, 1), 0.0, HOLE_KPE)
        width = self.config.kv_lora_rank
        for plane in range(len(self.config.full_attn_layer_indices) + 1):  # the MTP plane last
            cache = kv.layer_view(plane)
            cache[pages, offsets, width:] = rope.to(cache.dtype).expand(-1, cache.shape[-1] - width)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        mtp_seq: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if mtp_seq is not None:
            return self.forward_batched(graph_walk, engine_inputs, input_ids,
                                        mtp_seq=mtp_seq)[engine_inputs.request_ids[0]]
        return {"new_token": self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, kwargs)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        mtp_keep: torch.Tensor | None = None,
        mtp_seq: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList] | BatchedModelOutput:
        if mtp_seq is not None:
            # the emitted tokens are cut on the host (unpack_packed_outputs); the next input
            # is the token after the accepted drafts
            next_input = self._mtp_decode(engine_inputs, input_ids, mtp_seq)
            return {
                rid: {"text_inputs": [next_input[i : i + 1]]}
                for i, rid in enumerate(engine_inputs.request_ids)
            }
        new_tokens = self._forward(
            graph_walk, engine_inputs, input_ids, last_rows, mtp_keep, kwargs)
        # row i is request i: one clone out of the graph and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    # -- MTP host side: the verdict mailbox ---------------------------------

    def _log_acceptance(self, accepted: int, every_s: float = 10.0) -> None:
        """Rank 0 logs the accepted-draft histogram at most every ``every_s`` seconds."""
        if self.lm_head.tp_rank != 0:
            return
        self._mtp_hist[accepted] += 1
        now = time.monotonic()
        if now - self._mtp_logged >= every_s:
            self._mtp_logged = now
            steps = sum(self._mtp_hist)
            mean = sum(i * c for i, c in enumerate(self._mtp_hist)) / steps
            logger.info("glm5_next MTP k=%d: %.2f drafts accepted per step (%.2f tokens), "
                        "histogram %s over %d request-steps",
                        self.mtp_k, mean, mean + 1, self._mtp_hist, steps)
            self._mtp_hist = [0] * (self.mtp_k + 1)

    def _read_verdicts(self, request_ids: list[str]) -> list[tuple[list[int], int]]:
        """One verify step's ``(tokens, accepted)`` per request, waiting once for the device
        to stage them (a step's requests share its number and mailbox)."""
        states = [self.request_state(rid) for rid in request_ids]
        pending = [st.get("mtp_pending") for st in states]
        seq, key, _ = pending[0]
        assert all(p[:2] == (seq, key) for p in pending), pending
        # read once, then kept: a later step may reuse the mailbox before a second read
        cached = [st.get("mtp_verdict") for st in states]
        if all(c is not None and c[0] == seq for c in cached):
            return [c[1:] for c in cached]
        verdicts, staged_seq = self._mailboxes[key]
        deadline = time.monotonic() + _VERDICT_TIMEOUT_S
        while True:
            staged = int(staged_seq[0])
            if staged == seq:
                break
            if staged > seq:
                raise RuntimeError(
                    f"MTP verdict of step {seq} was overwritten by step {staged} before it "
                    "was read")
            if time.monotonic() > deadline:
                raise RuntimeError(f"MTP verdict of step {seq} never arrived")
            time.sleep(0)
        table = verdicts[:max(p[2] for p in pending) + 1].tolist()
        k1 = self.mtp_block
        out = [(table[p[2]][:k1], table[p[2]][k1]) for p in pending]
        for st, verdict in zip(states, out, strict=True):
            st.add("mtp_verdict", (seq, *verdict))
        return out

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[NodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, NameToTensorList]:
        """A verify step's emitted tokens per request: the accepted drafts and the token
        after them, cut at a stop token and at max_tokens. Waits for the device to stage the
        verdict (the MTP pass that seeds the next step is still running)."""
        if self.mtp_k == 0 or not request_ids:
            return {}
        if per_request_info[request_ids[0]].graph_walk != "decode":
            return {}
        out = {}
        for rid, (target, accepted) in zip(request_ids, self._read_verdicts(request_ids),
                                           strict=True):
            info = per_request_info[rid]
            state = self.request_state(rid)
            generated = state.get("mtp_generated", 1)
            emitted = target[:accepted + 1][:max(info.max_tokens - generated, 1)]
            if not info.resource_configs[SAMPLER].ignore_eos:
                for i, token in enumerate(emitted):
                    if token in self.config.eos_token_ids:
                        emitted = emitted[:i + 1]
                        break
            state.add("mtp_generated", generated + len(emitted))
            self._log_acceptance(accepted)
            out[rid] = {"new_token": [torch.tensor(emitted, dtype=torch.long)]}
        return out

    # -- slow-postprocess path (worker thread, after execute_batch) -------

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs,
    ):
        # The step ran, so its tokens are the request's context now.
        if inputs is not None:
            state = self.request_state(request_id)
            state.add("context", state.get("context", 0) + inputs.input_seq_len)
        # Seed the next decode step from the emitted token: the walk's
        # text_inputs loop-back edge (get_graph_walk_graphs) carries it. An MTP
        # verify step already set it: the token after the accepted drafts.
        if "new_token" not in outputs or "text_inputs" in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # The ONLY place .item() on a token value is allowed (slow-postprocess
        # path, off the step-critical GPU thread — base-class contract).
        if "new_token" not in outputs:
            return set()
        if self.mtp_k > 0:
            return self._mtp_check_stop(request_id, request_info, outputs["new_token"][0])
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        # Total generated = 1 prefill-emitted token + (decode iters + 1):
        # max_tokens counts the token the prefill emits, so the decode loop
        # runs one fewer.
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # The next step takes the context to prompt + generated, and it may
        # already be scheduled when this stop lands, so stop while it still
        # fits: past the context limit prepare_inputs refuses it.
        state = self.request_states.get(request_id)
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = prompt_len + generated >= self.config.context_limit
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()

    def _mtp_check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo, new_token: torch.Tensor,
    ) -> set[str]:
        """Stop on a stop token among this step's tokens, at max_tokens, or while two more
        blocks fit (the next step may already be scheduled when this stop lands): in the
        context, which counts emitted tokens against index_topk, and in the cache, where every
        verify step keeps its whole block (rejected rows as holes) against kv_rows. Counts the
        tokens emitted up to this step itself: the next step is in flight when this one is
        checked, so the GPU thread's ``mtp_generated`` may already include its tokens."""
        state = self.request_state(request_id)
        before = 0 if request_info.graph_walk == "prefill" else state.get("mtp_checked", 0)
        generated = before + new_token.numel()
        state.add("mtp_checked", generated)
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        is_eos = not ignore_eos and any(
            t in self.config.eos_token_ids for t in new_token.tolist())
        steps = request_info.dynamic_loop_iter_counts.get("decode_loop", 0)
        if request_info.graph_walk == "decode":
            steps += 1
        prompt, room = state.get("prompt_len", 0), 2 * self.mtp_block
        full = (prompt + generated + room > self.config.index_topk
                or prompt + steps * self.mtp_block + room > self.config.kv_rows)
        if is_eos or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
