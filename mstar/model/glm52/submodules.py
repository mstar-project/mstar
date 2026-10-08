"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
    distribute_tokens,
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
from mstar.model.glm52.config import (
    ATTN_RESOURCE,
    INDEX_KV_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.dsa_paged import Glm52DsaPagedContext
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.pinned_staging import to_device_async

logger = logging.getLogger(__name__)

_MAIN = "main"
# How long the host waits for an MTP step to stage its verdict before giving up.
_VERDICT_TIMEOUT_S = 120.0


class _PackedPrefillConfig(PackedCudaGraphConfig):
    """Packed prefill capture without the buckets whose capture rows would pass the context
    limit: ``distribute_tokens`` gives row 0 the remainder, and such a capture fails."""

    def __init__(self, *, max_row_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_row_tokens = max_row_tokens

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n * self.total_tokens_multiplier for n in self.capture_token_lengths
                if distribute_tokens(n, bs)[0] <= self.max_row_tokens]


class Glm52LLMSubmodule(ARNodeSubmodule):
    """Embed + 78 decoder layers + lm_head, one fat TP node."""

    def __init__(self, language_model: nn.Module, config: Glm52ModelConfig):
        super().__init__()
        self.language_model = language_model  # Glm52ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        self._capture_block_warned = False
        # MTP: a decode step drafts k tokens, verifies them with the last emitted token and
        # seeds the next step, all in one captured graph; the eager fallback stays uncompiled
        self._mtp_uncompiled = config.mtp_num_draft_tokens > 0
        # dsa_long_context under CUDA graphs: one sparse-attention plan per (walk, slot, rows);
        # a slot's plans never run at once, so they share its workspace and index buffer
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        self._sparse_workspaces: dict[int, torch.Tensor] = {}  # per slot
        self._sparse_indices: dict[int, torch.Tensor] = {}  # per slot
        # dsa_long_context: requests whose prefill starts past a cached prefix, which no
        # prefill bucket's capture can hold (cg_key_info runs their step eager)
        self._prefix_hits: set[int] = set()
        # MTP per-request state: total emitted tokens (incl. the prefill-
        # emitted one — max_tokens counts it) and the stop parameters stashed
        # at prepare_inputs time, so the host cuts each verdict without engine
        # round trips.
        self._mtp_emitted: dict[str, int] = {}
        # tokens of the steps check_stop has seen; the emitted counter runs a step ahead
        # when the next step is launched before this one's stop check
        self._mtp_checked: dict[str, int] = {}
        self._mtp_max_tokens: dict[str, int] = {}
        self._mtp_ignore_eos: dict[str, bool] = {}
        # per request: max_tokens capped at the context limit (prompt-aware,
        # set by the prefill's preprocess); read by check_stop for every k
        self._token_budget: dict[str, int] = {}
        # Which trunk stream the MTP plane pairs drafts against (see
        # _mtp_pair_rows): the post-final-norm one by default.
        self._mtp_pair_postnorm = (
            os.environ.get("MSTAR_GLM52_MTP_PAIR_POSTNORM", "1") == "1"
        )
        # Each request's seed slot (the MTP state that drafts d1, and d1) in the device seed
        # buffers; slot 0 is the sink that padding and capture rows write.
        self._mtp_slot: dict[str, int] = {}
        self._mtp_free_slots: list[int] = []
        # A verify step's number, and per request where its verdict lands
        # (seq, mailbox, row), the verdict once read, and the verdict already trimmed.
        self._mtp_seq = 0
        self._mtp_pending: dict[str, tuple[int, int | None, int]] = {}
        self._mtp_verdict: dict[str, tuple[int, list[int], int]] = {}
        self._mtp_trimmed: dict[str, int] = {}
        # capture slot (None: eager) -> the pinned verdict table and step number the host reads
        self._mailboxes: dict[int | None, tuple[torch.Tensor, torch.Tensor]] = {}
        # Acceptance instrumentation: raw emitted tokens (n_accepted + 1,
        # pre-truncation), request-step count, n_accepted histogram.
        self._mtp_stat_emitted = 0
        self._mtp_stat_steps = 0
        self._mtp_stat_logged = 0
        self._mtp_stat_acc_hist = [0] * (config.mtp_num_draft_tokens + 1)

    def cleanup_request(self, request_id: str):
        self._prefix_hits.discard(request_id)
        self._mtp_emitted.pop(request_id, None)
        self._mtp_checked.pop(request_id, None)
        self._mtp_max_tokens.pop(request_id, None)
        self._mtp_ignore_eos.pop(request_id, None)
        self._token_budget.pop(request_id, None)
        for state in (self._mtp_pending, self._mtp_verdict, self._mtp_trimmed):
            state.pop(request_id, None)
        slot = self._mtp_slot.pop(request_id, None)
        if slot is not None:
            self._mtp_free_slots.append(slot)
        super().cleanup_request(request_id)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # an MTP row costs k+1 tokens, so the decode capture pads less coarsely
    MTP_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64]

    @property
    def mtp_k(self) -> int:
        return self.config.mtp_num_draft_tokens

    @property
    def mtp_block(self) -> int:
        return self.mtp_k + 1

    # ── resources ──

    def _kv(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[KV_RESOURCE]

    def _attn(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[ATTN_RESOURCE]

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        super().bind_node_resources(resources)
        if self.mtp_k > 0:
            # a request holds at least one page, so the pool bounds the requests in flight
            slots = resources[KV_RESOURCE].config.max_num_pages + 1
            weight = self.lm_head.weight
            self._mtp_seed_hidden = torch.zeros(
                slots, self.config.hidden_size, dtype=weight.dtype, device=weight.device)
            self._mtp_seed_draft = torch.zeros(slots, dtype=torch.long, device=weight.device)
            self._mtp_free_slots = list(range(slots - 1, 0, -1))

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
        steps = {KV_RESOURCE: KVStep()}
        long_context = self.config.dsa_long_context
        if not (long_context and graph_walk == "prefill"
                and sum(inp.input_seq_len for inp in inputs) > self.config.prefill_chunk_tokens):
            # a prefill in row chunks attends sparse on every row, so it plans no dense
            # attention (FlashInfer's MLA planner corrupts the heap near a million rows)
            steps[ATTN_RESOURCE] = AttentionStep(causal=True)
        if long_context:
            steps[INDEX_KV_RESOURCE] = KVStep()
        if self.mtp_k > 0 and graph_walk == "decode":
            # Under TP async the plan thread declares, admits and plans this step before
            # prepare_inputs runs, so the last verdict's trim lands here. The verify and the
            # drafts are argmaxes in the forward: no sampler.
            for rid in request_ids:
                self._apply_verdict(rid)
        else:
            steps[SAMPLER_RESOURCE] = SamplerStep(
                apply_penalty=True, prefill_tracked_tokens=prefill_tokens)
        return SubmoduleStep(
            segments=[
                Segment(rid, _MAIN, inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
            cg_key_info=self._cg_key(graph_walk, request_ids),
        )

    def cg_key_info(self, graph_walk: str, per_request_info: dict,
                    per_request_input_metadata=None, **kwargs) -> str | None:
        return self._cg_key(graph_walk, per_request_info)

    def _cg_key(self, graph_walk: str, rids) -> str | None:
        """Every capture's key (None), or for a prefill with a request past a cached prefix
        one no capture has: its rows see more keys than its bucket's capture scores."""
        if (self.config.dsa_long_context and graph_walk == "prefill"
                and not self._prefix_hits.isdisjoint(rids)):
            return "eager"
        return None

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        if self.config.dsa_long_context and graph_walk == "prefill" and start > 0:
            self._prefix_hits.add(fwd_info.rid_handle)
        return super().split_inputs(graph_walk, fwd_info, inputs, start, end)

    def max_step_tokens(self, graph_walk: str) -> int | None:
        cap = self.config.prefill_max_step_tokens
        if graph_walk != "prefill" or cap is None:
            return None
        if cap == "auto":
            return max(self.config.prefill_batched_token_buckets
                       or self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS)
        return cap

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
        blocked = fp8_reference or naive_tp
        # getattr: some tests build submodules that never ran __init__
        if blocked and not getattr(self, "_capture_block_warned", False):
            self._capture_block_warned = True
            logger.warning(
                "Glm52LLMSubmodule: %s; no CUDA graphs are registered and "
                "decode runs eager, which is far slower. %s",
                "the reference MoE dispatch is active" if fp8_reference
                else "naive TP MoE dispatch is active",
                "moe_quant_kernel 'auto' or 'triton' serves the capturable "
                "fused fp8 kernel." if fp8_reference
                else "Serve the fp8 checkpoint to get the fused kernel.",
            )
        return blocked

    def _compile_flags(self) -> dict[str, Any]:
        # MSTAR_GLM52_GRAPH_COMPILE=0 captures the eager forward (escape hatch
        # for an Inductor toolchain crash); "default" is the cuBLAS-backed
        # mode of cuda_graph_runner.resolve_compile_mode, the fastest here.
        return {
            "compile": os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") == "1",
            "compile_mode": "default",
        }

    @property
    def disable_torch_compile(self) -> bool:
        # the escape hatch covers the uncaptured steps too, which the engine compiles
        return (getattr(self, "_mtp_uncompiled", False)
                or os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") != "1")

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
            self.__dict__.pop("_device", None)
            return super().to(device=device, non_blocking=non_blocking)
        return self

    def get_device(self) -> torch.device:
        # cached: the base walks parameters() on every call
        if "_device" not in self.__dict__:
            self._device = super().get_device()
        return self._device

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        if self._moe_capture_blocked(tp_world_size):
            return []
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        flags = self._compile_flags()
        groups = [(prefill_buckets, prefill_batch_sizes)]
        batched = self.config.prefill_batched_token_buckets
        if batched:
            # one request's buckets and a batch's apart, so each can be fine or coarse
            groups = [(prefill_buckets, [bs for bs in prefill_batch_sizes if bs == 1]),
                      (batched, [bs for bs in prefill_batch_sizes if bs > 1])]
        configs = [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                # one id in; under MTP the step verifies it and k drafts
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=self.mtp_block,
                ),
                capture_batch_sizes=self.MTP_CAPTURE_BATCH_SIZES if self.mtp_k else None,
                **flags,
            ),
            *(
                _PackedPrefillConfig(
                    max_row_tokens=self._context_limit(),
                    capture_graph_walk="prefill",
                    capture_token_lengths=list(buckets),
                    make_node_input=lambda n: ARNodeInputs(
                        input_ids=torch.zeros(n, dtype=torch.long, device=device),
                        input_seq_len=n,
                    ),
                    capture_batch_sizes=list(sizes),
                    **flags,
                )
                for buckets, sizes in groups if sizes
            ),
        ]
        if self.config.dsa_long_context and self.mtp_k:
            # MTP's prefill over DSA stays eager
            configs = [c for c in configs if c.capture_graph_walk == "decode"]
        self._fit_recompile_limit(configs)
        return configs

    def _fit_recompile_limit(self, configs: list[CudaGraphConfig]) -> None:
        """Raise dynamo's recompile limit to what these captures take: a decoder layer's
        frames hold an entry per captured token count and layer kind (dense, MoE, the last
        one on sampled rows), and the shapes past the limit capture uncompiled."""
        tokens = {
            n for config in configs
            for bs in config.capture_batch_sizes or DEFAULT_CAPTURE_BATCH_SIZES
            for n in config.get_total_tokens(bs)
        }
        need = 3 * len(tokens) + 16  # and the engine's compiled fallback
        if torch._dynamo.config.recompile_limit < need:
            logger.info("Glm52LLMSubmodule: dynamo recompile_limit %d -> %d for %d "
                        "captured token counts", torch._dynamo.config.recompile_limit, need,
                        len(tokens))
        self._recompile_limit = need
        self._apply_recompile_limit()

    def _apply_recompile_limit(self) -> None:
        # dynamo's config is per thread, and the engine's step thread starts at its default
        need = getattr(self, "_recompile_limit", 0)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, need)

    # ── per-step contract ──

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        # here, not in preprocess, so the refusal fails this request and not its batch
        # (process_prompt refuses the same prompts first, as a 400)
        if graph_walk == "prefill" and text_inputs.shape[0] > self.config.max_prompt_tokens:
            raise RuntimeError(
                f"request {fwd_info.request_id}: a prompt of {text_inputs.shape[0]} tokens "
                f"exceeds the {self.config.max_prompt_tokens} served"
            )
        if self.config.mtp_num_draft_tokens > 0:
            # the handle, which the batch, check_stop and cleanup_request key by
            rid = fwd_info.rid_handle
            sampling = fwd_info.resource_configs[SAMPLER_RESOURCE]
            # Greedy-only: decode drafts and verification bypass the sampler,
            # so a temperature would be silently ignored and a repetition
            # penalty would move even the prefill argmax. Refuse this rid.
            if sampling.temperature != 0 or sampling.repetition_penalty != 1:
                raise RuntimeError(
                    f"request {fwd_info.request_id}: MTP speculative decoding is greedy-only "
                    f"but the request asks for temperature={sampling.temperature}, "
                    f"repetition_penalty={sampling.repetition_penalty}. Send "
                    "temperature=0 without a penalty, or serve a k=0 config."
                )
            # capped by the prefill's context-limit clamp once that has run
            self._mtp_max_tokens[rid] = min(
                fwd_info.max_tokens, self._token_budget.get(rid, fwd_info.max_tokens))
            self._mtp_ignore_eos[rid] = sampling.ignore_eos
            if graph_walk == "decode":
                # one id in, the last emitted token: the step verifies it and k drafts
                return ARNodeInputs(input_ids=text_inputs, input_seq_len=self.mtp_block)
            self._mtp_emitted[rid] = 1
            if rid not in self._mtp_slot:
                self._mtp_slot[rid] = self._mtp_free_slots.pop()
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
        )

    def _context_limit(self) -> int:
        """Rows a request may hold: the serving window on the DSA engine
        path, else index_topk, where dense MLA is exactly GLM-5.2's DSA."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def _clamp_token_budget(self, engine_inputs: ModelInputsFromEngine, rid: str, room: int) -> None:
        """Cap the request's max_tokens at what fits before the context
        limit, so decode stops there (check_stop, the MTP verify truncation)
        instead of tripping preprocess's guard mid-batch. ``room`` is the
        prompt-aware count, known only once the prefill rows are declared."""
        info = (getattr(engine_inputs, "per_request_info", None) or {}).get(rid)
        asked = getattr(info, "max_tokens", None)
        if asked is None:
            return
        budget = min(int(asked), room)
        if budget < asked:
            logger.info(
                "request %s: max_tokens %d capped to %d by the context limit %d",
                rid, asked, budget, self._context_limit(),
            )
        self._token_budget[rid] = budget
        if rid in self._mtp_max_tokens:
            self._mtp_max_tokens[rid] = min(self._mtp_max_tokens[rid], budget)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        self._apply_recompile_limit()
        kv = self._kv(engine_inputs)
        request_ids = list(engine_inputs.request_ids)
        seq_lens = [inp.input_seq_len for inp in inputs]
        device = self.get_device()
        long_context = self.config.dsa_long_context
        if long_context and graph_walk == "prefill":
            self._prefix_hits.difference_update(request_ids)  # its step has been leased
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window.
        limit = self._context_limit()
        pos_ids_list: list[int] = []
        needs_selection = False
        index = engine_inputs.resources[INDEX_KV_RESOURCE] if long_context else None
        for rid, sl in zip(request_ids, seq_lens, strict=True):
            start = kv.stored_len(rid)
            if index is not None and index.stored_len(rid) != start:
                # each cache matches prefixes on its own: a prefix reused by one and not the
                # other would leave the index keys missing, and selection would read garbage
                raise RuntimeError(
                    f"request {rid}: the latent cache holds {start} tokens but the DSA index "
                    f"store {index.stored_len(rid)}; their prefix caches diverged")
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
            if graph_walk == "prefill":
                # a prompt of P rows emits at most limit - P tokens, one row short of
                # the limit: the next step may already be running when the stop lands.
                # Under MTP every step writes a whole block, that one included.
                room = limit - (start + sl) - (2 * self.mtp_block if self.mtp_k else 0)
                self._clamp_token_budget(engine_inputs, rid, room)
            # a row past index_topk keys picks its top index_topk; below, it attends all it sees
            needs_selection |= long_context and start + sl > self.config.index_topk
            pos_ids_list.extend(range(start, start + sl))
        # Every per-step index in one pinned H2D (a pageable torch.tensor(..., device=cuda)
        # drains the stream before the step even starts): positions; on prefill each
        # request's last row (for an eager prefill, when no plan buffer carries qo_indptr);
        # under MTP the seed slots (padding and capture rows read and write the sink) and,
        # on decode, the step's number.
        fields = {"position_ids": pos_ids_list}
        if graph_walk == "prefill":
            fields["last_token_indices"] = [end - 1 for end in itertools.accumulate(seq_lens)]
        if self.mtp_k > 0:
            fields["mtp_slots"] = [self._mtp_slot.get(rid, 0) for rid in request_ids]
            if graph_walk == "decode":
                fields["mtp_seq"] = [self._mtp_stage_seq(engine_inputs)]
        staged = to_device_async(
            [v for values in fields.values() for v in values], torch.long, device)
        out = dict(zip(fields, staged.split([len(v) for v in fields.values()]), strict=True))
        out["input_ids"] = (
            inputs[0].input_ids if len(inputs) == 1 else torch.cat([inp.input_ids for inp in inputs]))
        if long_context:
            out.update(self._dsa_inputs(
                graph_walk, engine_inputs, pos_ids_list, seq_lens, needs_selection))
        return out

    def _mtp_pair_rows(self, normed: torch.Tensor, prenorm: torch.Tensor) -> torch.Tensor:
        """The trunk stream the MTP plane pairs drafts against: the
        post-final-norm one, which the reference implementation pairs and
        which accepts markedly better, unless
        MSTAR_GLM52_MTP_PAIR_POSTNORM=0."""
        return normed if self._mtp_pair_postnorm else prenorm

    def _dsa_inputs(self, graph_walk: str, engine_inputs, positions: list[int],
                    seq_lens: list[int], needs_selection: bool) -> dict[str, Any]:
        """The step's DSA state as preprocess outputs. Its tensors are top level, so a CUDA
        graph interns them and each replay stages the step's values; the host half rides in
        ``dsa_meta``. Under a graph the page tables span the serving window, the sparse plan is
        the slot's own and re-planned here every step, and every row scores on its own (a row
        within index_topk selects all its keys): a decode over the window, a prefill bucket
        over its own rows."""
        device = self.get_device()
        step_ctx = engine_inputs.step.ctx
        graph = step_ctx.capture or step_ctx.slot_lease is not None
        plans = step_ctx.plan_results
        topk = self.config.index_topk
        lens = [p + 1 for p in positions]
        row_req = [i for i, sl in enumerate(seq_lens) for _ in range(sl)]
        page_size = self._kv(engine_inputs).config.page_size
        width = max(lens)
        if graph and graph_walk == "prefill":
            # A bucket holds whole prompts (cg_key_info runs a cached prefix's step eager), so
            # its rows see at most its own rows of keys, and one no wider than index_topk
            # selects nothing. Its padding rows, position 0 of the first request, are staged:
            # the static buffers keep what a larger bucket left there.
            rows = step_ctx.slot_lease.bucket.num_tokens if step_ctx.slot_lease else len(lens)
            if width > rows:
                raise RuntimeError(
                    f"a captured prefill row sees {width} keys, past its {rows}-row bucket")
            lens += [1] * (rows - len(lens))
            row_req += [0] * (rows - len(row_req))
            width, needs_selection = rows, rows > topk
        elif graph:
            width, needs_selection = -(-self.config.max_seq_len // page_size) * page_size, True

        def table(key):
            rows = [view.page_idxs for view in plans[key][_MAIN].views]
            page = engine_inputs.resources[key].config.page_size
            width = -(-self.config.max_seq_len // page) if graph else max(len(r) for r in rows)
            # zero-padded in C: the host work is the real pages, not rows x the window
            host = torch.zeros(len(rows), width, dtype=torch.int32)
            for i, r in enumerate(rows):
                host[i, : len(r)] = torch.as_tensor(r, dtype=torch.int32)
            return to_device_async(host, torch.int32, device)

        starts = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        sparse_plan = None
        if graph and needs_selection:
            attn = self.language_model.model.layers[0].self_attn
            slot, rows = step_ctx.slot, len(lens)
            key = (graph_walk, slot, rows)
            sparse_plan = self._sparse_plans.get(key)
            if sparse_plan is None:
                if slot not in self._sparse_workspaces:
                    self._sparse_workspaces[slot] = torch.empty(
                        sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8, device=device)
                indices = self._sparse_indices.get(slot)
                # +1: the plan's spare index entry
                if indices is None or indices.numel() < rows * topk + 1:
                    indices = self._sparse_indices[slot] = torch.zeros(
                        rows * topk + 1, dtype=torch.int32, device=device)
                sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                    rows, topk, self._sparse_workspaces[slot], indices)
            sparse_plan.plan([min(n, topk) for n in lens], attn.num_heads,
                             self.config.kv_lora_rank, self.config.qk_rope_head_dim,
                             attn.softmax_scale)
        return {
            "dsa_row_req": to_device_async(row_req, torch.int32, device),
            "dsa_lens": to_device_async(lens, torch.int32, device),
            "dsa_kv_table": table(KV_RESOURCE),
            "dsa_index_table": table(INDEX_KV_RESOURCE),
            "dsa_meta": dict(
                host_lens=lens,
                spans=[(r0, sl, i) for i, (r0, sl) in enumerate(zip(starts, seq_lens,
                                                                     strict=True))],
                width=width, page_size=page_size, topk=topk, needs_selection=needs_selection,
                decode_rows=graph or graph_walk == "decode", sparse_plan=sparse_plan),
        }

    @staticmethod
    def _dsa_ctx(kwargs: dict) -> Glm52DsaPagedContext | None:
        """The forward's DSA context, from its preprocess inputs."""
        meta = kwargs.get("dsa_meta")
        if meta is None:
            return None
        return Glm52DsaPagedContext(
            row_req=kwargs["dsa_row_req"], lens=kwargs["dsa_lens"], host_lens=meta["host_lens"],
            kv_table=kwargs["dsa_kv_table"], index_table=kwargs["dsa_index_table"],
            spans=meta["spans"], width=meta["width"], page_size=meta["page_size"],
            topk=meta["topk"], needs_selection=meta["needs_selection"],
            decode_rows=meta["decode_rows"], sparse_plan=meta["sparse_plan"])

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaPagedContext | None = None,
        rows: torch.Tensor | None = None,
        with_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.language_model.model(
            input_ids, position_ids, dsa_ctx=dsa_ctx, rows=rows, return_prenorm=with_prenorm)

    @torch.compiler.disable
    def _last_row_index(
        self, engine_inputs: ModelInputsFromEngine, kwargs: dict,
    ) -> torch.Tensor:
        """Each request's last row of a packed prefill: from the plan's
        qo_indptr buffer where the attention resource has one (captured, or
        the MLA resource), else from preprocess's real lengths."""
        qo_indptr = self._attn(engine_inputs).qo_indptr_buf(_MAIN)
        if qo_indptr is not None:
            # the buffer is sized to the capture bucket and tail-filled with
            # the last real offset; slice to the real request count so a
            # padded prefill does not duplicate the final request's row
            n = len(engine_inputs.request_ids)
            return (qo_indptr[1 : n + 1] - 1).long()
        last = kwargs.get("last_token_indices")
        assert last is not None, "eager prefill needs last_token_indices from preprocess"
        return last

    # Outside dynamo: traced, the frame after the qo_indptr read is keyed by the tokens and
    # the rows together, one entry per capture bucket, and past the recompile limit it
    # runs uncompiled.
    @torch.compiler.disable
    def _last_rows(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, kwargs: dict,
    ) -> torch.Tensor:
        return hidden.index_select(0, self._last_row_index(engine_inputs, kwargs))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        dsa_ctx = self._dsa_ctx(kwargs)
        if self._long_prefill(graph_walk, input_ids, dsa_ctx):
            last = [input_ids.shape[0] - 1]
            return {"logits": [self.lm_head(
                self._prefill_rows_chunked(input_ids, position_ids, dsa_ctx, last))]}
        hidden = self._hidden(input_ids, position_ids, dsa_ctx)
        return {"logits": [self.lm_head(hidden[-1:])]}

    def _long_prefill(self, graph_walk: str, input_ids: torch.Tensor, dsa_ctx) -> bool:
        """An eager prefill past prefill_chunk_tokens; a captured one (its rows score on their
        own, ``decode_rows``) never runs in chunks, whatever the bucket's size."""
        return (graph_walk == "prefill" and dsa_ctx is not None and not dsa_ctx.decode_rows
                and input_ids.shape[0] > self.config.prefill_chunk_tokens)

    def _prefill_rows_chunked(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaPagedContext, rows: list[int],
    ) -> torch.Tensor:
        """The trunk's final hidden state at ``rows`` (ascending) of a paged-DSA prefill longer
        than ``prefill_chunk_tokens``: the trunk runs over row chunks in turn, each writing its
        keys and latents before the next one selects over them, so only a chunk's activations
        are ever live."""
        chunk = self.config.prefill_chunk_tokens
        picked = []
        for c0 in range(0, input_ids.shape[0], chunk):
            c1 = min(c0 + chunk, input_ids.shape[0])
            hidden = self._hidden(input_ids[c0:c1], position_ids[c0:c1], dsa_ctx.rows(c0, c1))
            want = [r - c0 for r in rows if c0 <= r < c1]
            if want:
                picked.append(hidden[want])
        return torch.cat(picked)

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
        if self.mtp_k > 0 and graph_walk == "prefill":
            return self._mtp_prefill(engine_inputs, input_ids, position_ids, kwargs)
        if self.mtp_k > 0 and graph_walk == "decode":
            return self._mtp_decode(
                engine_inputs, input_ids, position_ids, kwargs["mtp_slots"], kwargs["mtp_seq"],
                self._dsa_ctx(kwargs))
        if graph_walk not in ("prefill", "decode"):
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")
        if self.config.dsa_long_context and not engine_inputs.captured:
            # DSA outside a graph (prefill, oversized decode) is host-driven per step:
            # traced, its metadata would key the compiled frame and recompile it every step
            return self._forward_batched_eager(
                graph_walk, engine_inputs, input_ids, position_ids, **kwargs)
        return self._forward_batched(graph_walk, engine_inputs, input_ids, position_ids, **kwargs)

    @torch.compiler.disable
    def _forward_batched_eager(self, *args, **kwargs) -> dict[str, NameToTensorList]:
        return self._forward_batched(*args, **kwargs)

    def _forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        dsa_ctx = self._dsa_ctx(kwargs)
        if self._long_prefill(graph_walk, input_ids, dsa_ctx):
            last = [r0 + n - 1 for r0, n, _ in dsa_ctx.spans]
            return self._sample(engine_inputs, self.lm_head(
                self._prefill_rows_chunked(input_ids, position_ids, dsa_ctx, last)))
        rows = None
        if graph_walk == "prefill" and self.config.prefill_last_layer_rows:
            rows = self._last_row_index(engine_inputs, kwargs)
        hidden = self._hidden(input_ids, position_ids, dsa_ctx, rows=rows)
        if graph_walk == "prefill" and rows is None:
            hidden = self._last_rows(engine_inputs, hidden, kwargs)
        logits = self.lm_head(hidden)  # (bs, vocab)
        return self._sample(engine_inputs, logits)

    # Keyed by request id, so outside dynamo: traced, the engine's compiled fallback guards
    # on the ids and recompiles this tail for every new request (the sampler runs eager
    # either way).
    @torch.compiler.disable
    def _sample(
        self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        request_ids = list(engine_inputs.request_ids)
        new_tokens = engine_inputs.resources[SAMPLER_RESOURCE].sample(
            request_ids, logits=logits)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }

    # ── MTP: one captured graph per step ──
    #
    # A decode step's block is k+1 rows per request at positions L..L+k: the last emitted token
    # and k drafts. The MTP plane's row p pairs (embed(token p+1), hidden p) at RoPE position
    # p+1, so it shares the trunk's page table and write slots.

    def _mtp_prefill(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        position_ids: torch.Tensor, kwargs: dict,
    ) -> dict[str, NameToTensorList]:
        """The trunk over the prompt and its sampled token, then the MTP pass over every prompt
        row (each request's last row pairs with its sampled token): it fills the MTP plane and
        seeds the first decode step."""
        lm = self.language_model
        dsa_ctx = self._dsa_ctx(kwargs)
        if self._long_prefill("prefill", input_ids, dsa_ctx):
            return self._mtp_prefill_chunked(engine_inputs, input_ids, position_ids, kwargs,
                                             dsa_ctx)
        hidden, prenorm = self._hidden(input_ids, position_ids, dsa_ctx, with_prenorm=True)
        last = self._last_row_index(engine_inputs, kwargs)
        new_tokens = self._sample_tokens(engine_inputs, self.lm_head(hidden.index_select(0, last)))
        # roll, not a shifted copy into empty memory: a captured bucket's tail is padding,
        # and every row must hold a valid id for the embedding
        nxt = torch.roll(input_ids, -1)
        nxt.index_copy_(0, last, new_tokens.to(nxt.dtype))
        h_head, h_raw = lm.mtp(
            lm.model.embed_tokens(nxt), self._mtp_pair_rows(hidden, prenorm), position_ids + 1,
            dsa_ctx=dsa_ctx)
        self._mtp_seed(kwargs["mtp_slots"], h_head.index_select(0, last),
                       h_raw.index_select(0, last), new_tokens)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _mtp_prefill_chunked(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        position_ids: torch.Tensor, kwargs: dict, dsa_ctx: Glm52DsaPagedContext,
    ) -> dict[str, NameToTensorList]:
        """``_mtp_prefill`` past ``prefill_chunk_tokens``: the trunk and then the MTP pass over
        each row chunk in turn. A request's last row pairs with its sampled token, unknown
        until every chunk has run, so its MTP row is written again once that token is in."""
        lm, chunk = self.language_model, self.config.prefill_chunk_tokens
        last = [r0 + n - 1 for r0, n, _ in dsa_ctx.spans]
        nxt = torch.roll(input_ids, -1)
        normed, paired = [], []
        for c0 in range(0, input_ids.shape[0], chunk):
            c1 = min(c0 + chunk, input_ids.shape[0])
            sub = dsa_ctx.rows(c0, c1)
            hidden, prenorm = self._hidden(
                input_ids[c0:c1], position_ids[c0:c1], sub, with_prenorm=True)
            pair = self._mtp_pair_rows(hidden, prenorm)
            lm.mtp(lm.model.embed_tokens(nxt[c0:c1]), pair, position_ids[c0:c1] + 1, dsa_ctx=sub)
            want = [r - c0 for r in last if c0 <= r < c1]
            if want:
                normed.append(hidden[want])
                paired.append(pair[want])
        new_tokens = self._sample_tokens(engine_inputs, self.lm_head(torch.cat(normed)))
        rows = torch.tensor(last, device=input_ids.device)
        h_head, h_raw = lm.mtp(
            lm.model.embed_tokens(new_tokens), torch.cat(paired),
            position_ids.index_select(0, rows) + 1, dsa_ctx=dsa_ctx.pick(last))
        self._mtp_seed(kwargs["mtp_slots"], h_head, h_raw, new_tokens)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _mtp_decode(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        position_ids: torch.Tensor, mtp_slots: torch.Tensor, mtp_seq: torch.Tensor,
        dsa_ctx: Glm52DsaPagedContext | None = None,
    ) -> dict[str, NameToTensorList]:
        """One verify step. Drafts: d1 from the seed, each later one an MTP pass whose query
        sits at its row of this step's block. Verify: the trunk over [last | drafts], the
        verdict (the target's argmax per row, then the accepted count) to the host's mailbox.
        Seed: the MTP pass over the block with the target's argmax as each row's next token (up
        to the accepted row it equals the drafts), read at the last accepted row. With paged
        DSA the verify and the seed pass select per row; the draft passes attend densely, which
        moves only the acceptance."""
        lm, k, block = self.language_model, self.mtp_k, self.mtp_block
        embed, n = lm.model.embed_tokens, input_ids.shape[0]
        positions = position_ids.view(n, block)
        h, d1 = self._mtp_read_seed(mtp_slots)
        drafts = [d1]
        rows: dict[str, torch.Tensor] = {}
        for row in range(k - 1):
            h_head, h = lm.mtp.forward_block_row(
                embed(drafts[-1]), h, positions[:, row] + 1, row, block, rows)
            drafts.append(self._draft_tokens(h_head, drafts[-1]))
        drafts = torch.stack(drafts, 1)
        ids = torch.cat([input_ids.view(n, 1), drafts], 1).view(-1)
        hidden, prenorm = self._hidden(ids, position_ids, dsa_ctx, with_prenorm=True)
        target = self.lm_head(hidden).argmax(-1).view(n, block)
        accepted = (drafts == target[:, :k]).long().cumprod(1).sum(1)
        self._stage_verdict(engine_inputs, torch.cat([target, accepted[:, None]], 1), mtp_seq)
        h_head, h_raw = lm.mtp(
            embed(target.view(-1)), self._mtp_pair_rows(hidden, prenorm), position_ids + 1,
            dsa_ctx=dsa_ctx)
        at = torch.arange(n, device=accepted.device) * block + accepted
        next_input = target.view(-1).index_select(0, at)
        self._mtp_seed(mtp_slots, h_head.index_select(0, at), h_raw.index_select(0, at), next_input)
        return {
            rid: {"text_inputs": [next_input[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # The seed buffers and the mailbox live outside dynamo: they are the submodule's own
    # state, written at fixed addresses that a captured replay keeps.
    @torch.compiler.disable
    def _mtp_read_seed(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mtp_seed_hidden.index_select(0, slots), self._mtp_seed_draft.index_select(0, slots)

    @torch.compiler.disable
    def _mtp_seed(
        self, slots: torch.Tensor, h_head: torch.Tensor, h_raw: torch.Tensor, prev: torch.Tensor,
    ) -> None:
        """The next step's draft seed in each row's slot: the MTP state and d1, the draft for
        the token after ``prev``. The head reads the shared_head-normed rows; the chain threads
        the raw layer output."""
        self._mtp_seed_hidden.index_copy_(0, slots, h_raw.to(self._mtp_seed_hidden.dtype))
        self._mtp_seed_draft.index_copy_(0, slots, self._draft_tokens(h_head, prev))

    def _draft_tokens(self, h_head: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """The draft for the token after ``prev``, from the MTP head input ``h_head``."""
        del prev  # the tests' oracle drafts read it
        return self.lm_head(h_head).argmax(-1)

    @torch.compiler.disable
    def _stage_verdict(
        self, engine_inputs: ModelInputsFromEngine, verdict: torch.Tensor, mtp_seq: torch.Tensor,
    ) -> None:
        """The host's copy: the verdict first, then its step number."""
        verdicts, seq = self._mailboxes[self._mailbox_key(engine_inputs)]
        verdicts[: verdict.shape[0]].copy_(verdict, non_blocking=True)
        seq.copy_(mtp_seq, non_blocking=True)

    # Keyed by request id, so outside dynamo (as _sample).
    @torch.compiler.disable
    def _sample_tokens(
        self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor,
    ) -> torch.Tensor:
        return engine_inputs.resources[SAMPLER_RESOURCE].sample(
            list(engine_inputs.request_ids), logits=logits)

    # ── MTP host side: the verdict mailbox ──

    @staticmethod
    def _mailbox_key(engine_inputs: ModelInputsFromEngine) -> int | None:
        """A captured step's capture slot, None eager: each has its own mailbox, which a graph
        writes at a fixed address."""
        lease = engine_inputs.step.ctx.slot_lease if engine_inputs.step is not None else None
        if not engine_inputs.captured or lease is None or lease.bucket is None:
            return None
        return lease.slot

    def _mtp_stage_seq(self, engine_inputs: ModelInputsFromEngine) -> int:
        """Number this verify step and note where each real request's verdict lands."""
        key = self._mailbox_key(engine_inputs)
        if key not in self._mailboxes:
            # here, not in the forward: pinned memory can't be allocated under capture
            pin = self.lm_head.weight.device.type == "cuda"
            rows = self._mtp_seed_draft.shape[0]
            self._mailboxes[key] = (
                torch.zeros(rows, self.mtp_block + 1, dtype=torch.long, pin_memory=pin),
                torch.full((1,), -1, dtype=torch.long, pin_memory=pin),
            )
        self._mtp_seq += 1
        ctx = engine_inputs.step.ctx if engine_inputs.step is not None else None
        if ctx is None or not ctx.capture:
            # the real rows only: a padded replay's dummy rows run too
            real = set(ctx.request_ids if ctx is not None else engine_inputs.per_request_info)
            for row, rid in enumerate(engine_inputs.request_ids):
                if rid in real:
                    self._mtp_pending[rid] = (self._mtp_seq, key, row)
        return self._mtp_seq

    def _read_verdicts(self, request_ids: list[str]) -> list[tuple[list[int], int]]:
        """One verify step's ``(tokens, accepted)`` per request, waiting once for the device
        to stage them (a step's requests share its number and mailbox)."""
        pending = [self._mtp_pending[rid] for rid in request_ids]
        seq, key, _ = pending[0]
        assert all(p[:2] == (seq, key) for p in pending), pending
        # read once, then kept: a later step may reuse the mailbox before a second read
        cached = [self._mtp_verdict.get(rid) for rid in request_ids]
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
                    f"MTP verdict of step {seq} was overwritten by step {staged} before it was read")
            if time.monotonic() > deadline:
                raise RuntimeError(f"MTP verdict of step {seq} never arrived")
            time.sleep(0)
        table = verdicts[: max(p[2] for p in pending) + 1].tolist()
        block = self.mtp_block
        out = [(table[p[2]][:block], table[p[2]][block]) for p in pending]
        for rid, verdict in zip(request_ids, out, strict=True):
            self._mtp_verdict[rid] = (seq, *verdict)
        return out

    def _apply_verdict(self, rid: str) -> None:
        """Take the last verify step's rejected rows back from the KV length, before this step
        is admitted and planned. Once per verdict: a re-declared or abandoned step must not
        trim twice. Ids with no pending verdict (prefill just ran, capture rows) are skipped."""
        pending = self._mtp_pending.get(rid)
        if pending is None or self._mtp_trimmed.get(rid) == pending[0]:
            return
        _, accepted = self._read_verdicts([rid])[0]
        if accepted < self.mtp_k:
            self._kv().correct_len(rid, _MAIN, accepted - self.mtp_k)
            if self.config.dsa_long_context:
                # the index keys of the rejected rows go with their latents
                self.node_resources[INDEX_KV_RESOURCE].correct_len(
                    rid, _MAIN, accepted - self.mtp_k)
        self._mtp_trimmed[rid] = pending[0]

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[NodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """A verify step's emitted tokens per request: the accepted drafts and the token after
        them, cut at max_tokens and at a stop token (always the last element, which check_stop
        relies on). Waits for the device to stage the verdict; the seed pass is still running."""
        if self.mtp_k == 0 or not request_ids:
            return {}
        if per_request_info[request_ids[0]].graph_walk != "decode":
            return {}
        eos_ids = self.config.eos_token_ids
        out = {}
        for rid, (target, accepted) in zip(
            request_ids, self._read_verdicts(request_ids), strict=True,
        ):
            self._mtp_stat_steps += 1
            self._mtp_stat_emitted += accepted + 1
            self._mtp_stat_acc_hist[accepted] += 1
            budget = self._mtp_max_tokens[rid] - self._mtp_emitted[rid]
            emitted = target[: min(accepted + 1, max(budget, 1))]
            if not self._mtp_ignore_eos[rid]:
                for j, token in enumerate(emitted):
                    if token in eos_ids:
                        emitted = emitted[: j + 1]
                        break
            self._mtp_emitted[rid] += len(emitted)
            out[rid] = {"new_token": [torch.tensor(emitted, dtype=torch.long)]}
        self._maybe_log_mtp_acceptance()
        return out

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
            "post-final-norm (default, vLLM's)" if self._mtp_pair_postnorm
            else "pre-final-norm (MSTAR_GLM52_MTP_PAIR_POSTNORM=0)",
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
            # MTP verify step: the forward returned the next input (the token after the
            # accepted drafts); new_token is the host's cut of the verdict
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
        # the prompt-aware cap from the prefill's preprocess, else the request's own
        budget = self._token_budget.get(request_id, request_info.max_tokens)
        if self.mtp_k > 0:
            # multi-token emission: in-step truncation guarantees a stop id
            # can only be the LAST element; totals live in the per-request
            # counter (loop iters no longer count tokens)
            tokens = outputs["new_token"][0]
            last = int(tokens[-1])
            is_eos = last in self.config.eos_token_ids
            generated = self._mtp_checked.get(request_id, 0) + tokens.numel()
            self._mtp_checked[request_id] = generated
            if (not ignore_eos and is_eos) or generated >= budget:
                return {"decode_loop"}
            return set()
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        # max_tokens counts the token the prefill emits, so the total is
        # 1 + (iters + 1) decode tokens
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and is_eos) or generated >= budget:
            return {"decode_loop"}
        return set()
