"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
import os
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
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
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
from mstar.utils.pinned_staging import to_device_async

logger = logging.getLogger(__name__)

_MAIN = "main"


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
        # DSA indexer k-cache (dsa.py): per-request index keys, appended by
        # FULL layers each forward when dsa_long_context is on; evicted in
        # cleanup_request.
        self._dsa_k_store = Glm52DsaKStore()
        # per request: max_tokens capped at the context limit (prompt-aware,
        # set by the prefill's preprocess); read by check_stop
        self._token_budget: dict[str, int] = {}

    def cleanup_request(self, request_id: str):
        self._dsa_k_store.evict(request_id)
        self._token_budget.pop(request_id, None)
        super().cleanup_request(request_id)

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    # ── resources ──

    def _kv(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[KV_RESOURCE]

    def _attn(self, engine_inputs: ModelInputsFromEngine | None = None):
        res = engine_inputs.resources if engine_inputs is not None else self.node_resources
        return res[ATTN_RESOURCE]

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
        return SubmoduleStep(
            segments=segments,
            steps={
                KV_RESOURCE: KVStep(),
                ATTN_RESOURCE: AttentionStep(causal=True),
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
        return os.environ.get("MSTAR_GLM52_GRAPH_COMPILE", "1") != "1"

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
        if self.config.dsa_long_context:
            # DSA maintenance is host-side per-request work; a captured
            # decode would skip index upkeep. Eager-only.
            return []
        if self._moe_capture_blocked(tp_world_size):
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
            _PackedPrefillConfig(
                max_row_tokens=self._context_limit(),
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
        limit, so decode stops there (check_stop) instead of tripping
        preprocess's guard mid-batch. ``room`` is the prompt-aware count,
        known only once the prefill rows are declared."""
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

    @staticmethod
    def _page_tables(engine_inputs: ModelInputsFromEngine) -> list[list[int]]:
        """Each request's KV pages, in request order, from this step's plan."""
        views = engine_inputs.step.ctx.plan_results[KV_RESOURCE][_MAIN].views
        return [list(view.page_idxs) for view in views]

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        kv = self._kv(engine_inputs)
        request_ids = list(engine_inputs.request_ids)
        seq_lens = [inp.input_seq_len for inp in inputs]

        device = self.get_device()
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window.
        long_context = self.config.dsa_long_context
        limit = self._context_limit()
        topk = self.config.index_topk
        pos_ids_list: list[int] = []
        spans: list[Glm52DsaRequestSpan] = []
        needs_selection = False
        q_start = 0
        page_tables = self._page_tables(engine_inputs) if long_context else None
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
            if graph_walk == "prefill":
                # a prompt of P rows emits at most limit - P tokens, one row short of
                # the limit: the next step may already be running when the stop lands
                self._clamp_token_budget(engine_inputs, rid, limit - (start + sl))
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
                    ctx_start=start, page_indices=page_tables[i],
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
            "dsa_ctx": Glm52DsaForwardContext(
                spans=spans, k_store=self._dsa_k_store,
                needs_selection=needs_selection,
            ) if long_context else None,
        }

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
    ) -> torch.Tensor:
        return self.language_model.model(input_ids, position_ids, dsa_ctx=dsa_ctx)

    def _last_rows(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, kwargs: dict,
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
            return hidden.index_select(0, (qo_indptr[1 : n + 1] - 1).long())
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
        hidden = self._hidden(input_ids, position_ids, kwargs.get("dsa_ctx"))
        if graph_walk == "prefill":
            hidden = self._last_rows(engine_inputs, hidden, kwargs)
        elif graph_walk != "decode":
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")
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

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
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
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        # max_tokens counts the token the prefill emits, so the total is
        # 1 + (iters + 1) decode tokens
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and is_eos) or generated >= budget:
            return {"decode_loop"}
        return set()
