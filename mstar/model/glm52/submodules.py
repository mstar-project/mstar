"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.engine.cuda_graph_config import FlashInferPackedCudaGraphConfig
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
        # The reference MoE dispatch paths (fp8-resident per-hit-expert loop;
        # naive bf16 loop under TP) use .nonzero()/host loops — illegal under
        # CUDA-graph stream capture. Capturing would fail for every bucket,
        # leave runner.graphs empty, and serve eagerly anyway; registering
        # no configs makes eager-only serving explicit. The fused
        # fused_experts_fp8 path IS capture-safe: when the loaded MoE blocks
        # resolved to it (moe_quant_kernel triton/auto-on-cuda), graphs come
        # back. Called post-load, so the resolved flag is authoritative.
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        if fp8_reference or naive_tp:
            return []
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        prefill_packed = {
            num_tokens: self._build_prefill_packed(num_tokens, device)
            for num_tokens in prefill_buckets
        }
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="decode",
                requires_cfg=False,
                labels=[_MAIN],
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill",
                replay_graph_walks=["prefill"],
                packed_seq_len_to_inputs=prefill_packed,
                requires_cfg=False,
                labels=[_MAIN],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=prefill_batch_sizes,
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
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

        cache_manager.set_active_label(_MAIN)
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label=_MAIN)
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
            "dsa_ctx": Glm52DsaForwardContext(
                spans=spans, k_store=self._dsa_k_store,
                needs_selection=needs_selection,
            ) if long_context else None,
        }

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        dsa_ctx: Glm52DsaForwardContext | None = None,
    ) -> torch.Tensor:
        return self.language_model.model(
            input_ids, cache_handle, position_ids, dsa_ctx=dsa_ctx)

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
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
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
