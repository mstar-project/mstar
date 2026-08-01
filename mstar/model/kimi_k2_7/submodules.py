"""AR submodule for the Kimi-K2.7 text backbone."""
from __future__ import annotations

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
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.sampling import Sampler

_MAIN = "main"


class KimiLLMSubmodule(ARNodeSubmodule):
    def __init__(self, language_model: nn.Module, config: KimiK2Config):
        super().__init__()
        self.language_model = language_model  # KimiForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def _build_prefill_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros((num_tokens,), dtype=torch.long, device=device),
            "position_ids": torch.arange(num_tokens, dtype=torch.long, device=device),
        }

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig | FlashInferPackedCudaGraphConfig]:
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
        pos_ids_list: list[int] = []
        for rid, sl in zip(cache_manager.request_ids, seq_lens, strict=True):
            start = cache_manager._get_state(rid, _MAIN).position_id_start
            pos_ids_list.extend(range(start, start + sl))
        position_ids = torch.tensor(pos_ids_list, dtype=torch.long, device=device)

        return {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
            "position_ids": position_ids,
        }

    def _hidden(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        return self.language_model.model(input_ids, cache_handle, position_ids)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        hidden = self._hidden(input_ids, position_ids, cache_handle)
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
        cache_handle = engine_inputs.cache_manager
        sampler = engine_inputs.sampler
        cache_handle.set_active_label(_MAIN)

        hidden = self._hidden(input_ids, position_ids, cache_handle)

        if graph_walk == "prefill":
            qo_indptr_buf = cache_handle.get_qo_indptr_buf(_MAIN)
            assert qo_indptr_buf is not None, (
                "prefill forward_batched requires a CUDA-graph "
                "FlashInferPrefillWrapper (qo_indptr static buffer); got None."
            )
            last_token_indices = (qo_indptr_buf[1:] - 1).long()
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
        eos_token_id = self.config.eos_token_id
        ignore_eos = request_info.sampling_config["LLM"].ignore_eos
        if (not ignore_eos and eos_token_id == token) or (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        ):
            return {"decode_loop"}
        return set()
