"""The Kimi K3 text node (``LLM``): one ``ARNodeSubmodule`` serving the ``prefill`` and
``decode`` walks. It declares the step over four resources (paged MLA latent cache, MLA
attention, KDA recurrent state, sampler) and runs the packed forward.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    RecurrentStateStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.kimi_k3.config import KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER, KimiK3Config
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs

logger = logging.getLogger(__name__)


class KimiK3LLMSubmodule(ARNodeSubmodule):
    # Built in explicit dtypes (bf16 activations, fp32 gates and router, packed experts in the
    # kernel layouts): the engine must neither re-cast it -- ``Module.to(dtype)`` would turn the
    # E8M0 expert scales into bf16 and back, doubling them and leaving the expert backend on
    # stale copies (12 GiB per rank on pruned75) -- nor run its forward under autocast.
    disable_autocast = True
    # the kernels are hand-fused and CUDA-graphed; inductor autotuning breaks on their shapes
    disable_torch_compile = True
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self, language_model: nn.Module, config: KimiK3Config, cuda_graphs: bool = True,
                 max_capture_batch_size: int | None = None, max_prefill_batch_size: int | None = 8,
                 mixed_prefill_decode: bool = False):
        super().__init__()
        # prefill runs eagerly and its transient memory grows with the tokens in the step: the
        # attention-residual stack alone is [tokens, blocks, hidden] (about 1 GB per 8k tokens at
        # K3 width), so the scheduler is asked to split prefills beyond this many requests
        self.max_prefill_batch_size = max_prefill_batch_size
        # let the decoding requests ride along in prefill steps (one-token rows next to the packed
        # prompts) instead of stalling for the length of every prefill; the varlen kernels and the
        # paged attention take the mixed spans as they are
        self.mixed_prefill_decode = mixed_prefill_decode
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.config = config
        # the torch reference KDA kernel addresses state slots from the host, so it must
        # run eagerly; the fla/fused kernels take device index tensors and can be captured
        self.cuda_graphs = cuda_graphs
        # deployments cap the decode buckets at their KDA slot count: a bucket wider than the
        # number of resident requests can only ever replay with padding rows
        self.capture_batch_sizes = [b for b in self.DECODE_CAPTURE_BATCH_SIZES
                                    if max_capture_batch_size is None or b <= max_capture_batch_size]

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        if not self.cuda_graphs:
            return []
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1,
                ),
                capture_batch_sizes=self.capture_batch_sizes, compile=False),
            # no prefill capture: the KDA varlen conv/chunk kernels size work on the host
            # (fla's repeat_interleave), which CUDA streams refuse while capturing; prefill
            # runs eager on FlashKDA
        ]

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0].reshape(-1)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs,
    ):
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                MLA_KV: KVStep(),
                MLA_ATTN: AttentionStep(causal=True),
                KDA_STATE: RecurrentStateStep(),
                SAMPLER: SamplerStep(apply_penalty=False),
            },
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {"text_inputs": torch.cat([inp.input_ids for inp in inputs])}

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn = engine_inputs.resources[MLA_ATTN]
        hidden = self.language_model.model(self.embed_tokens(text_inputs), label="main")
        if graph_walk == "prefill":
            hidden = attn.select_last_hidden(hidden)
        logits = self.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor, **kwargs):
        return {"new_token": self._forward(graph_walk, engine_inputs, text_inputs)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        """Requests per step: prefill is bounded (see ``max_prefill_batch_size``), decode by the
        captured graph buckets (the engine takes the smaller cap)."""
        return self.max_prefill_batch_size if graph_walk == "prefill" else None

    def mixed_step_walks(self, graph_walk: str) -> set[str]:
        """Decode rows may join a prefill step (``mixed_prefill_decode``); they add one token each
        to the packed batch and their outputs are routed as decode-loop outputs."""
        return {"decode"} if self.mixed_prefill_decode and graph_walk == "prefill" else set()

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor, **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(graph_walk, engine_inputs, text_inputs)
        return {rid: {"new_token": [new_tokens[i : i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def postprocess(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict, **kwargs):
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = int(outputs["new_token"][0].item())
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        # tokens emitted so far: one from the prefill node plus one per finished decode
        # iteration (the loop index is 0-based), so max_tokens means max_tokens tokens
        n_done = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and token in self.config.stop_token_ids) or n_done >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
