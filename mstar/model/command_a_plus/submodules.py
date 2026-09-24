"""Packed prefill and one-token-per-request decode for Command A+."""

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.command_a_plus.components.language_model import CommandAPlusForCausalLM
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    SAMPLER,
    CommandAPlusTextConfig,
)
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine


class CommandAPlusLLMSubmodule(ARNodeSubmodule):
    # Start with eager execution. Model-level compilation/capture gets its own
    # parity checks after the full generation path works.
    disable_torch_compile = True

    def __init__(self, language_model: CommandAPlusForCausalLM, config: CommandAPlusTextConfig):
        super().__init__()
        self.language_model = language_model
        self.config = config

    @staticmethod
    def _check_walk(graph_walk: str) -> None:
        if graph_walk not in ("prefill", "decode"):
            raise ValueError(f"Unknown Command A+ graph walk: {graph_walk!r}")

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        self._check_walk(graph_walk)
        tokens = inputs["text_inputs"]
        if len(tokens) != 1 or tokens[0].ndim != 1 or tokens[0].numel() == 0:
            raise ValueError("text_inputs must contain one nonempty 1D token tensor")
        ids = tokens[0]
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("text_inputs must contain integer token IDs")
        if graph_walk == "decode" and ids.numel() != 1:
            raise ValueError("decode requires exactly one token per request")
        ids = ids.to(device=self.get_device(), dtype=torch.long)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.numel())

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs,
    ) -> SubmoduleStep:
        self._check_walk(graph_walk)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                LOCAL_ATTN: AttentionStep(causal=True),
                GLOBAL_ATTN: AttentionStep(causal=True),
                ROPE: PositionStep(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens={
                        rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
                    } if graph_walk == "prefill" else {},
                ),
            },
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        self._check_walk(graph_walk)
        return {"text_inputs": torch.cat([inp.input_ids for inp in inputs])}

    def _sample(self, graph_walk, engine_inputs, text_inputs):
        self._check_walk(graph_walk)
        embeddings = self.language_model.model.embed_tokens(text_inputs)
        hidden = self.language_model(embeddings, label="main")
        if graph_walk == "prefill":
            # Both managers have the same packing. Pick the final real prompt
            # token for each request, never the last row of the whole batch.
            hidden = engine_inputs.resources[LOCAL_ATTN].select_last_hidden(hidden, label="main")
        logits = self.language_model.compute_logits(hidden)
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._sample(graph_walk, engine_inputs, text_inputs)]}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor, **kwargs,
    ) -> dict[str, NameToTensorList]:
        tokens = self._sample(graph_walk, engine_inputs, text_inputs)
        return {
            rid: {"new_token": [tokens[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        if "new_token" not in outputs:
            return
        if request_info.graph_walk == "prefill":
            # The conductor sees tensor pointers, not token values. Decide
            # once at prefill whether to publish a decode seed. This read may
            # synchronize CUDA; decode's hot path remains metadata-only.
            ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
            is_eos = outputs["new_token"][0].item() == self.config.eos_token_id
            if request_info.max_tokens > 1 and (ignore_eos or not is_eos):
                outputs["decode_input"] = outputs["new_token"]
        else:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id, request_info, outputs) -> set[str]:
        if request_info.graph_walk != "decode" or "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        # The loop index is zero-based; include the prefill token and the
        # token produced by this decode iteration.
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and token == self.config.eos_token_id) or generated >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
