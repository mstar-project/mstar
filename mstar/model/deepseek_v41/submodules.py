"""AR submodule for the DeepSeek-V4.1 text backbone on the resource-pool engine."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import KVStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.csa.config import CSAStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.deepseek_v41.config import CSA, LABEL, SAMPLER, STATE, DeepseekV41Config
from mstar.model.submodule_base import (
    END_PARTITION,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


class DeepseekV41LLMSubmodule(ARNodeSubmodule):
    """Embed, the hyper-connected blocks and the LM head, one TP node."""

    # the reference numerics run in fp32 on purpose in many places, which the engine's bf16
    # autocast and a compile of the eager forward would both undo
    disable_torch_compile: bool = True
    disable_autocast: bool = True

    def __init__(self, language_model: nn.Module, config: DeepseekV41Config) -> None:
        super().__init__()
        self.language_model = language_model  # DeepseekV41ForCausalLM
        self.config = config
        self._long_stager = PinnedStager(torch.long)

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        # the CSA resource is the torch reference, which cannot be captured: every step is eager
        return []

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one STATE slot per request: the scheduler splits a larger step rather than fail its admit
        pool = self.node_resources.get(STATE) if self.node_resources else None
        if pool is None:
            return None
        return pool.num_free_slots if graph_walk == "prefill" else pool.config.usable_slots

    def batch_cap_varies(self, graph_walk: str) -> bool:
        return graph_walk == "prefill"

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts, which would round the fp32 parameters."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info("DeepseekV41LLMSubmodule: ignoring post-load dtype cast to %s", dtype)
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        state = self.request_state(fwd_info.rid_handle)
        context = state.get("context", 0)
        if context + seq_len > self.config.max_seq_len:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {context + seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}")
        if graph_walk == "prefill":
            state.add("prompt_len", context + seq_len)
        return ARNodeInputs(input_ids=text_inputs, input_seq_len=seq_len)

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
            prefill_tokens = {rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)}
        segments = [
            Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        # a cache advances by the groups the step's tokens complete, which the CSA resource counts
        caches = self.node_resources[CSA].kv_segments(segments)
        return SubmoduleStep(
            segments=segments,
            steps={
                **{key: KVStep(segments=rows) for key, rows in caches.items()},
                CSA: CSAStep(),
                STATE: RecurrentStep(),
                SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        if graph_walk == "prefill":
            # a prefill samples each request's last row: the head runs on those alone
            last, end = [], 0
            for inp in inputs:
                end += inp.input_seq_len
                last.append(max(end - 1, 0))
            rows = torch.empty(len(last), dtype=torch.long, device=out["input_ids"].device)
            self._long_stager.copy_(rows, last)
            out["last_rows"] = rows
        return out

    def _forward(
        self, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.language_model(input_ids, last_rows)
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(engine_inputs, input_ids, last_rows)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        last_rows: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList] | BatchedModelOutput:
        new_tokens = self._forward(engine_inputs, input_ids, last_rows)
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs,
    ):
        # the step ran, so its tokens are the request's context now
        if inputs is not None:
            state = self.request_state(request_id)
            state.add("context", state.get("context", 0) + inputs.input_seq_len)
        if "new_token" in outputs:
            # the decode loop's text_inputs edge carries the emitted token
            outputs.setdefault("text_inputs", outputs["new_token"])

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        # check_stop gets host tensors, so .item() here does not sync
        return self._stops(request_id, request_info, outputs["new_token"][0].item())

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        stops: dict[str, set[str]] = {}
        for rid, token in pairs:
            if loops := self._stops(rid, request_infos[rid], token):
                stops[rid] = loops
        return stops

    def _stops(self, request_id: str, request_info: CurrentForwardPassInfo, token: int) -> set[str]:
        """The loops a step's emitted ``token`` stops."""
        state = self.request_states.get(request_id)
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if request_info.graph_walk == "prefill":
            # no decode loop on this walk to stop: a stop token ends the partition
            return {END_PARTITION} if not ignore_eos and is_eos else set()
        # the prefill emitted one token and this is decode step iters + 1
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # the next step may already be scheduled when this stop lands, so stop while it still
        # fits the window; a next step that emits the reply's last token may fill it
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = (prompt_len + generated >= self.config.max_seq_len
                and generated + 1 < request_info.max_tokens)
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
