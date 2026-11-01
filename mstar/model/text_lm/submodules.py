"""The text LM node: a packed prefill that samples each prompt's first token, then one token per
request per decode step."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    TokenStop,
)
from mstar.model.text_lm.config import ATTN, DECODE_LOOP, DECODE_WALK, KV_CACHE, PREFILL_WALK, ROPE, SAMPLER

_MAIN = "main"


class TextLMSubmodule(TokenStepNode, ARNodeSubmodule):
    """``lm`` embeds, runs its layers against the node's KV / attention / position resources
    under one label, and projects to logits. Decode is captured per batch size; prefill per
    packed token count, unless ``prefill_token_buckets`` is empty."""

    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    SAMPLER = SAMPLER
    # prefill's batch is capped by its captures (and uncapped eager)
    MAX_PREFILL_BATCH_SIZE = None
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # Only the eager forwards skip torch.compile: the decode capture compiles at the deployment's
    # level (`mstar serve -O`), and the prefill captures replay eager kernels.
    disable_torch_compile = True

    def __init__(
        self,
        lm: nn.Module,
        stop_token_ids: frozenset[int],
        *,
        accelerator_graph: bool = True,
        prefill_token_buckets: list[int] | tuple[int, ...] = (),
    ):
        super().__init__()
        self.lm = lm
        self.stop_token_ids = frozenset(stop_token_ids)
        self.accelerator_graph = accelerator_graph
        self.prefill_token_buckets = list(prefill_token_buckets)
        # a stop id ends decode, or the reply at the prefill; the budget is the loop's cap
        self.token_stop = TokenStop(
            loop=DECODE_LOOP, buffer="new_token", stop_ids=self.stop_token_ids, sampler=SAMPLER,
            prefill_walks=frozenset({PREFILL_WALK}),
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if not self.accelerator_graph:
            return []
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        if self.prefill_token_buckets:
            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n, input_ids=torch.zeros(n, dtype=torch.long, device=device),
                ),
                capture_token_lengths=self.prefill_token_buckets,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                compile=False,
            ))
        return configs

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0].to(self.get_device()).reshape(-1)
        return ARNodeInputs(input_seq_len=ids.shape[0], input_ids=ids)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        rows = list(zip(request_ids, inputs, strict=True))
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label=_MAIN, span=inp.input_seq_len) for rid, inp in rows],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                ROPE: PositionStep(),
                # the prompt's ids count toward a repetition penalty
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens={rid: inp.input_ids for rid, inp in rows}
                    if graph_walk == PREFILL_WALK else {},
                ),
            },
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}

    def _sample(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor):
        hidden = self.lm(self.lm.embed(input_ids), label=_MAIN)
        if graph_walk == PREFILL_WALK:
            hidden = engine_inputs.resources[ATTN].select_last_hidden(hidden, label=_MAIN)
        return self._draw(engine_inputs, self.lm.logits(hidden))

    # Keyed by request id, so outside dynamo (the sampler runs eager either way).
    @torch.compiler.disable
    def _draw(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return engine_inputs.resources[SAMPLER].sample(list(engine_inputs.request_ids), logits=logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._sample(graph_walk, engine_inputs, input_ids)]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, **kwargs,
    ) -> BatchedModelOutput:
        tokens = self._sample(graph_walk, engine_inputs, input_ids)
        # row i is request i: one clone and one D2H copy per step, not one per request
        return BatchedModelOutput(row_outputs={"new_token": tokens}, check_stop_buffers={"new_token": tokens})
