"""The dense Qwen3 ``LLM`` node: a packed prefill that samples each prompt's
first token, then a decode loop."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.components.token_step import TokenStepNode
from mstar.model.qwen3.config import ATTN, DECODE_LOOP, DECODE_WALK, KV_CACHE, PREFILL_WALK, ROPE, SAMPLER
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    ModelInputsFromEngine,
    TokenStop,
)

_LABEL = "main"


def token_stop(stop_token_ids) -> TokenStop:
    """A stop id ends the decode loop, or the reply at the prefill. The graph
    runtime checks a decode step's."""
    return TokenStop(
        loop=DECODE_LOOP, buffer="new_token", stop_ids=frozenset(stop_token_ids), sampler=SAMPLER,
        walk=DECODE_WALK, prefill_walks=frozenset({PREFILL_WALK}), runtime_check=True,
    )


# the decode walk reads each row's last token off the sampler's slot masters
DEVICE_LOOPBACK = DeviceLoopback(walk=DECODE_WALK, signal="text_inputs", sampler=SAMPLER, buffer="new_token")


class Qwen3LMSubmodule(TokenStepNode, ARNodeSubmodule):
    """``lm`` embeds, runs its layers against the node's resources and projects
    to logits."""

    SAMPLER = SAMPLER
    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    PREFILL_WALK = PREFILL_WALK
    # a prefill's batch is capped by its captures (and uncapped eager)
    MAX_PREFILL_BATCH_SIZE = None
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def __init__(
        self,
        lm: nn.Module,
        stop_token_ids,
        *,
        accelerator_graph: bool = True,
        prefill_token_buckets: list[int] | None = None,
    ):
        super().__init__(prefill_token_buckets)
        self.lm = lm
        self.accelerator_graph = accelerator_graph
        self.token_stop = token_stop(stop_token_ids)
        self.device_loopback = DEVICE_LOOPBACK

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if not self.accelerator_graph:
            return []

        def ids(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=ids,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens
        return frozenset({DECODE_WALK})

    def step_is_input_free(self, graph_walk: str) -> bool:
        return graph_walk == DECODE_WALK

    def speculative_input_seq_len(self, graph_walk, fwd_info, prev_seq_len):
        if graph_walk == DECODE_WALK:
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
        has_loopback: bool = False, **kwargs: Any,
    ) -> ARNodeInputs:
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # read off the sampler's slot master in `preprocess`
            return row
        ids = inputs["text_inputs"][0].reshape(-1)
        if graph_walk == DECODE_WALK:
            # a row's first decode step brings the prefill's token routed
            return ARNodeInputs(input_ids=ids[-1:], input_seq_len=1)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs], **kwargs,
    ) -> SubmoduleStep:
        segments = [Segment(rid, _LABEL, inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)]
        steps = {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            ROPE: PositionStep(),
        }
        # the prompt's ids count toward a repetition penalty
        tracked = {rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)} \
            if graph_walk == PREFILL_WALK else {}
        steps[SAMPLER] = SamplerStep(apply_penalty=True, prefill_tracked_tokens=tracked)
        return SubmoduleStep(segments=segments, steps=steps)

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        if graph_walk == DECODE_WALK and (engine_inputs.captured or any(inp.input_ids is None for inp in inputs)):
            # some row's token is on the device (a captured step's sampler always keeps the
            # slot masters, and its padding rows read the trash row)
            return {"input_ids": self.loopback_inputs(engine_inputs, inputs)["input_ids"]}
        # every row brought its token or prompt routed: an eager node keeps no slot masters
        return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}

    # Keyed by request id, so outside dynamo (the sampler runs eager either way).
    @torch.compiler.disable
    def _draw(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return engine_inputs.resources[SAMPLER].sample(list(engine_inputs.request_ids), logits=logits)

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, **kwargs):
        out = self.forward_batched(graph_walk, engine_inputs, input_ids)
        return {"new_token": [out.row_outputs["new_token"]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor, **kwargs,
    ) -> BatchedModelOutput:
        hidden = self.lm(self.lm.embed(input_ids), label=_LABEL)
        if graph_walk == PREFILL_WALK:
            hidden = engine_inputs.resources[ATTN].select_last_hidden(hidden, label=_LABEL)
        tokens = self._draw(engine_inputs, self.lm.logits(hidden))
        # row i is request i: one clone and one D2H copy a step. The prefix cache keys a
        # reply from this host copy, under the `text_inputs` its PrefixKeys name.
        return BatchedModelOutput(
            row_outputs={"new_token": tokens},
            check_stop_buffers={"new_token": tokens, "text_inputs": tokens},
        )
