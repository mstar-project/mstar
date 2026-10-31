"""The text LM node: a packed prefill that samples each prompt's first token, then one token per
request per decode step; or, under a ``DraftConfig``, a verify step per decode step that keeps
the drafts the target agrees with (``draft_verify_step``)."""
from __future__ import annotations

from collections.abc import Mapping
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
from mstar.engine.resources import (
    DRAFT_VERIFY,
    AttentionStep,
    DraftConfig,
    KVStep,
    PositionStep,
    ProposeStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    draft_verify_step,
    with_draft_verify,
)
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
        draft: DraftConfig | None = None,
        drafter: nn.Module | None = None,
        drafter_decl=None,
    ):
        super().__init__()
        self.lm = lm
        self.stop_token_ids = frozenset(stop_token_ids)
        self.accelerator_graph = accelerator_graph
        self.prefill_token_buckets = list(prefill_token_buckets)
        # a stop id ends decode, or the reply at the prefill; the budget is the loop's cap (a
        # verify step's stop is the engine's)
        self.token_stop = TokenStop(
            loop=DECODE_LOOP, buffer="new_token", stop_ids=self.stop_token_ids, sampler=SAMPLER,
            prefill_walks=frozenset({PREFILL_WALK}),
        )
        # Draft verify (speculative decoding): each decode step runs the request's last kept
        # tokens and k drafts, from the walk's engine-side proposer (n-gram) or from
        # `drafter`, a drafter in the node (`mstar.model.components.drafters`) that
        # `drafter_decl` declares.
        self.draft = draft
        self.drafter = drafter
        self.drafter_decl = drafter_decl
        fields = {} if drafter_decl is None else drafter_decl.policy_fields()
        self._policy = None if draft is None else draft.policy(stop_ids=self.stop_token_ids, loop=DECODE_LOOP, **fields)
        # the tokens a verify step's input carries
        self._carry = 1 if drafter_decl is None else drafter_decl.carry_width()

    def get_draft_policy(self, graph_walk: str):
        return self._policy if graph_walk == DECODE_WALK else None

    def _verifies(self, graph_walk: str) -> bool:
        return self._policy is not None and graph_walk == DECODE_WALK

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if not self.accelerator_graph:
            return []
        if self._policy is None:
            configs = super().get_accelerator_graph_configs(device, tp_world_size)
        else:
            # a verify step of a block a row, the drafter's passes and the verdict in one graph
            configs = [BatchedAcceleratorGraphConfig(
                capture_graph_walk=DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(self._carry, dtype=torch.long, device=device),
                    input_seq_len=self._policy.block,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            )]
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
        if self._verifies(graph_walk):
            # the last kept tokens; the step verifies the last of them and the drafts
            return ARNodeInputs(input_seq_len=self._policy.block, input_ids=ids[-self._carry:])
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
        segments = [Segment(request_id=rid, label=_MAIN, span=inp.input_seq_len) for rid, inp in rows]
        steps = {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True), ROPE: PositionStep()}
        if self._policy is None:
            # the prompt's ids count toward a repetition penalty
            steps[SAMPLER] = SamplerStep(
                apply_penalty=True,
                prefill_tracked_tokens={rid: inp.input_ids for rid, inp in rows} if graph_walk == PREFILL_WALK else {},
            )
            return SubmoduleStep(segments=segments, steps=steps)
        if self.drafter_decl is not None:
            steps.update(
                self.drafter_decl.verify_steps(inputs[0].input_seq_len - 1 if inputs else None)
                if graph_walk == DECODE_WALK
                else self.drafter_decl.prefill_steps(request_ids, [inp.input_seq_len for inp in inputs])
            )
        if graph_walk == DECODE_WALK:
            return with_draft_verify(SubmoduleStep(segments=segments, steps=steps), self._policy, DRAFT_VERIFY, SAMPLER)
        steps[SAMPLER] = SamplerStep(apply_penalty=False)
        if self._policy.proposer is not None:
            # the prompt is the history the engine-side proposer looks drafts up in
            steps[self._policy.proposer] = ProposeStep(history={rid: inp.input_ids for rid, inp in rows})
        return SubmoduleStep(segments=segments, steps=steps)

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        if self._verifies(graph_walk):
            return {"input_ids": torch.stack([inp.input_ids for inp in inputs])}
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        if self.drafter_decl is not None and graph_walk == PREFILL_WALK:
            # what the drafter's prefill takes beside the target's
            out["drafter_inputs"] = self.drafter_decl.prefill_inputs([inp.input_ids for inp in inputs])
        return out

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
        if self._policy is not None:
            raise NotImplementedError("a drafting text LM node runs batched")
        return {"new_token": [self._sample(graph_walk, engine_inputs, input_ids)]}

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.logits(self.lm(self.lm.embed(input_ids), label=_MAIN))

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        drafter_inputs: dict | None = None, **kwargs,
    ) -> BatchedModelOutput:
        if self._verifies(graph_walk):
            # the next step's inputs; the engine emits the kept tokens as new_token
            carry = draft_verify_step(
                engine_inputs.resources[DRAFT_VERIFY], list(engine_inputs.request_ids), input_ids, self._logits,
                self.drafter,
            )
            return BatchedModelOutput(row_outputs={"text_inputs": carry})
        tokens = self._sample(graph_walk, engine_inputs, input_ids)
        if drafter_inputs is not None:
            # the drafter fills its cache from the prompts (a DSpark drafter from the taps the
            # target's prefill just wrote); a drafter whose verify step takes more than the
            # sample hands the decode loop its first inputs
            carry = self.drafter.prefill(tokens=tokens, **drafter_inputs)
            if carry is not None:
                return BatchedModelOutput(
                    row_outputs={"new_token": tokens, "decode_input": carry}, check_stop_buffers={"new_token": tokens},
                )
        # row i is request i: one clone and one D2H copy per step, not one per request
        return BatchedModelOutput(row_outputs={"new_token": tokens}, check_stop_buffers={"new_token": tokens})

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], **kwargs,
    ):
        if self._verifies(request_info.graph_walk):
            # a verify step's loop-back is its forward's, and new_token the engine's cut
            return
        super().postprocess(request_id, request_info, outputs, **kwargs)
