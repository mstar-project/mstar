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
    LinearAttnStep,
    RecurrentStep,
    SamplerStep,
    Segment,
    SlotLease,
    SpecStep,
    SubmoduleStep,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.speculative import SpecAcceptance
from mstar.model.kimi_k3.config import KDA_ATTN, KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER, SPEC, KimiK3Config
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
    # the per-step CPU work (scheduling, per-request outputs) hides behind the GPU step: +10-14%
    # decode throughput at every concurrency on pruned75 TP8, bit-identical outputs (2026-09-16)
    prefers_tp_async_scheduling = True
    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self, language_model: nn.Module, config: KimiK3Config, cuda_graphs: bool = True,
                 max_capture_batch_size: int | None = None, max_prefill_batch_size: int | None = 8,
                 mixed_prefill_decode: bool = False, speculative_tokens: int = 0):
        super().__init__()
        # speculative decoding (plan section 8.5): a decode row carries k + 1 ids (the bonus token
        # and k drafts), the target verifies them in one pass, `sample_verify` keeps the accepted
        # prefix plus a new bonus, and the next row is drafted. 0 = one token per step. Until the
        # DSpark draft lands the drafts are the bonus token repeated (`_draft`), which exercises the
        # whole verify path with near-zero acceptance.
        self.speculative_tokens = int(speculative_tokens)
        self.k1 = self.speculative_tokens + 1
        self._acceptance = None  # the SpecAcceptance resource, bound with the node's resources
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
                    input_ids=torch.zeros(self.k1, dtype=torch.long, device=device), input_seq_len=self.k1,
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
                KDA_STATE: RecurrentStep(),
                KDA_ATTN: LinearAttnStep(),
                SAMPLER: SamplerStep(apply_penalty=False),
                **({SPEC: SpecStep()} if self.speculative_tokens > 0 else {}),
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

    # ------------------------------------------------------------ speculation
    def _draft(self, bonus: torch.Tensor) -> torch.Tensor:
        """``bonus [bs, 1]`` -> ``drafts [bs, k]``. The stand-in until the DSpark draft: the bonus
        token repeated, accepted only where the target repeats itself."""
        return bonus.expand(-1, self.speculative_tokens)

    def _forward_verify(self, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """One verify step over decode rows of ``k1`` ids each (``[bonus, d1..dk]``): the target's
        logits at every position, greedy verification, the acceptance counts staged for the next
        plan and the KDA prefix lengths, the next rows drafted from the new bonus tokens. Batch-wide
        sentinels of static shape; ``unpack_packed_outputs`` cuts them per request post-replay."""
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        acceptance: SpecAcceptance = engine_inputs.resources[SPEC]
        kda = engine_inputs.resources[KDA_ATTN]
        pool = engine_inputs.resources[KDA_STATE]
        ids = text_inputs.view(-1, self.k1)
        hidden = self.language_model.model(self.embed_tokens(text_inputs), label="main")
        logits = self.lm_head(hidden)
        tokens, accepted = sampler.sample_verify(engine_inputs.request_ids, logits, ids[:, 1:])
        acceptance.stage(accepted)
        kda.set_prefix_len(pool.block("spec_len", 0), accepted)
        bonus = tokens.gather(1, accepted.to(torch.long).unsqueeze(1))
        next_inputs = torch.cat([bonus, self._draft(bonus)], dim=1)
        return {"spec_tokens": tokens, "spec_accepted": accepted, "next_inputs": next_inputs}

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor, **kwargs):
        if self.speculative_tokens > 0 and graph_walk == "decode":
            return self._forward_verify(engine_inputs, text_inputs)
        new_token = self._forward(graph_walk, engine_inputs, text_inputs)
        if self.speculative_tokens > 0:
            # a prefill under speculation also drafts the first decode row
            bonus = new_token.view(-1, 1)
            return {"new_token": new_token, "next_inputs": torch.cat([bonus, self._draft(bonus)], dim=1)}
        return {"new_token": new_token}

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
        if self.speculative_tokens > 0 and graph_walk == "decode":
            # batch-wide sentinels only: the per-request cut needs the accepted counts on the
            # host, which `unpack_packed_outputs` reads after the replay
            return self._forward_verify(engine_inputs, text_inputs)
        new_tokens = self._forward(graph_walk, engine_inputs, text_inputs)
        out = {rid: {"new_token": [new_tokens[i : i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}
        if self.speculative_tokens > 0:
            bonus = new_tokens.view(-1, 1)
            next_inputs = torch.cat([bonus, self._draft(bonus)], dim=1)
            for i, rid in enumerate(engine_inputs.request_ids):
                out[rid]["text_inputs"] = [next_inputs[i]]
        return out

    def unpack_packed_outputs(self, static_output: dict, request_ids: list[str], real_seq_lens: list[int],
                              inputs: list, per_request_info: dict) -> dict[str, dict[str, list[torch.Tensor]]]:
        """A verify step's per-request outputs: ``new_token`` = the accepted tokens and the new
        bonus (``accepted + 1`` of the row's ``k1`` verified tokens), ``text_inputs`` = the next row.
        Waits for the step's acceptance counts (recorded mid-step, so the wait overlaps the draft);
        the slices are cloned off the static buffers."""
        if "spec_tokens" not in static_output:
            return {}
        acceptance: SpecAcceptance = self._acceptance
        acceptance.note_step(request_ids)
        counts = acceptance.accepted_for(request_ids)
        tokens, nxt = static_output["spec_tokens"], static_output["next_inputs"]
        return {
            rid: {"new_token": [tokens[i, : counts[i] + 1].clone()], "text_inputs": [nxt[i].clone()]}
            for i, rid in enumerate(request_ids)
        }

    def bind_node_resources(self, resources: dict) -> None:
        super().bind_node_resources(resources)
        self._acceptance = resources.get(SPEC)

    def postprocess(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict, **kwargs):
        if "new_token" in outputs and "text_inputs" not in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict) -> set[str]:
        if "new_token" not in outputs:
            return set()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if self.speculative_tokens > 0:
            # several tokens per step: tally what was emitted; a stop token anywhere in them stops
            # (the few tokens after it in the same step still reach the client: 4a limitation)
            emitted = outputs["new_token"][0].reshape(-1).tolist()
            state = self.request_state(request_id)
            n_done = state.get("emitted", 0) + len(emitted)
            state.add("emitted", n_done)
            hit = not ignore_eos and any(t in self.config.stop_token_ids for t in emitted)
            return {"decode_loop"} if hit or n_done >= request_info.max_tokens else set()
        token = int(outputs["new_token"][0].item())
        # tokens emitted so far: one from the prefill node plus one per finished decode
        # iteration (the loop index is 0-based), so max_tokens means max_tokens tokens
        n_done = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and token in self.config.stop_token_ids) or n_done >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
