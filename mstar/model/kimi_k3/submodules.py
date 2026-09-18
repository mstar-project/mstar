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
from mstar.model.kimi_k3.dspark.model import DSPARK_ATTN, DSPARK_KV, DSparkDraft
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
                 mixed_prefill_decode: bool = False, speculative_tokens: int = 0, draft: DSparkDraft | None = None):
        super().__init__()
        # speculative decoding (plan sections 8.5, 8.6): a decode row arrives with its bonus token
        # (one id, as the conductor hands the prefill's token to the decode loop), the step drafts
        # k tokens for it, the target verifies [bonus, drafts] in one pass, `sample_verify` keeps
        # the accepted prefix plus a new bonus. 0 = one token per step. Without the DSpark draft the
        # drafts are the bonus token repeated (`_draft`), which exercises the whole verify path
        # with near-zero acceptance.
        self.speculative_tokens = int(speculative_tokens)
        self.k1 = self.speculative_tokens + 1
        self._acceptance = None  # the SpecAcceptance resource, bound with the node's resources
        # the DSpark draft (plan section 8.6): drafts at the start of a step from the row's bonus
        # token against its own context cache, which the end of the step extends from the target's
        # aux states. Without it, the stand-in draft and k + 1 ids per decode row.
        self.draft = draft
        if draft is not None:
            assert self.speculative_tokens > 0, "a draft needs speculative_tokens > 0"
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
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=self.k1,
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
        # a speculating decode row carries its bonus token and spans k + 1 tokens (it and the
        # drafts the step verifies)
        span = self.k1 if self.speculative_tokens > 0 and graph_walk == "decode" else ids.shape[0]
        return ARNodeInputs(input_ids=ids, input_seq_len=span)

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs,
    ):
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        steps = {
            MLA_KV: KVStep(),
            MLA_ATTN: AttentionStep(causal=True),
            KDA_STATE: RecurrentStep(),
            KDA_ATTN: LinearAttnStep(),
            SAMPLER: SamplerStep(apply_penalty=False),
        }
        if self.speculative_tokens > 0:
            steps[SPEC] = SpecStep()
        if self.draft is not None:
            # the draft cache appends the rows' spans (the prompt, then k + 1 context entries per
            # step); in a decode step the draft's k queries per row attend to the stored context alone
            steps[DSPARK_KV] = KVStep()
            if graph_walk == "decode":
                steps[DSPARK_ATTN] = AttentionStep(
                    causal=False, context_only=True,
                    segments=tuple(Segment(request_id=rid, label="main", span=self.speculative_tokens) for rid in request_ids),
                )
        return SubmoduleStep(segments=segments, steps=steps)

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {"text_inputs": torch.cat([inp.input_ids for inp in inputs])}

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn = engine_inputs.resources[MLA_ATTN]
        if self.draft is not None:
            # the prefill also writes the draft's context KV for the whole prompt (each row's
            # positions count from 0) so the first decode step drafts against a complete context
            context = self.draft.context_accumulator()
            hidden, _ = self.language_model.model(
                self.embed_tokens(text_inputs), label="main", aux_layers=self.draft.cfg.target_layer_ids,
                aux_sink=context)
            qo = attn.qo_indptr_buf().long()
            rows = torch.repeat_interleave(torch.arange(qo.numel() - 1, device=qo.device), qo[1:] - qo[:-1])
            positions = torch.arange(text_inputs.shape[0], device=qo.device) - qo[rows]
            self.draft.write_context(context.finish(), positions)
        else:
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
        """One speculative decode step (plan section 8.6). The rows carry their bonus tokens (one
        id each); the draft proposes ``k`` tokens per row (the DSpark draft against its context cache,
        or the stand-in's repeated bonus); the target verifies ``[bonus, drafts]`` in one pass
        (greedy); the acceptance counts are staged for the next plan and the KDA prefix lengths; with
        the DSpark draft the context KV of the block's ``k + 1`` positions is written from this pass's
        aux states. Batch-wide sentinels of static shape; ``unpack_packed_outputs`` cuts them per
        request post-replay, the new bonus being the next row's id."""
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        acceptance: SpecAcceptance = engine_inputs.resources[SPEC]
        kda = engine_inputs.resources[KDA_ATTN]
        pool = engine_inputs.resources[KDA_STATE]
        k = self.speculative_tokens
        bonus = text_inputs.view(-1)
        if self.draft is not None:
            ctx_len = engine_inputs.resources[DSPARK_ATTN].kv_len_buf()[: bonus.shape[0]]
            offsets = torch.arange(self.k1, device=bonus.device)
            drafts = self.draft.draft(bonus, (ctx_len[:, None] + offsets[None, :k]).reshape(-1), k)
        else:
            drafts = self._draft(bonus[:, None])
        ids = torch.cat([bonus[:, None], drafts], dim=1)
        if self.draft is not None:
            context = self.draft.context_accumulator()
            hidden, _ = self.language_model.model(
                self.embed_tokens(ids.reshape(-1)), label="main", aux_layers=self.draft.cfg.target_layer_ids,
                aux_sink=context)
        else:
            hidden = self.language_model.model(self.embed_tokens(ids.reshape(-1)), label="main")
        logits = self.lm_head(hidden)
        tokens, accepted = sampler.sample_verify(engine_inputs.request_ids, logits, ids[:, 1:])
        acceptance.stage(accepted, tokens)
        kda.set_prefix_len(pool.block("spec_len", 0), accepted)
        new_bonus = tokens.gather(1, accepted.to(torch.long).unsqueeze(1))
        if self.draft is not None:
            positions = (ctx_len[:, None] + offsets[None, :]).reshape(-1)
            self.draft.write_context(context.finish(), positions)
        return {"spec_tokens": tokens, "spec_accepted": accepted, "next_inputs": new_bonus}

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, text_inputs: torch.Tensor, **kwargs):
        if self.speculative_tokens > 0 and graph_walk == "decode":
            return self._forward_verify(engine_inputs, text_inputs)
        new_token = self._forward(graph_walk, engine_inputs, text_inputs)
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
        # the prefill's token reaches the decode loop through the conductor (the persisted
        # `new_token` becomes the first `text_inputs`), one id per row, as every later row
        return {rid: {"new_token": [new_tokens[i : i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def unpack_packed_outputs(self, static_output: dict, request_ids: list[str], real_seq_lens: list[int],
                              inputs: list, per_request_info: dict) -> dict[str, dict[str, list[torch.Tensor]]]:
        """A verify step's per-request outputs: ``new_token`` = the accepted tokens and the new
        bonus (``accepted + 1`` of the row's ``k1`` verified tokens, cut after the first stop token
        unless the request ignores EOS), ``text_inputs`` = the next row. Waits for the step's
        verdicts (recorded mid-step, so the wait overlaps the draft). ``new_token`` is built on the
        host from the verdict's pinned mirror, so no device copy is needed for it; ``text_inputs``
        is a view of the step's static row like the plain path's sampled token (the capture slots
        alternate, and the row is consumed before its slot is replayed again). Cloning both per
        request cost two device copies and two host reads per request per step."""
        if "spec_tokens" not in static_output:
            return {}
        acceptance: SpecAcceptance = self._acceptance
        acceptance.note_step(request_ids)
        verdicts = acceptance.verdicts_for(request_ids)
        dtype, nxt = static_output["spec_tokens"].dtype, static_output["next_inputs"]
        out = {}
        for i, rid in enumerate(request_ids):
            info = per_request_info.get(rid) if per_request_info else None
            ignore_eos = info is None or info.resource_configs[SAMPLER].ignore_eos
            emitted = self._emitted(verdicts[i], ignore_eos)
            out[rid] = {"new_token": [torch.tensor(emitted, dtype=dtype)], "text_inputs": [nxt[i]]}
        return out

    def _emitted(self, verdict, ignore_eos: bool) -> list[int]:
        """The tokens a verify step emits for a row: the accepted drafts and the bonus, cut after
        the first stop token unless the request ignores EOS."""
        n = verdict.accepted + 1
        if not ignore_eos:
            for j, t in enumerate(verdict.tokens[:n]):
                if t in self.config.stop_token_ids:
                    n = j + 1
                    break
        return list(verdict.tokens[:n])

    def bind_node_resources(self, resources: dict) -> None:
        super().bind_node_resources(resources)
        self._acceptance = resources.get(SPEC)
        if self.draft is not None:
            self.draft.bind_resources(resources)

    def postprocess(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict, **kwargs):
        if "new_token" in outputs and "text_inputs" not in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict) -> set[str]:
        if "new_token" not in outputs:
            return set()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if self.speculative_tokens > 0:
            # several tokens per step: tally what was emitted (already cut at a stop token by
            # `unpack_packed_outputs`); a stop token among them ends the loop
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
