"""The ``LLM`` node of Kimi K3 / K3.1: embedding, the hybrid trunk and the LM head, one TP node.

Prefill and decode share the forward; per step it declares the latent KV cache, the MLA
attention over it, the KDA recurrent pool and the KDA resource, the sampler, and under
``dsa_long_context`` the DSA resource. Decode is captured per batch size (the routed experts on
CUDA take the fused kernels: bf16, fp8, or MXFP4 on Triton / FlashInfer's FP4 MoE); prefill is
eager.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.dsa.config import DSAStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.kimi_k3.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    KimiK3ModelConfig,
)
from mstar.model.submodule_base import (
    END_PARTITION,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
)

logger = logging.getLogger(__name__)


class KimiK3LLMSubmodule(ARNodeSubmodule):
    # the KDA resource and the AttnRes reads are compiler-disabled or untested under compile
    disable_torch_compile: bool = True
    # bf16 weights; the KDA state, the router, AttnRes and SiTU run fp32 on purpose, which the
    # engine's bf16 autocast would undo
    disable_autocast: bool = True

    def __init__(self, language_model: nn.Module, config: KimiK3ModelConfig,
                 decode_capture_batch_sizes: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.language_model = language_model  # KimiK3ForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        # decode graphs' batch sizes, capped by the KDA slots: the decode batch never passes the
        # largest one that fits, so a config with more slots names a longer ladder
        sizes = tuple(decode_capture_batch_sizes or DEFAULT_CAPTURE_BATCH_SIZES)
        if not sizes or list(sizes) != sorted(set(sizes)) or sizes[0] < 1:
            raise ValueError(f"decode_capture_batch_sizes must ascend from 1 up: {sizes}")
        self.decode_capture_batch_sizes = sizes

    # -- scheduling ---------------------------------------------------------

    def max_batch_size(self, graph_walk: str) -> int | None:
        """One KDA slot per row: the scheduler splits a larger step rather than fail its admit.
        Not the free slots alone: a chunked prompt's later chunks run in the slot its first took,
        so with every slot held mid-prefill a cap of the free ones would schedule none of them;
        the pool's admit refuses a new row past them, and the scheduler defers it."""
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        return None if pool is None else pool.config.usable_slots

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Decode, one graph per batch size (``decode_capture_batch_sizes``) up to the KDA slots.
        Not with dsa_long_context (the DSA
        resource's selection runs per request on the host), nor on MLA's torch fallback (its plan
        holds the capture's pages). Prefill stays eager."""
        if self.config.dsa_long_context:
            logger.warning("kimi_k3: dsa_long_context is eager only; decode runs eager")
            return []
        attn = self.node_resources.get(ATTN) if self.node_resources else None
        if device.type != "cpu" and attn is not None and not attn.uses_kernel:
            logger.warning("kimi_k3: MLA is on the torch fallback, which cannot be captured; decode runs eager")
            return []
        pool = self.node_resources.get(KDA_STATE) if self.node_resources else None
        slots = None if pool is None else pool.config.usable_slots
        sizes = [b for b in self.decode_capture_batch_sizes if slots is None or b <= slots]
        # compile=False: the forward hosts compiler-disabled resources (KDA, the MLA cache write)
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode",
            single_request_inputs=ARNodeInputs(
                input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1),
            capture_batch_sizes=sizes,
            compile=False,
        )]

    # -- dtype discipline ---------------------------------------------------

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts (the fp32 parameters are fixed at
        load, see ``restore_fp32_params``)."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info("KimiK3LLMSubmodule: ignoring post-load dtype cast to %s", dtype)
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    # -- engine seam ----------------------------------------------------------

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        state = self.request_state(fwd_info.rid_handle)
        context = state.get("context", 0)
        limit = self.config.context_limit
        if context + seq_len > limit:
            # fails this request only; the engine drops it from the batch
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {context + seq_len} exceeds the "
                f"{limit}-token limit this deployment serves"
                + ("" if self.config.dsa_long_context or not self.config.dsa else
                   " (index_topk, where dense MLA is exactly K3.1's DSA)"))
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
        steps = {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            KDA_STATE: RecurrentStep(),
            KDA: LinearAttnStep(),
        }
        if self.config.dsa_long_context:
            steps[DSA] = DSAStep()
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._logits(graph_walk, engine_inputs, input_ids)
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def _logits(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``(rows, vocab)``: each request's last token's logits in a prefill, every row's in a decode."""
        attn = engine_inputs.resources[ATTN]
        rows = None
        if graph_walk == "prefill":
            # each row samples its last token: the last layer's MLP and the head run only those
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
            rows = attn.select_last_hidden(positions, LABEL)
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        return self.language_model(input_ids, rows=rows)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(graph_walk, engine_inputs, input_ids)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids)
        # row i is request i: one clone and one host copy a step
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})

    # -- postprocess / stop ---------------------------------------------------

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], inputs: ARNodeInputs | None = None, **kwargs,
    ):
        # the step ran: its tokens are the request's context now
        if inputs is not None:
            state = self.request_state(request_id)
            state.add("context", state.get("context", 0) + inputs.input_seq_len)
        if "new_token" in outputs:
            outputs.setdefault("text_inputs", outputs["new_token"])

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        # check_stop gets host tensors: .item() does not sync
        return self._stops(request_id, request_info, outputs["new_token"][0].item())

    def check_stop_batched(
        self, request_ids: list[str], request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows, step_walk: str | None = None,
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
        state = self.request_states.get(request_id)
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if request_info.graph_walk == "prefill":
            # no decode_loop to stop on this walk: a stop token ends the partition here
            return {END_PARTITION} if not ignore_eos and is_eos else set()
        # 1 token from the prefill + (decode iters + 1)
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        # stop while the next step (possibly already scheduled) still fits the context limit
        prompt_len = state.get("prompt_len", 0) if state is not None else 0
        full = (prompt_len + generated >= self.config.context_limit
                and generated + 1 < request_info.max_tokens)
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens or full:
            return {"decode_loop"}
        return set()
