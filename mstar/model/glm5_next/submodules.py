"""AR submodule for the GLM-5.3-Flash text backbone on the resource-pool engine."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.cuda_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    SamplerStep,
    Segment,
    SlotLease,
    SlotStateStep,
    SubmoduleStep,
)
from mstar.model.glm5_next.config import (
    ATTN,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
)

logger = logging.getLogger(__name__)


class Glm5NextLLMSubmodule(ARNodeSubmodule):
    """Embed + 45 hybrid decoder layers + lm_head, one fat TP node (MTP off)."""

    # The eager prefill forward hosts the KDA span loop (@torch.compiler.disable
    # inside a compiled outer frame) and the fp8 reference MoE loop; the
    # post-capture torch.compile of the eager forwards is untested for it.
    disable_torch_compile: bool = os.environ.get("MSTAR_GLM53_TORCH_COMPILE", "0") != "1"

    def __init__(self, language_model: nn.Module, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.language_model = language_model  # Glm5NextForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config

    # -- CUDA-graph configs ----------------------------------------------

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the clamped fused fp8
        kernel (``moe_quant_kernel`` "triton"/"auto" on CUDA); False on the
        "reference" default, whose per-hit-expert loop hosts ``.nonzero()``
        and cannot be captured.
        """
        from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock

        lm = getattr(self, "language_model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm5NextSparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        return fp8_reference or naive_tp

    def _kda_max_slots(self) -> int | None:
        kda = self.node_resources.get(KDA_STATE) if self.node_resources else None
        return None if kda is None else kda.max_slots

    def max_batch_size(self, graph_walk: str) -> int | None:
        # A step can hold at most one slot per row of real state; the
        # scheduler splits anything larger rather than admit-failing it.
        return self._kda_max_slots()

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        if self._moe_capture_blocked(tp_world_size):
            logger.warning(
                "glm5_next: reference MoE dispatch active; decode runs eager "
                "(set moe_quant_kernel=auto for the capturable fused kernel)"
            )
            return []
        # The MLA fallback bakes page indices into its plan, so a captured
        # replay would attend the capture's pages (the resource refuses).
        attn = self.node_resources.get(ATTN)
        if device.type == "cuda" and attn is not None and not attn.use_kernel:
            logger.warning(
                "glm5_next: MLA is on the SDPA fallback, which cannot be "
                "captured; decode runs eager"
            )
            return []
        # DECODE-ONLY: KDA prefill is a host span loop over python-int slot
        # views (decoder_layer._run_kda_prefill), structurally uncapturable.
        max_slots = self._kda_max_slots()
        batch_sizes = [
            b for b in DEFAULT_CAPTURE_BATCH_SIZES
            if max_slots is None or b <= max_slots
        ]
        # MSTAR_GLM53_GRAPH_COMPILE=1 captures the torch.compile'd forward;
        # the default captures the eager one, because a Triton crash in the
        # Inductor compile subprocess fails every capture and degrades back
        # to eager without saying so.
        graph_compile = os.environ.get("MSTAR_GLM53_GRAPH_COMPILE", "0") == "1"
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=batch_sizes,
                compile=graph_compile,
            ),
        ]

    # -- dtype discipline -------------------------------------------------

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm5NextLLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    # -- engine seam: prepare -> declare -> preprocess -> forward ---------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        text_inputs = inputs["text_inputs"][0]
        seq_len = text_inputs.shape[0]
        # dsa_long_context is refused at model __init__, so the serving
        # regime is ALWAYS ctx <= index_topk, where dense MLA computes
        # exactly what GLM-5.3-Flash's DSA would — refuse beyond it rather
        # than serve off-spec logits. Raising here fails only this request
        # (the engine registers the failure and drops the rid from the batch).
        limit = self.config.index_topk
        kda = self.node_resources.get(KDA_STATE) if self.node_resources else None
        committed = kda.committed(fwd_info.request_id) if kda is not None else 0
        if committed + seq_len > limit:
            raise RuntimeError(
                f"request {fwd_info.request_id}: context {committed + seq_len} "
                f"exceeds index_topk={limit}, the regime where dense MLA is "
                "exactly GLM-5.3-Flash's DSA computation. Long context needs "
                "the k-pool indexer engine path (dsa_long_context), a post-M1 "
                "follow-up refused at model __init__."
            )
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
            prefill_tokens = {
                rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            }
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(
                    apply_penalty=True, prefill_tracked_tokens=prefill_tokens,
                ),
                # prefill (and any chunked continue) walks spans on the host;
                # decode is one recurrent step per row
                KDA_STATE: SlotStateStep(
                    mode="chunk" if graph_walk == "prefill" else "step",
                ),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # The one per-replay-varying input. Everything else the forward reads
        # is a resource plan (slot index, KV write slots, planned kernels).
        return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn = engine_inputs.resources[ATTN]
        sampler = engine_inputs.resources[SAMPLER]
        hidden = self.language_model.model(input_ids)
        if graph_walk == "prefill":
            hidden = attn.select_last_hidden(hidden, LABEL)
        elif graph_walk != "decode":
            raise ValueError(f"unsupported graph walk: {graph_walk!r}")
        logits = self.lm_head(hidden)  # (rows, vocab)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(graph_walk, engine_inputs, input_ids)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids)
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # -- slow-postprocess path (worker thread, after execute_batch) -------

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Seed the next decode step from the emitted token: the walk's
        # text_inputs loop-back edge (get_graph_walk_graphs) carries it.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # The ONLY place .item() on a token value is allowed (slow-postprocess
        # path, off the step-critical GPU thread — base-class contract).
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        is_eos = token in self.config.eos_token_ids
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        # Total generated = 1 prefill-emitted token + (decode iters + 1):
        # max_tokens counts the token the prefill emits, so the decode loop
        # runs one fewer.
        generated = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2
        if (not ignore_eos and is_eos) or generated >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
