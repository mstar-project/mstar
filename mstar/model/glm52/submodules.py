"""AR submodule for the GLM-5.2 text backbone."""
from __future__ import annotations

import logging
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    distribute_tokens,
)
from mstar.engine.resources import AttentionStep, DSAStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.glm52.config import ATTN_RESOURCE, DSA_RESOURCE, KV_RESOURCE, POS_RESOURCE
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

logger = logging.getLogger(__name__)

_MAIN = "main"


class _PackedPrefillConfig(PackedAcceleratorGraphConfig):
    """Packed prefill capture without the buckets whose capture rows would pass the context
    limit: ``distribute_tokens`` gives row 0 the remainder, and such a capture fails."""

    def __init__(self, *, max_row_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_row_tokens = max_row_tokens

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n * self.total_tokens_multiplier for n in self.capture_token_lengths
                if distribute_tokens(n, bs)[0] <= self.max_row_tokens]


class Glm52LLMSubmodule(SampledTokenDecoder):
    """Embed + 78 decoder layers + lm_head, one fat TP node, on the shared decoder
    scaffold. Its own: the DSA step under ``dsa_long_context``, no captures where the
    MoE dispatch or the DSA resource cannot replay, a prefill capture that skips rows
    past the context limit, and the guards that hold every step to that limit. It has
    no combined walk, so it captures no mixed step."""

    PREFILL_WALK = "prefill"
    ATTN_KEY = ATTN_RESOURCE
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    # the engine's default ladder: the shipped configs serve at most 64 requests at once
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # torch.compile's own mode (cuBLAS GEMMs, no autotuning), the fastest here
    COMPILE_MODE = "default"
    _capture_block_warned = False

    def layer_steps(self) -> dict[str, ResourceStep]:
        steps = {KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True)}
        if self.config.dsa_long_context:
            steps[DSA_RESOURCE] = DSAStep()
        return steps

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        num_tokens: int, **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        # the layers and the indexer rotate by the positions the position resource
        # planned, in the KV plan's order; captured, a slice of the slot's buffer
        return {"position_ids": engine_inputs.resources[POS_RESOURCE].pos_ids(_MAIN)[:num_tokens]}

    # ── capture configs ──

    def _moe_resolved_fused(self) -> bool:
        """True iff the loaded MoE blocks resolved to the fused fp8 kernel."""
        from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

        lm = getattr(self, "model", None)
        if lm is None:
            return False
        for module in lm.modules():
            if isinstance(module, Glm52SparseMoeBlock):
                return bool(getattr(module, "_use_fused", False))
        return False

    def _moe_capture_blocked(self, tp_world_size: int) -> bool:
        """The reference MoE dispatch paths (.nonzero() / host loops) are not
        stream-capturable; only the fused fp8 path is.
        """
        fp8_reference = (
            self.config.quantization_config is not None
            and self.config.moe_fp8_resident
            and not self._moe_resolved_fused()
        )
        naive_tp = self.config.quantization_config is None and tp_world_size > 1
        blocked = fp8_reference or naive_tp
        if blocked and not self._capture_block_warned:
            self._capture_block_warned = True
            logger.warning(
                "Glm52LLMSubmodule: %s; no CUDA graphs are registered and "
                "decode runs eager, which is far slower. %s",
                "the reference MoE dispatch is active" if fp8_reference
                else "naive TP MoE dispatch is active",
                "moe_quant_kernel 'auto' or 'triton' serves the capturable "
                "fused fp8 kernel." if fp8_reference
                else "Serve the fp8 checkpoint to get the fused kernel.",
            )
        return blocked

    def _compile_flags(self) -> dict[str, Any]:
        return {"compile": self.config.graph_compile, "compile_mode": self.COMPILE_MODE}

    @property
    def disable_torch_compile(self) -> bool:
        # graph_compile covers the uncaptured steps too, which the engine compiles
        return not self.config.graph_compile

    def to(self, *args, **kwargs):
        """Honor device moves; refuse post-load dtype casts (per-param dtypes
        are fixed at load — fp32 block scales + router bias, uint8 fp8)."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            logger.info(
                "Glm52LLMSubmodule: ignoring post-load dtype cast to %s "
                "(per-param dtypes are fixed at load; see restore_fp32_params).",
                dtype,
            )
        if device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        return self

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        if self.config.dsa_long_context:
            # the DSA resource selects per request on the host; a captured
            # decode would skip index upkeep. Eager-only.
            return []
        if self._moe_capture_blocked(tp_world_size):
            return []

        def ids(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        flags = self._compile_flags()
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=self.DECODE_WALK,
                single_request_inputs=ids(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                **flags,
            ),
            _PackedPrefillConfig(
                max_row_tokens=self._context_limit(),
                capture_graph_walk=self.PREFILL_WALK,
                capture_token_lengths=list(self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS),
                make_node_input=ids,
                capture_batch_sizes=list(
                    self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
                ),
                **flags,
            ),
        ]

    # ── the context limit ──

    def _context_limit(self) -> int:
        """Rows a request may hold: the serving window on the DSA engine
        path, else index_topk, where dense MLA is exactly GLM-5.2's DSA."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        # here, not in preprocess, so the refusal fails this request and not its batch
        # (process_prompt refuses the same prompts first, as a 400)
        if graph_walk == self.PREFILL_WALK:
            tokens = inputs["text_inputs"][0].shape[0]
            if tokens > self.config.max_prompt_tokens:
                raise RuntimeError(
                    f"request {fwd_info.request_id}: a prompt of {tokens} tokens "
                    f"exceeds the {self.config.max_prompt_tokens} served"
                )
        return super().prepare_inputs(graph_walk, fwd_info, inputs, has_loopback, **kwargs)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Dense MLA equals the reference DSA computation only while every
        # context fits the top-k window; refuse beyond it unless the DSA
        # engine path is on, where the cap is the serving window. The KV pool
        # is sized to this limit, so the conductor holds every request inside
        # it and this only fires on a bug.
        pos = engine_inputs.resources[POS_RESOURCE]
        long_context = self.config.dsa_long_context
        limit = self._context_limit()
        topk = self.config.index_topk
        for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True):
            sl = inp.input_seq_len
            start = pos.position(rid, _MAIN)  # where the step's first token lands
            if start + sl > limit:
                raise RuntimeError(
                    f"request {rid}: context {start + sl} exceeds {limit}, "
                    + (
                        "the configured max_seq_len serving window."
                        if long_context
                        else "the regime where dense MLA is exactly GLM-5.2's "
                        "DSA computation. Long context needs dsa_long_context=True."
                    )
                )
            if long_context and start + sl > topk and sl > 1:
                raise RuntimeError(
                    f"request {rid}: prefill context {start + sl} exceeds "
                    f"index_topk={topk}; sparse attention beyond topk is "
                    "decode-only."
                )
        return super().preprocess(graph_walk, engine_inputs, inputs)
