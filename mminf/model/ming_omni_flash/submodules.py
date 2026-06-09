"""mminf engine submodule for the Ming-flash-omni-2.0 thinker.

Wraps :class:`LingMoeModel` so the engine can call its forward with the
right inputs/cache plumbing. Text-only for step 3d — audio/vision
prefill walks land in step 4.

Reference: mminf's :class:`OrpheusLLMSubmodule`
(`mminf/model/orpheus/submodules.py:20-176`) is the cleanest text-LLM
template; Qwen3-Omni's `ThinkerSubmodule`
(`mminf/model/qwen3_omni/submodules.py:217+`) shows the multimodal
extensions we'll grow into.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.kv_store import PositionInfo
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
)

logger = logging.getLogger(__name__)


class BailingMoeV2ThinkerSubmodule(ARNodeSubmodule):
    """Text-only thinker submodule for Ming-flash-omni-2.0.

    Two graph walks:
      * ``prefill``: embed text token ids, fill KV cache, sample first
        token's logits.
      * ``decode``: embed the previous token, single-step forward,
        sample next-token logits.

    The submodule does NOT use ``cache_handle.apply_rope`` — Ling-2.0's
    partial 3D ``video_rope`` is applied inline by
    :class:`LingAttention` using the explicit ``position_ids`` argument.
    """

    def __init__(self, model: LingMoeModel, eos_token_id: int = 156895) -> None:
        super().__init__()
        self.model = model
        self.eos_token_id = eos_token_id
        # Stash the embed_tokens / lm_head as direct attributes so the
        # engine's CUDA-graph captures don't reach through .model.
        self.embed_tokens = model.embed_tokens
        self.lm_head = model.lm_head

    # ------------------------------------------------------------------
    # ARNodeSubmodule contract
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        """Build per-request ARNodeInputs from the engine-provided tensors.

        ``inputs["text_inputs"]`` is the token-id tensor — either the
        full prompt (prefill) or the single previous token (decode).
        Mirrors :class:`OrpheusLLMSubmodule.prepare_inputs` since the
        Ling thinker also takes packed token ids.
        """
        token_ids = inputs["text_inputs"][0]
        return ARNodeInputs(
            input_ids=token_ids,
            input_seq_len=token_ids.shape[0],
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Plan attention for the engine; pack token ids for forward.

        Single-request only in step 3d; batched preprocess folds in
        step 3e+ via ``can_batch`` + ``forward_batched``.
        """
        if len(inputs) > 1:
            raise NotImplementedError(
                f"BailingMoeV2ThinkerSubmodule: multi-request batching is "
                f"step-3e scope; got {len(inputs)} requests"
            )
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]

        cache_manager.set_active_label("main")
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main",
        )
        # We don't call ``cache_manager.apply_rope`` in attention (we
        # have our own partial 3D rope), but mminf's plan_rope also
        # advances internal position-id state used by ``advance_seq_lens``
        # — keep this call for parity with Orpheus.
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "text_inputs": torch.cat([inp.input_ids for inp in inputs]),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        # Resolve position_ids from per-request position state. For
        # text-only the rope only needs 1D positions: a contiguous span
        # starting at ``position_id_start``.
        request_info = engine_inputs.single_request_info
        start_pos = 0
        try:
            start_pos = (
                request_info.position_info.get("main", PositionInfo())
                .position_id_start
            )
        except AttributeError:
            # ARNodeSubmodule contract may not always provide
            # position_info; fall back to 0 for prefill, 1 + len for decode.
            pass

        num_tokens = text_inputs.shape[0]
        position_ids = torch.arange(
            start_pos, start_pos + num_tokens,
            dtype=torch.long, device=text_inputs.device,
        )

        # Embed + transformer + lm_head. The LingMoeModel forward calls
        # cache_handle.set_layer_idx per layer + cache_handle.run_attention
        # inside LingAttention.
        logits = self.model(
            cache_handle,
            input_ids=text_inputs,
            position_ids=position_ids,
        )

        # Advance the cache's sequence lengths so the next decode step
        # knows where to read/write. This is the standard post-forward
        # call that mminf's KV cache uses to track positions.
        cache_handle.advance_seq_lens()

        # Sample only the last position's logits (next-token sampling).
        # Engine expects "new_token" downstream, but for prefill we
        # also publish logits so the engine's sampling layer can run.
        last_logits = logits[-1:, :]
        return {"logits": [last_logits]}

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ) -> None:
        """Rebind ``new_token`` → ``text_inputs`` for the decode loop.

        The decode walk's output edge is named ``text_inputs`` so the loop
        feeds the previous sampled token back into the next iteration.
        ``submodule.forward`` returns ``{"logits": [...]}``; the KV-cache
        engine samples that into ``{"new_token": [...]}``; this hook then
        publishes the same tensor under the ``text_inputs`` key so the
        graph router finds an output to attach to the loop edge.

        Mirrors :meth:`OrpheusLLMSubmodule.postprocess`.
        """
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the ``decode_loop`` when the sampled token is the EOS
        (``<|role_end|>`` for Ming, token id 156895)."""
        new_tokens = outputs.get("new_token") or []
        if not new_tokens:
            return set()
        last = new_tokens[-1]
        if isinstance(last, torch.Tensor):
            tok = int(last.flatten()[0].item())
        else:
            tok = int(last)
        if tok == self.eos_token_id:
            return {"decode_loop"}
        return set()

    def can_batch(self, batch, model_inputs) -> bool:
        # Step 3d is single-request; step 3e adds batching.
        return False
