"""Engine-internal chunked prefill orchestrator.

Splits a single-request prefill batch into multiple back-to-back forward
passes of ``chunk_size`` tokens each. The paged KV-cache manager carries
state across chunks via its existing ``plan_attention(seq_lens=...)``
semantics — no cache-side changes are needed.

This module is pure orchestration: no engine state, no submodule registry
lookup. It takes a callable ``inner_pass(batch, inputs) -> NodeOutput``
that runs one forward pass (the engine's existing batched / sequential /
CUDA-graph dispatch) and drives it once per chunk.
"""
from __future__ import annotations

import torch

from mminf.model.submodule_base import ARNodeInputs


def _slice_ar_inputs(inp: ARNodeInputs, start: int, end: int) -> ARNodeInputs:
    """Return a new ARNodeInputs covering token range [start, end).

    Slices token-axis tensors (input_ids, input_embeds, custom_pos_ids).
    tensor_inputs and kwargs are passed through by reference — they hold
    non-token-axis state (e.g. flags) that the chunked path must not mutate.
    """
    chunk_len = end - start

    input_ids = inp.input_ids[:, start:end] if inp.input_ids is not None else None
    input_embeds = (
        inp.input_embeds[:, start:end, :] if inp.input_embeds is not None else None
    )

    custom_pos_ids = inp.custom_pos_ids
    if isinstance(custom_pos_ids, torch.Tensor):
        custom_pos_ids = custom_pos_ids[start:end]
    elif isinstance(custom_pos_ids, dict):
        custom_pos_ids = {k: v[start:end] for k, v in custom_pos_ids.items()}

    return ARNodeInputs(
        input_seq_len=chunk_len,
        input_ids=input_ids,
        input_embeds=input_embeds,
        custom_pos_ids=custom_pos_ids,
        # Aliased (not cloned): downstream must not mutate.
        tensor_inputs=inp.tensor_inputs,
        kwargs=inp.kwargs,
    )
