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

from dataclasses import dataclass
from typing import Callable

import torch

from mminf.engine.base import NodeBatch, NodeOutput
from mminf.model.submodule_base import ARNodeInputs


@dataclass(frozen=True)
class ChunkSlice:
    """One chunk of a single-request prefill, in token-axis coordinates."""
    index: int
    start: int
    end: int
    is_last: bool


def _plan_chunks(seq_len: int, chunk_size: int) -> list[ChunkSlice]:
    """Return the list of chunks covering [0, seq_len) at ``chunk_size`` granularity.

    The last chunk may be shorter than ``chunk_size``. Pure: no torch
    dependency, easy to test and reason about.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    plans: list[ChunkSlice] = []
    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        plans.append(
            ChunkSlice(index=i, start=start, end=end, is_last=(i == n_chunks - 1))
        )
    return plans


def _slice_ar_inputs(inp: ARNodeInputs, start: int, end: int) -> ARNodeInputs:
    """Return a new ARNodeInputs covering token range [start, end).

    Slices token-axis tensors (input_ids, input_embeds, custom_pos_ids).
    tensor_inputs and kwargs are passed through by reference — they hold
    non-token-axis state (e.g. flags) that the chunked path must not mutate.

    Per-tensor token-axis convention:
      - ``input_ids``: token axis is dim 0 if 1D, else dim 1.
      - ``input_embeds``: token axis is dim 0 if 2D (``[seq_len, hidden]``),
        else dim 1 (``[bs, seq_len, hidden]``).
      - ``custom_pos_ids``: ``inp.input_seq_len`` lives on whichever axis
        matches its size.  qwen3_omni packs MRoPE as ``[3, seq_len]`` so
        the token axis is the LAST one; plain text models use 1D.
    """
    chunk_len = end - start
    seq_len = inp.input_seq_len

    def _slice_token(t: torch.Tensor) -> torch.Tensor:
        # Pick the axis whose size equals seq_len. If multiple axes match
        # (degenerate seq_len=1 inputs), fall back to the LAST axis as a
        # convention — chunking a seq_len==1 prefill makes no sense anyway.
        token_axis = -1
        for dim in range(t.dim()):
            if t.shape[dim] == seq_len:
                token_axis = dim
                break
        return t.narrow(token_axis, start, chunk_len)

    input_ids = _slice_token(inp.input_ids) if inp.input_ids is not None else None
    input_embeds = (
        _slice_token(inp.input_embeds) if inp.input_embeds is not None else None
    )

    custom_pos_ids = inp.custom_pos_ids
    if isinstance(custom_pos_ids, torch.Tensor):
        custom_pos_ids = _slice_token(custom_pos_ids)
    elif isinstance(custom_pos_ids, dict):
        custom_pos_ids = {k: _slice_token(v) for k, v in custom_pos_ids.items()}

    return ARNodeInputs(
        input_seq_len=chunk_len,
        input_ids=input_ids,
        input_embeds=input_embeds,
        custom_pos_ids=custom_pos_ids,
        # Aliased (not cloned): downstream must not mutate.
        tensor_inputs=inp.tensor_inputs,
        kwargs=inp.kwargs,
    )


InnerPass = Callable[[NodeBatch, list[ARNodeInputs]], NodeOutput]


def execute_chunked_prefill(
    batch: NodeBatch,
    node_inputs: list[ARNodeInputs],
    chunk_size: int,
    inner_pass: InnerPass,
) -> NodeOutput:
    """Drive a single-request prefill as N forward passes of ``chunk_size`` tokens.

    The orchestrator is stateless. ``inner_pass`` is the engine's existing
    one-pass dispatch (batched / sequential / CUDA-graph). It is called
    once per chunk with a sliced ARNodeInputs whose ``input_seq_len``
    equals the chunk's token count. The KV-cache manager (read inside
    ``inner_pass``) carries state across calls via its existing
    ``plan_attention(seq_lens=...)`` semantics.

    Only the final chunk's NodeOutput is returned; intermediate outputs
    are discarded. This matches the semantics of an unchunked prefill,
    where the model produces sampled tokens / final-position logits only
    once per request.
    """
    if len(batch.request_ids) != 1:
        raise ValueError(
            f"execute_chunked_prefill requires a single-request batch, "
            f"got {len(batch.request_ids)}"
        )
    if len(node_inputs) != 1:
        raise ValueError(
            f"execute_chunked_prefill requires len(node_inputs) == 1, "
            f"got {len(node_inputs)}"
        )

    inp = node_inputs[0]
    plans = _plan_chunks(seq_len=inp.input_seq_len, chunk_size=chunk_size)

    last_output: NodeOutput | None = None
    for plan in plans:
        chunk_inputs = [_slice_ar_inputs(inp, plan.start, plan.end)]
        last_output = inner_pass(batch, chunk_inputs)

    assert last_output is not None  # plans is always non-empty
    return last_output
