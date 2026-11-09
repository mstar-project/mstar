"""A prompt prefilled in chunks keys its reply by the tokens the request ran.

Every chunk of a prompt samples a token, and only the last chunk's starts the reply. The
prefix chain names a page by the tokens it holds, so a chain extended with an earlier
chunk's sample names the reply's pages by a token that was never written: the next turn
misses them, and a prompt that spells that token is served KV computed from other tokens.

Drives the real ``StepExecutor.postprocess`` and ``Engine.step_finished`` over a real
``KVManager``. The KV steps are driven by hand, and the graph runtime and tensor store are
the stubs of ``test_dropped_row_stream_stop.py``.
"""

from __future__ import annotations

from concurrent.futures import Future

import pytest
import test_kv_prefix_decode as H
import torch
from test_dropped_row_stream_stop import _Runtime, _Sub, _worker

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext, StepRunner
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.worker.scheduled_batch import ScheduledBatch, unfinished_chunk_rids
from mstar.worker.step_executor import PendingBatch

NODE, WALK, PAGE = H.NODE, H.WALK, H.PAGE_SIZE
RID = 1
# two whole pages and part of a third, which the reply fills and runs past
PROMPT = list(range(2 * PAGE + 44))
REPLY = list(range(9000, 9000 + PAGE))
FIRST_CHUNK = PAGE + 72
MID_PREFILL_SAMPLE = 7777
NEXT_TURN = list(range(500, 560))


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(H.manager_mod, "KVTransferManager", H._StubTransfer)


def _serve(row_buffers: bool) -> KVManager:
    """One request as the worker finishes its steps: ``PROMPT`` in two chunks, then ``REPLY``
    a token a step. Each step writes its tokens to the KV cache, then postprocess reads the
    token it sampled, from the per-request outputs or from the stop check's row buffers."""
    kv = H._manager()
    engine = make_engine(
        _submodules={NODE: make_mgmt(_Sub())}, _runner=StepRunner({"kv": kv}, node_resources={NODE: ["kv"]}),
    )
    worker = _worker(engine, _Runtime(), [RID], in_loop=False, through_send=True)
    H._ingest(kv, RID, PROMPT, decode=len(REPLY))

    def step(span: int, sampled: int, chunk: ChunkRange | None = None) -> None:
        H._step(kv, RID, span)
        chunk_ranges = {RID: chunk} if chunk else {}
        scheduled = ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph={RID: 0}, chunk_ranges=chunk_ranges,
            incomplete_node_rids=unfinished_chunk_rids(chunk_ranges),
        )
        executing = ExecutingBatch(
            node_name=NODE, per_request_info={RID: CurrentForwardPassInfo(str(RID), WALK, 0, 0, len(REPLY))},
            step_context=StepContext(request_ids=(RID,), graph_walk=WALK, slot=0, capture=False),
        )
        outputs = BatchedModelOutput(per_rid_outputs={RID: {H.TENSOR: [torch.tensor([sampled])]}})
        if row_buffers:
            outputs.check_stop_buffers, outputs.row_request_ids = {H.TENSOR: torch.tensor([[sampled]])}, (RID,)
        worker.executor.postprocess(PendingBatch(scheduled, executing, NODE, "p", WALK, Future()), outputs)

    total = len(PROMPT)
    step(FIRST_CHUNK, MID_PREFILL_SAMPLE, ChunkRange(0, FIRST_CHUNK, total, FIRST_CHUNK))
    step(total - FIRST_CHUNK, REPLY[0], ChunkRange(FIRST_CHUNK, total, total, total - FIRST_CHUNK))
    # a decode step writes the token sampled before it and samples the next
    for token in REPLY[1:]:
        step(1, token)
    kv.assert_pages_conserved()
    return kv


def _served(kv: KVManager, rid: str, prompt: list[int]) -> int:
    """The tokens of ``prompt`` a new request is served from the cache."""
    H._ingest(kv, rid, prompt)
    hit = kv.resolve_cached_prefix(rid, NODE, WALK, ARNodeInputs(input_seq_len=len(prompt)))
    return 0 if hit is None else hit.tokens


@pytest.mark.parametrize("row_buffers", [False, True], ids=["per-request outputs", "stop-check rows"])
def test_a_chunked_prompt_keys_its_reply_by_the_tokens_it_ran(row_buffers):
    kv = _serve(row_buffers)

    assert _served(kv, "next turn", PROMPT + REPLY + NEXT_TURN) == 3 * PAGE, (
        "the page the reply filled is not there for the next turn"
    )
    spelled = PROMPT + [MID_PREFILL_SAMPLE] + REPLY + NEXT_TURN
    assert _served(kv, "another prompt", spelled) == 2 * PAGE, (
        "a prompt that spells a mid-prefill chunk's sample was served a page holding other tokens"
    )
