"""Chunked prefill: the token budget, chunk bookkeeping and output accumulation."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.worker.batch_builder import fill_token_budget
from mstar.worker.chunk_outputs import ChunkOutputAccumulator

# ── the token budget ────────────────────────────────────────────────────


def test_decode_rows_go_in_whole_and_the_prompt_splits_the_rest():
    take, dropped = fill_token_budget(
        ["d0", "p0", "d1"], {"d0": 1, "p0": 100, "d1": 1}, {"p0"}, 10,
    )
    assert take == {"d0": 1, "d1": 1, "p0": 8}
    assert not dropped


def test_two_long_prompts_share_what_is_left():
    take, _ = fill_token_budget(["a", "b"], {"a": 50, "b": 50}, {"a", "b"}, 10)
    assert take == {"a": 5, "b": 5}


def test_a_row_that_cannot_be_cut_sits_out_over_budget():
    take, dropped = fill_token_budget(["d0", "p0"], {"d0": 1, "p0": 100}, set(), 10)
    assert take == {"d0": 1}
    assert dropped == {"p0"}


def test_the_head_runs_over_budget_rather_than_nothing():
    take, dropped = fill_token_budget(["p0"], {"p0": 100}, set(), 10)
    assert take == {"p0": 100}
    assert not dropped


# ── output accumulation ─────────────────────────────────────────────────


def test_held_chunks_come_back_concatenated_in_order():
    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"states": [torch.tensor([1.0])]})
    acc.hold(0, "LLM", {"states": [torch.tensor([2.0])]})

    out = acc.release(0, "LLM", {"states": [torch.tensor([3.0])], "new_token": [torch.tensor([7])]})

    assert out["states"][0].tolist() == [1.0, 2.0, 3.0]
    assert out["new_token"][0].tolist() == [7]
    assert acc.release(0, "LLM", {"x": []}) == {"x": []}, "released once"


def test_a_removed_request_leaves_nothing_held():
    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"states": [torch.tensor([1.0])]})
    acc.drop(0)
    assert acc.release(0, "LLM", {}) == {}


# ── engine: prepare once, cut per chunk ─────────────────────────────────


def _engine_with(submodule):
    from types import SimpleNamespace

    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine._submodules = {"LLM": SimpleNamespace(submodule=submodule, resources={})}
    engine._chunk_inputs = {}
    return engine


class _Prompt:
    """Counts prepares; cuts with the AR default."""

    def __init__(self, n):
        self.n = n
        self.prepares = 0

    def prepare_inputs(self, **kwargs):
        from mstar.model.submodule_base import ARNodeInputs

        self.prepares += 1
        return ARNodeInputs(input_ids=torch.arange(self.n), input_seq_len=self.n)

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        from mstar.model.submodule_base import ARNodeSubmodule

        return ARNodeSubmodule.split_inputs(self, graph_walk, fwd_info, inputs, start, end)


def _batch():
    from types import SimpleNamespace

    return SimpleNamespace(
        node_name="LLM", per_request_info_wrapped={0: None},
        per_request_input_tensors={}, final_stream_rids=set(),
        per_request_input_metadata={},
    )


def test_a_chunked_row_is_prepared_once_and_cut_per_chunk():
    sub = _Prompt(10)
    engine = _engine_with(sub)

    first = engine._prepare_chunk(_batch(), 0, "prefill", (0, 4))
    last = engine._prepare_chunk(_batch(), 0, "prefill", (4, 10))

    assert sub.prepares == 1
    assert first.input_ids.tolist() == [0, 1, 2, 3] and not first.is_final_chunk
    assert last.input_ids.tolist() == list(range(4, 10)) and last.is_final_chunk
    assert (first.chunk_start, first.chunk_total, last.chunk_start) == (0, 10, 4)


def test_the_whole_input_is_kept_until_the_last_chunk_lands():
    """A refused admit retries the last chunk: it must not prepare again."""
    sub = _Prompt(10)
    engine = _engine_with(sub)
    engine._prepare_chunk(_batch(), 0, "prefill", (0, 10))
    engine._prepare_chunk(_batch(), 0, "prefill", (0, 10))
    assert sub.prepares == 1

    engine.release_chunk_inputs(0, "LLM")
    assert not engine._chunk_inputs
