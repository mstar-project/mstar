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


def test_concat_joins_along_the_declared_dim():
    from mstar.model.submodule_base import ChunkedPrefillOutputPolicy

    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"mask": [torch.zeros(2, 3)]})

    out = acc.release(0, "LLM", {"mask": [torch.ones(2, 4)]}, {"mask": ChunkedPrefillOutputPolicy(dim=1)})

    assert out["mask"][0].shape == (2, 7)


def test_list_keeps_every_chunk():
    from mstar.model.submodule_base import ChunkedPrefillOutputMode, ChunkedPrefillOutputPolicy

    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"states": [torch.tensor([1.0])]})

    out = acc.release(
        0, "LLM", {"states": [torch.tensor([2.0])]},
        {"states": ChunkedPrefillOutputPolicy(mode=ChunkedPrefillOutputMode.LIST)},
    )

    assert [t.tolist() for t in out["states"]] == [[1.0], [2.0]]


def test_concat_joins_each_tensor_of_an_edge_with_its_own():
    """An edge of several tensors per chunk (one per layer, say) stays several."""
    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"layers": [torch.tensor([1.0]), torch.tensor([10.0])]})

    out = acc.release(0, "LLM", {"layers": [torch.tensor([2.0]), torch.tensor([20.0])]})

    assert [t.tolist() for t in out["layers"]] == [[1.0, 2.0], [10.0, 20.0]]


def test_a_final_chunk_with_no_outputs_still_releases_the_held_ones():
    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"states": [torch.tensor([1.0])]})

    assert acc.release(0, "LLM", {})["states"][0].tolist() == [1.0]


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
    engine._keyed_walks = {}
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


# ── engine: a chunked row starts past its cached prefix ────────────────


class _Runner:
    """Answers a fixed cached prefix and records what was applied."""

    def __init__(self, matched):
        self.matched = matched
        self.applied = []

    def resolve_cached_prefix(self, rid, node_name, graph_walk):
        return self.matched

    def apply_cached_prefix(self, rid, node_name, graph_walk, inputs, matched_len):
        self.applied.append((rid, inputs.input_seq_len, matched_len))


class _KeyedPrompt(_Prompt):
    def __init__(self, n, reuses=True, guided=False):
        super().__init__(n)
        self.reuses = reuses
        self.guided = guided

    def get_input_sequence_len(self, graph_walk, fwd_info, inputs):
        from mstar.model.submodule_base import InputSeqLenInfo

        return InputSeqLenInfo(self.n)

    def supports_chunked_prefill(self, graph_walk):
        return True

    def reuses_cached_prefix(self, graph_walk, fwd_info):
        return self.reuses

    def prepare_inputs(self, **kwargs):
        inputs = super().prepare_inputs(**kwargs)
        inputs.resource_step_info = self.guided
        return inputs


def _keyed_engine(sub, matched):
    engine = _engine_with(sub)
    engine._keyed_walks = {"LLM": {"prefill"}}
    engine._runner = _Runner(matched)
    engine._prefix_model = "Test"
    return engine


def test_a_keyed_row_is_measured_without_its_cached_prefix():
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (6, 4)


def test_a_row_its_model_will_not_cut_is_measured_whole():
    engine = _keyed_engine(_KeyedPrompt(10, reuses=False), matched=4)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (10, 0)


def test_the_first_chunk_applies_the_prefix_it_starts_past():
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    first = engine._prepare_chunk(_batch(), 0, "prefill", (4, 6))
    engine._prepare_chunk(_batch(), 0, "prefill", (6, 10))

    assert engine._runner.applied == [(0, 10, 4)], "once, with the whole input"
    assert first.input_ids.tolist() == [4, 5] and first.chunk_start == 4


def test_a_row_with_no_cached_prefix_drops_any_lease():
    engine = _keyed_engine(_KeyedPrompt(10), matched=0)

    engine._prepare_chunk(_batch(), 0, "prefill", (0, 4))

    assert engine._runner.applied == [(0, 10, 0)]


def test_a_first_chunk_past_a_prefix_its_inputs_cannot_skip_fails():
    import pytest

    engine = _keyed_engine(_KeyedPrompt(10, guided=True), matched=4)

    with pytest.raises(RuntimeError, match="reuses_cached_prefix"):
        engine._prepare_chunk(_batch(), 0, "prefill", (4, 6))


def test_a_step_over_its_token_budget_is_reported(caplog):
    from types import SimpleNamespace

    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine._token_budget_overruns = {}
    engine._token_budget_overrides = {}
    submodule = SimpleNamespace(max_batch_tokens=lambda walk: 4)
    batch = SimpleNamespace(node_name="LLM", step_context=SimpleNamespace(graph_walk="mixed"))
    over = [SimpleNamespace(input_seq_len=3), SimpleNamespace(input_seq_len=2)]

    with caplog.at_level("WARNING"):
        for _ in range(3):
            engine._check_token_budget(batch, submodule, over)
        engine._check_token_budget(batch, submodule, over[:1])

    assert engine._token_budget_overruns == {("LLM", "mixed"): 3}
    assert len(caplog.records) == 2  # the 1st and 2nd overrun; the 4th would be next


def test_an_unprepared_row_has_no_length():
    from mstar.engine.engine import ExecutingBatch

    batch = ExecutingBatch(node_name="LLM", per_request_info={}, step_context=None)
    batch.input_seq_lens[0] = 7

    assert (batch.seq_len_of(0), batch.seq_len_of(1)) == (7, -1)


def test_final_keeps_only_the_last_chunk_and_holds_nothing_before():
    from mstar.model.submodule_base import ChunkedPrefillOutputMode, ChunkedPrefillOutputPolicy

    final = {"token": ChunkedPrefillOutputPolicy(mode=ChunkedPrefillOutputMode.FINAL)}
    acc = ChunkOutputAccumulator()
    acc.hold(0, "LLM", {"token": [torch.tensor([1])], "states": [torch.tensor([1.0])]}, final)

    assert "token" not in acc._held[(0, "LLM")]
    out = acc.release(0, "LLM", {"token": [torch.tensor([9])], "states": [torch.tensor([2.0])]}, final)
    assert out["token"][0].tolist() == [9]
    assert out["states"][0].tolist() == [1.0, 2.0]


# ── engine: the serving config's max_batch_tokens ──────────────────────


def _budgeted_engine():
    from types import SimpleNamespace

    from mstar.engine.engine import Engine

    sub = SimpleNamespace(max_batch_tokens={"prefill": 512, "vision": 2048}.get)
    engine = Engine.__new__(Engine)
    engine._submodules = {"LLM": SimpleNamespace(submodule=sub)}
    engine._token_budget_overrides = {}
    return engine


def test_a_node_budget_overrides_every_budgeted_walk():
    engine = _budgeted_engine()
    engine.set_token_budgets({"LLM": 64, "other_worker_node": 8}, {"prefill", "vision", "decode"})

    budgets = [engine.get_max_batch_tokens("LLM", w) for w in ("prefill", "vision", "decode")]
    assert budgets == [64, 64, None]


def test_a_walk_budget_wins_and_null_turns_chunking_off():
    engine = _budgeted_engine()
    engine.set_token_budgets({"LLM": {"prefill": None, "vision": 4096}}, {"prefill", "vision"})

    assert engine.get_max_batch_tokens("LLM", "prefill") is None
    assert engine.get_max_batch_tokens("LLM", "vision") == 4096


def test_a_budget_the_model_does_not_set_is_refused():
    import pytest

    engine = _budgeted_engine()
    with pytest.raises(ValueError, match="no budget there"):
        engine.set_token_budgets({"LLM": {"decode": 8}}, {"prefill", "decode"})
    with pytest.raises(ValueError, match="never runs"):
        engine.set_token_budgets({"LLM": {"prefil": 8}}, {"prefill"})
    with pytest.raises(ValueError, match="positive int"):
        engine.set_token_budgets({"LLM": 0}, {"prefill"})
