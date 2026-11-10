"""Chunked prefill: the token budget and chunk bookkeeping.

What a chunk's output means is the node's Loop's business, covered in
``test/rust/test_runtime_parity.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.chunking import ChunkRange
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.testing_utils.model import make_model
from mstar.worker.batch_builder import fill_token_budget, fill_token_budget_in_start_order

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


# ── the chunk floor ─────────────────────────────────────────────────────


def _decodes(n: int) -> dict[str, int]:
    return {f"d{i}": 1 for i in range(n)}


def test_decode_rows_that_leave_less_than_the_floor_leave_the_prompt_for_the_next_step():
    decodes = _decodes(2047)
    take, dropped = fill_token_budget([*decodes, "p"], decodes | {"p": 1000}, {"p"}, 2048, 128)
    # neither 1 token nor 127 over budget, which would pad to the next capture
    assert "p" not in take and dropped == {"p"}
    assert sum(take.values()) == 2047


def test_the_floor_takes_a_short_remainder_whole_not_padded():
    decodes = _decodes(1800)
    take, dropped = fill_token_budget(
        [*decodes, "p", "q"], decodes | {"p": 50, "q": 1000}, {"p", "q"}, 2048, 128,
    )
    assert take["p"] == 50 and take["q"] == 198 and not dropped


def test_a_share_under_the_floor_cuts_fewer_rows_within_budget():
    prompts = {f"p{i}": 1000 for i in range(20)}
    take, dropped = fill_token_budget(list(prompts), prompts, set(prompts), 2048, 128)
    assert take == {f"p{i}": 128 for i in range(16)}
    assert dropped == {f"p{i}" for i in range(16, 20)}


def test_the_floor_shares_what_is_left_among_the_rows_it_cuts():
    decodes = _decodes(1748)
    take, dropped = fill_token_budget(
        [*decodes, "a", "b", "c"], decodes | {"a": 1000, "b": 1000, "c": 1000}, {"a", "b", "c"}, 2048, 128,
    )
    assert (take["a"], take["b"]) == (150, 150) and dropped == {"c"}


def test_a_share_over_the_floor_is_left_alone():
    take, _ = fill_token_budget(["a", "b"], {"a": 5000, "b": 5000}, {"a", "b"}, 2048, 128)
    assert take == {"a": 1024, "b": 1024}


def test_a_head_that_cannot_be_cut_keeps_its_place_over_a_floored_row():
    take, dropped = fill_token_budget(["d0", "p0"], {"d0": 3, "p0": 100}, {"p0"}, 0, 4)
    assert take == {"d0": 3}
    assert dropped == {"p0"}


def test_a_row_that_cannot_be_cut_sits_out_over_budget():
    take, dropped = fill_token_budget(["d0", "p0"], {"d0": 1, "p0": 100}, set(), 10)
    assert take == {"d0": 1}
    assert dropped == {"p0"}


def test_the_head_runs_over_budget_rather_than_nothing():
    take, dropped = fill_token_budget(["p0"], {"p0": 100}, set(), 10)
    assert take == {"p0": 100}
    assert not dropped


# ── start order ─────────────────────────────────────────────────────────


def test_a_prompt_mid_prefill_takes_the_budget_before_newer_ones():
    prompts = {"a": 5000, "b": 5000, "c": 5000}
    take, dropped = fill_token_budget_in_start_order(
        ["a", "b", "c"], prompts, set(prompts), 2048, 128, {"c": 3, "b": 7},
    )
    assert take == {"c": 2048}
    assert dropped == {"a", "b"}


def test_the_next_prompt_takes_what_the_last_chunk_of_one_leaves():
    take, dropped = fill_token_budget_in_start_order(
        ["a", "b"], {"a": 500, "b": 5000}, {"a", "b"}, 2048, 128, {"a": 0},
    )
    assert take == {"a": 500, "b": 1548}
    assert not dropped


def test_a_short_prompt_goes_in_whole_ahead_of_a_long_one_mid_prefill():
    take, _ = fill_token_budget_in_start_order(
        ["a", "s"], {"a": 5000, "s": 50}, {"a", "s"}, 2048, 128, {"a": 0},
    )
    assert take == {"s": 50, "a": 1998}


def test_a_prompt_that_can_finish_this_step_goes_ahead_of_one_mid_prefill():
    prompts = {"a": 5000, "b": 1500, "c": 900}
    take, dropped = fill_token_budget_in_start_order(
        ["a", "b", "c"], prompts, set(prompts), 2048, 128, {"a": 0},
    )
    assert take == {"c": 900, "a": 1148}, "shortest first; the oldest takes the rest"
    assert dropped == {"b"}


def test_short_prompts_arriving_every_step_cannot_starve_the_oldest_prompt():
    """Four new 500-token prompts each step would finish in what 16 decode
    rows leave, and take it all: the oldest prompt mid-prefill keeps the
    chunk floor out of the finishing pass, so it advances every step."""
    decodes = _decodes(16)
    fresh = {f"n{i}": 500 for i in range(4)}
    take, dropped = fill_token_budget_in_start_order(
        [*decodes, *fresh, "old"], decodes | fresh | {"old": 20000},
        set(fresh) | {"old"}, 2048, 128, {"old": 0},
    )
    assert take["old"] >= 128
    assert sum(take.values()) == 2048
    assert dropped == {"n3"}


def test_decode_rows_go_in_whole_beside_the_prompt_in_start_order():
    decodes = _decodes(40)
    take, dropped = fill_token_budget_in_start_order(
        [*decodes, "p", "q"], decodes | {"p": 5000, "q": 5000}, {"p", "q"}, 2048, 128, {"q": 0},
    )
    assert take == decodes | {"q": 2008}
    assert dropped == {"p"}


def test_a_row_that_cannot_be_cut_does_not_hold_up_the_rows_behind_it():
    take, dropped = fill_token_budget_in_start_order(
        ["n", "p"], {"n": 3000, "p": 5000}, {"p"}, 2048, 128,
    )
    assert take == {"p": 2048}
    assert dropped == {"n"}


def test_in_start_order_decode_rows_that_leave_less_than_the_floor_leave_the_prompts_out():
    decodes = _decodes(2047)
    take, dropped = fill_token_budget_in_start_order(
        [*decodes, "p", "q"], decodes | {"p": 1000, "q": 1000}, {"p", "q"}, 2048, 128,
    )
    assert dropped == {"p", "q"}
    assert sum(take.values()) == 2047


def test_in_start_order_a_prompt_that_finishes_leaves_the_next_out_rather_than_over_budget():
    # a step that would run over its budget pays for the next capture bucket up
    take, dropped = fill_token_budget_in_start_order(
        ["a", "b"], {"a": 1990, "b": 6000}, {"a", "b"}, 2048, 128, {"a": 0},
    )
    assert take == {"a": 1990} and dropped == {"b"}


def test_in_start_order_a_prompt_behind_one_that_finishes_takes_what_is_left():
    take, dropped = fill_token_budget_in_start_order(
        ["a", "b"], {"a": 1500, "b": 6000}, {"a", "b"}, 2048, 128, {"a": 0},
    )
    assert take == {"a": 1500, "b": 548} and not dropped


def test_in_start_order_the_head_runs_over_budget_rather_than_nothing():
    take, dropped = fill_token_budget_in_start_order(["p0"], {"p0": 100}, set(), 10)
    assert take == {"p0": 100}
    assert not dropped
    take, dropped = fill_token_budget_in_start_order(["d0", "p0"], {"d0": 3, "p0": 100}, {"p0"}, 0, 4)
    assert take == {"d0": 3}
    assert dropped == {"p0"}


@pytest.mark.parametrize("fill, cut", [
    (fill_token_budget_in_start_order, {"a": 1500, "b": 512}),
    (fill_token_budget, {"a": 896, "b": 896}),
])
def test_a_cut_row_takes_a_multiple_of_the_alignment_so_its_next_chunk_starts_on_it(fill, cut):
    """A recurrent pool keeps state only on its page grid, counted from where a chunk starts;
    a chunk cut off the grid leaves every later chunk of its prompt off it too."""
    decodes = _decodes(31)
    take, dropped = fill(
        [*decodes, "a", "b"], decodes | {"a": 1500, "b": 1500}, {"a", "b"}, 2048, 128, align=128,
    )
    assert take == decodes | cut
    assert not dropped


def test_the_head_cut_over_its_budget_takes_a_multiple_of_the_alignment():
    take, _ = fill_token_budget_in_start_order(["p"], {"p": 5000}, {"p"}, 300, 128, align=128)
    assert take == {"p": 256}


@pytest.mark.parametrize("floor, align, budget, short", [
    (128, 256, 512, 128),  # a KV page_size of 256 under the default floor
    (64, 128, 2048, 62),
    (200, 128, 2048, 100),
])
def test_short_prompts_cannot_starve_the_oldest_prompt_under_an_alignment(floor, align, budget, short):
    """The oldest prompt mid-prefill keeps the floor out of the finishing pass. A floor between
    two multiples of the alignment leaves it too little to cut, every step that short prompts
    take the rest, so the floor rounds up to one."""
    from mstar.engine.chunking import ChunkingManagement

    chunking = ChunkingManagement({})
    chunking.set_budgets({}, set(), min_chunk_tokens=floor)
    min_chunk = chunking.min_chunk_tokens(budget, align)
    shorts = {f"s{i}": short for i in range(2 * budget // short)}
    take, dropped = fill_token_budget_in_start_order(
        [*shorts, "old"], shorts | {"old": 20000}, set(shorts) | {"old"}, budget, min_chunk, {"old": 0}, align,
    )
    assert "old" not in dropped
    assert take["old"] >= floor and take["old"] % align == 0


def test_the_floor_rounds_up_to_the_alignment_within_the_budget():
    from mstar.engine.chunking import ChunkingManagement

    chunking = ChunkingManagement({})
    assert chunking.min_chunk_tokens(2048, 128) == 128
    assert chunking.min_chunk_tokens(512, 256) == 256
    chunking.set_budgets({}, set(), min_chunk_tokens=200)
    assert chunking.min_chunk_tokens(2048, 128) == 256
    # never past the budget's whole multiples, which a cut could not reach
    assert chunking.min_chunk_tokens(250, 128) == 128
    assert chunking.min_chunk_tokens(2048) == 200, "a node that declares no alignment keeps its floor"


# ── engine: prepare once, cut per chunk ─────────────────────────────────


def _engine_with(submodule, policies=None):
    """``policies`` is the model's ``chunking_policy_of()``: (node, walk) -> policy."""

    from mstar.testing_utils.engine import make_engine, make_mgmt

    engine = make_engine(_submodules={"LLM": make_mgmt(submodule)})
    engine._chunking.set_policies(policies or {})
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
        per_request_input_metadata={}, first_chunk_rids=set(), chunk_ranges={},
    )


def test_a_chunked_row_is_prepared_once_and_cut_per_chunk():
    sub = _Prompt(10)
    engine = _engine_with(sub)

    first = engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))
    last = engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(4, 10, 10, 6))

    assert sub.prepares == 1
    assert first.input_ids.tolist() == [0, 1, 2, 3] and not first.is_final_chunk
    assert last.input_ids.tolist() == list(range(4, 10)) and last.is_final_chunk
    assert (first.chunk_start, first.chunk_total, last.chunk_start) == (0, 10, 4)


def test_the_whole_input_is_kept_until_the_last_chunk_lands():
    """A refused admit retries the last chunk: it must not prepare again."""
    sub = _Prompt(10)
    engine = _engine_with(sub)
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 10, 10, 10))
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 10, 10, 10))
    assert sub.prepares == 1

    engine.release_chunk_inputs(0, "LLM")
    assert not engine._chunking.prepared


def _refused_admit(engine, sub, batch):
    from types import SimpleNamespace

    from mstar.engine.resources.step import AdmitOutcome, AllocationFailed, FullAdmitOutcome

    refused = AdmitOutcome(ok=False, reason=AllocationFailed(
        message="full", pages_short=1, label="main", request_id=0,
    ))
    engine._runner = SimpleNamespace(admit=lambda step: FullAdmitOutcome(refused, "main"))
    batch.register_admit_error = lambda *a: None
    outcome, _ = engine._declare_and_admit(
        batch, [0], [], sub, False, None, step=SimpleNamespace(set_ctx=lambda ctx: None),
    )
    assert not outcome.ok


def test_a_refused_admit_drops_the_input_its_first_chunk_prepared():
    """The retry prepares it again, so its length and cached prefix are checked
    against the pages it then finds."""
    sub = _Prompt(10)
    engine = _engine_with(sub)
    batch = _batch()
    engine._prepare_chunk(batch, 0, "prefill", ChunkRange(0, 4, 10, 4))
    _refused_admit(engine, sub, batch)
    assert not engine._chunking.prepared

    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))
    assert sub.prepares == 2


def test_a_refused_later_chunk_keeps_the_prepared_input():
    sub = _Prompt(10)
    engine = _engine_with(sub)
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))
    later = _batch()
    engine._prepare_chunk(later, 0, "prefill", ChunkRange(4, 10, 10, 6))
    _refused_admit(engine, sub, later)
    assert engine._chunking.prepared, "the row's earlier chunks already ran from it"
    assert sub.prepares == 1


def test_a_request_removed_mid_prompt_drops_its_prepared_input():
    """Cancelled or failed between chunks: nothing else would release it."""
    from types import SimpleNamespace

    class _Sub(_Prompt):
        def cleanup_request(self, request_id):
            pass

    engine = _engine_with(_Sub(10))
    engine._loopback_steps, engine._resident_delta = {}, None
    engine._runner = SimpleNamespace(remove_request=lambda rid: None)
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))

    engine.remove_request(0)

    assert not engine._chunking.prepared


# ── engine: a chunked row starts past its cached prefix ────────────────


class _Runner:
    """Answers a fixed cached prefix and records what was applied."""

    def __init__(self, matched):
        self.matched = matched
        self.applied = []

    def resolve_cached_prefix(self, rid, node_name, graph_walk, inputs):
        from mstar.engine.resources.base import CachedPrefix

        return CachedPrefix("main", self.matched, self.matched)

    def apply_cached_prefix(self, rid, node_name, graph_walk, inputs, prefix):
        self.applied.append((rid, inputs.input_seq_len, prefix and prefix.tokens))


class _KeyedPrompt(_Prompt):
    def __init__(self, n, reuses=True, guided=False):
        super().__init__(n)
        self.reuses = reuses
        self.guided = guided

    def get_input_sequence_len(self, graph_walk, fwd_info, inputs):
        from mstar.model.submodule_base import InputSeqLenInfo

        return InputSeqLenInfo(self.n)

    def prepare_inputs(self, **kwargs):
        inputs = super().prepare_inputs(**kwargs)
        inputs.resource_step_info = self.guided
        return inputs


def _keyed_engine(sub, matched):
    engine = _engine_with(sub, {("LLM", "prefill"): ChunkingPolicy(
        walks=("prefill",), chunkable=True,
        reuses_cached_prefix=lambda info, sub=sub: sub.reuses,
    )})
    engine._keyed_walks = {"LLM": {"prefill"}}
    engine._runner = _Runner(matched)
    engine._prefix_model = "Test"
    return engine


def test_a_keyed_row_is_measured_without_its_cached_prefix():
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (6, 4)


def test_a_row_the_cache_holds_whole_is_measured_at_nothing():
    """Measured whole, the budget would cut it into chunks from 0, and its
    first chunk would write over the lease the walks before it were served
    from, failing the request when those walks end inside a page."""
    engine = _keyed_engine(_KeyedPrompt(10), matched=10)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (0, 10)


def test_a_row_its_model_will_not_cut_is_measured_whole():
    engine = _keyed_engine(_KeyedPrompt(10, reuses=False), matched=4)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (10, 0)


def test_the_first_chunk_applies_the_prefix_it_starts_past():
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    first = engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(4, 6, 10, 6))
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(6, 10, 10, 4))

    assert engine._runner.applied == [(0, 10, 4)], "once, with the whole input"
    assert first.input_ids.tolist() == [4, 5] and first.chunk_start == 4


def test_a_row_with_no_cached_prefix_drops_any_lease():
    engine = _keyed_engine(_KeyedPrompt(10), matched=0)

    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))

    assert engine._runner.applied == [(0, 10, 0)]


def test_a_row_chunked_from_the_start_writes_over_a_lease_it_declined():
    engine = _keyed_engine(_KeyedPrompt(10, reuses=False), matched=4)

    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))

    assert engine._runner.applied == [(0, 10, None)]


def test_a_hit_that_lands_after_the_row_was_measured_is_declined_not_dropped():
    """Measured with nothing cached, then another request's pages landed while
    the row waited. Its chunks are cut from 0, so it runs from 0, but with a
    zero-token prefix rather than none: no prefix would end the row's chain,
    and it would never key its own pages."""
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))

    assert engine._runner.applied == [(0, 10, 0)]


def test_a_rebuild_of_a_prompt_mid_prefill_is_cut_from_the_prompts_input_and_leaves_it():
    """The prompt ran its first chunk and was dropped. Its rebuild runs that chunk again from the input prepared for
    it, applying the cached prefix it starts past, and the prompt's next chunk is cut from the same input."""
    sub = _KeyedPrompt(10)
    engine = _keyed_engine(sub, matched=0)
    engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 10, 4))
    rebuild = _batch()

    rerun = engine._prepare_chunk(rebuild, 0, "prefill", ChunkRange(0, 4, 4, 4))
    resumed = engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(4, 10, 10, 6))

    assert sub.prepares == 1 and not rebuild.first_chunk_rids, "a refused rebuild would release the prompt's input"
    assert rerun.input_ids.tolist() == [0, 1, 2, 3] and resumed.input_ids.tolist() == list(range(4, 10))
    assert engine._runner.applied == [(0, 10, 0), (0, 10, 0)], (
        "the rebuild would write over the pages the drop left cached, unprobed"
    )


def test_a_first_chunk_past_a_prefix_its_inputs_cannot_skip_fails():
    import pytest

    engine = _keyed_engine(_KeyedPrompt(10, guided=True), matched=4)

    with pytest.raises(RuntimeError, match="start past a 4-token cached prefix"):
        engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(4, 6, 10, 6))


def test_a_row_prepared_at_another_length_than_it_was_cut_by_fails():
    import pytest

    engine = _engine_with(_Prompt(10))

    with pytest.raises(RuntimeError, match="measured at 12 tokens .* prepared inputs hold 10"):
        engine._prepare_chunk(_batch(), 0, "prefill", ChunkRange(0, 4, 12, 4))
    assert not engine._chunking.prepared, "a failed row keeps no prepared input"


def test_the_mismatched_row_fails_and_the_rest_of_the_step_runs():
    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.step import StepContext
    from mstar.model.submodule_base import ARNodeInputs

    class _Sub(_Prompt):
        def uniform_row_inputs(self, graph_walk, has_loopback=False):
            return None

        def can_batch(self, batch, model_inputs):
            return True

        def prepare_inputs(self, **kwargs):
            # row 1 is a 10-token prompt; row 2 a one-token decode row
            if kwargs["fwd_info"].info == 2:
                return ARNodeInputs(input_ids=torch.arange(1), input_seq_len=1)
            return super().prepare_inputs(**kwargs)

    engine = _engine_with(_Sub(10))
    batch = ExecutingBatch(
        node_name="LLM",
        per_request_info={1: 1, 2: 2},
        chunk_ranges={1: ChunkRange(0, 4, 12, 4)},  # measured at 12, prepares 10
        step_context=StepContext(request_ids=(1, 2), graph_walk="prefill", slot=0, capture=False),
    )

    engine.prepare_inputs(batch)

    assert list(batch.failed_requests) == [1]
    assert batch.request_ids == [2]


def test_a_step_over_its_token_budget_is_reported(caplog):
    from types import SimpleNamespace

    from mstar.engine.chunking import ChunkingManagement
    from mstar.model.submodule_base import ChunkingPolicy
    from mstar.testing_utils.engine import make_mgmt

    chunking = ChunkingManagement({"LLM": make_mgmt(submodule=SimpleNamespace())})
    chunking.set_policies(
        {("LLM", "mixed"): ChunkingPolicy(walks=("mixed",), max_batch_tokens=4)},
    )
    batch = SimpleNamespace(node_name="LLM", step_context=SimpleNamespace(graph_walk="mixed"))
    over = [SimpleNamespace(input_seq_len=3), SimpleNamespace(input_seq_len=2)]

    with caplog.at_level("WARNING"):
        for _ in range(3):
            chunking.check_budget(batch, over)
        chunking.check_budget(batch, [SimpleNamespace(input_seq_len=4)])  # on the budget

    assert chunking._overruns == {("LLM", "mixed"): 3}
    assert len(caplog.records) == 2  # the 1st and 2nd overrun; the 4th would be next


def test_an_unprepared_row_has_no_length():
    from mstar.engine.engine import ExecutingBatch

    batch = ExecutingBatch(node_name="LLM", per_request_info={}, step_context=None)
    assert batch.seq_len_of(0) == -1
    batch.register_prepare_batch([SimpleNamespace(input_seq_len=7)])
    batch.prepared_rids = (0,)

    assert (batch.seq_len_of(0), batch.seq_len_of(1)) == (7, -1)


def test_a_continuing_decode_row_is_estimated_at_one_token():
    from mstar.model.bagel.submodules import LLMSubmodule as BagelLLM
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
    from mstar.model.qwen3_5.submodules import LLMSubmodule as QwenLLM

    for cls, prompt_walk in ((QwenLLM, "prefill_text"), (OrpheusLLMSubmodule, "prefill"), (BagelLLM, "prefill_text")):
        llm = object.__new__(cls)
        assert llm.speculative_input_seq_len("decode", None, 40) == 1
        # elsewhere the span in flight on this node, and nothing from another node
        assert llm.speculative_input_seq_len(prompt_walk, None, 64) == 64
        assert llm.speculative_input_seq_len(prompt_walk, None, -1) is None



# ── engine: the serving config's max_batch_tokens ──────────────────────


def _budgeted_engine(align: int = 1):
    """An engine whose LLM node carries a resource declaring ``align`` (`Resource.chunk_alignment`)."""
    from types import SimpleNamespace

    from mstar.model.submodule_base import ChunkingPolicy
    from mstar.testing_utils.engine import make_engine, make_mgmt
    from mstar.testing_utils.resource import make_resource

    state = make_resource(chunk_alignment=align)
    engine = make_engine(_submodules={"LLM": make_mgmt(submodule=SimpleNamespace(), resources={"state": state})})
    engine.set_chunking_policies({
        ("LLM", "prefill"): ChunkingPolicy(walks=("prefill",), max_batch_tokens=512),
        ("LLM", "vision"): ChunkingPolicy(walks=("vision",), max_batch_tokens=2048),
    })
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


def test_a_node_the_worker_will_not_chunk_keeps_no_budget():
    """Chunking is off under TP/SP, so the node's walks report no budget and
    no chunking whatever the model declares -- the scheduler then takes the
    same path as a walk the model never budgeted."""
    engine = _budgeted_engine()
    engine.set_chunking_policies({
        ("LLM", "prefill"): ChunkingPolicy(
            walks=("prefill",), chunkable=True, max_batch_tokens=512),
        ("LLM", "vision"): ChunkingPolicy(
            walks=("vision",), chunkable=True, max_batch_tokens=2048),
    })
    engine.set_token_budgets(
        {"LLM": {"vision": 4096}}, {"prefill", "vision"}, no_chunking={"LLM"},
    )

    assert engine.get_max_batch_tokens("LLM", "prefill") is None
    assert engine.get_max_batch_tokens("LLM", "vision") is None
    assert not engine.supports_chunked_prefill("LLM", "vision")


def test_a_budget_the_model_does_not_set_is_refused():
    import pytest

    engine = _budgeted_engine()
    with pytest.raises(ValueError, match="no budget there"):
        engine.set_token_budgets({"LLM": {"decode": 8}}, {"prefill", "decode"})
    with pytest.raises(ValueError, match="never runs"):
        engine.set_token_budgets({"LLM": {"prefil": 8}}, {"prefill"})
    with pytest.raises(ValueError, match="positive int"):
        engine.set_token_budgets({"LLM": 0}, {"prefill"})


def test_the_chunk_floor_defaults_to_128_and_a_serving_config_can_set_it():
    import pytest

    engine = _budgeted_engine()
    engine.set_token_budgets({}, {"prefill"})
    assert engine.min_chunk_tokens(512) == 128
    # never more than the budget itself
    assert engine.min_chunk_tokens(64) == 64

    engine.set_token_budgets({}, {"prefill"}, min_chunk_tokens=32)
    assert engine.min_chunk_tokens(512) == 32

    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match="min_chunk_tokens"):
            engine.set_token_budgets({}, {"prefill"}, min_chunk_tokens=bad)


# ── engine: budgets under a combined walk ──────────────────────────────

COMBINED = {("LLM", "prefill"): "mixed", ("LLM", "decode"): "mixed"}
WALKS = {"prefill", "decode"}


def _mixed_engine(align: int = 1):

    from mstar.model.submodule_base import ChunkingPolicy

    engine = _budgeted_engine(align)
    engine.set_chunking_policies({
        ("LLM", "prefill"): ChunkingPolicy(
            walks=("prefill",), chunkable=True, max_batch_tokens=512),
        ("LLM", "mixed"): ChunkingPolicy(walks=("mixed",), max_batch_tokens=1024),
    })
    return engine


def test_a_walk_budget_inside_a_combined_walk_is_refused_without_the_flag():
    """Its steps run under the combined walk's budget, so it would do nothing."""
    import pytest

    engine = _mixed_engine()
    with pytest.raises(ValueError, match="can_override_mixed_max_tokens for LLM"):
        engine.set_token_budgets({"LLM": {"prefill": 64}}, WALKS, COMBINED)


def test_the_combined_walk_itself_takes_a_budget():
    engine = _mixed_engine()
    engine.set_token_budgets({"LLM": {"mixed": 256}}, WALKS, COMBINED)

    assert engine.get_max_batch_tokens("LLM", "mixed") == 256
    assert engine.walk_token_caps("LLM", "mixed") is None


def test_with_the_flag_a_walk_budget_caps_its_rows_in_a_mixed_step():
    engine = _mixed_engine()
    engine.set_token_budgets({"LLM": {"prefill": 64}}, WALKS, COMBINED, override_mixed={"LLM"})

    assert engine.get_max_batch_tokens("LLM", "mixed") == 1024
    assert engine.walk_token_caps("LLM", "mixed") == {"prefill": 64}


def test_with_the_flag_the_models_own_walk_budget_caps_too():
    engine = _mixed_engine()
    engine.set_token_budgets({}, WALKS, COMBINED, override_mixed={"LLM"})

    assert engine.walk_token_caps("LLM", "mixed") == {"prefill": 512}


def test_without_the_flag_a_combined_step_has_no_walk_caps():
    engine = _mixed_engine()
    engine.set_token_budgets({}, WALKS, COMBINED)

    assert engine.walk_token_caps("LLM", "mixed") is None


def test_with_the_flag_a_null_walk_budget_turns_its_chunking_off():
    engine = _mixed_engine()
    engine.set_token_budgets({"LLM": {"prefill": None}}, WALKS, COMBINED, override_mixed={"LLM"})

    assert not engine.supports_chunked_prefill("LLM", "prefill")
    assert engine.walk_token_caps("LLM", "mixed") is None


def test_a_budget_under_the_chunk_alignment_is_refused_and_one_under_three_is_warned(caplog):
    """A cut chunk is a whole number of alignments: under one, every cut would run past the
    budget, and under three, up to an alignment of each step can go unused."""
    engine = _mixed_engine(align=128)
    with pytest.raises(ValueError, match="LLM/mixed: 64 is under the 128-token chunk alignment"):
        engine.set_token_budgets({"LLM": {"mixed": 64}}, WALKS, COMBINED)

    with caplog.at_level("WARNING", logger="mstar.engine.engine"):
        engine.set_token_budgets({"LLM": {"mixed": 300}}, WALKS, COMBINED)
    assert [r.getMessage().split(":")[0] for r in caplog.records] == ["max_batch_tokens for LLM/mixed"]

    caplog.clear()
    with caplog.at_level("WARNING", logger="mstar.engine.engine"):
        _mixed_engine(align=128).set_token_budgets({}, WALKS, COMBINED)
    assert not caplog.records


def test_a_budget_the_scheduler_never_cuts_by_is_not_held_to_the_alignment():
    engine = _budgeted_engine(align=128)
    engine.set_token_budgets({"LLM": 64}, {"prefill", "vision"})
    assert engine.get_max_batch_tokens("LLM", "prefill") == 64


def test_a_step_relabelled_to_one_walk_is_checked_against_its_combined_budget(caplog):
    """A step of only prompt rows runs as ``prefill`` but was scheduled under
    ``mixed``'s budget; the walk's own cap is not what the step is held to."""
    from types import SimpleNamespace

    engine = _mixed_engine()
    engine.set_token_budgets({"LLM": {"prefill": 64}}, WALKS, COMBINED, override_mixed={"LLM"})
    batch = SimpleNamespace(node_name="LLM", step_context=SimpleNamespace(graph_walk="prefill"))

    with caplog.at_level("WARNING"):
        engine._chunking.check_budget(batch, [SimpleNamespace(input_seq_len=600)])
        assert not caplog.records
        engine._chunking.check_budget(batch, [SimpleNamespace(input_seq_len=2000)])

    assert engine._chunking._overruns == {("LLM", "mixed"): 1}
    assert "over its budget of 1024" in caplog.records[0].getMessage()


# ── engine: the serving config's prefill_token_buckets ─────────────────


def _captured_buckets(submodule, walk, ladders, engine, bs=1) -> list[int]:
    """The token counts the engine's runner captures ``submodule``'s ``walk``
    at (batch size ``bs``), given the serving config's ladders and ``engine``'s budgets."""
    import functools

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    runner = AcceleratorGraphRunner(
        submodule_name="LLM", submodule=submodule, resources={}, step_runner=None,
        device=torch.device("cpu"), autocast_dtype=None,
        joint_comm_group=SimpleNamespace(world_size=1), num_slots=1,
        token_ladders=ladders, step_tokens=functools.partial(engine._chunking.max_step_tokens, "LLM"),
    )
    return sorted({
        spec.bucket.num_tokens for spec in runner._get_slot_specs()
        if spec.bucket.graph_walk == walk and spec.bucket.bs == bs
    })


def _orpheus_prefill_buckets(ladders) -> list[int]:
    """Orpheus's ``prefill`` capture."""
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

    engine = _mixed_engine()
    engine.set_token_budgets({}, WALKS, COMBINED)
    # its capture configs read no weights
    return _captured_buckets(object.__new__(OrpheusLLMSubmodule), "prefill", ladders, engine)


def test_a_ladder_reaches_the_capture_of_a_model_that_plumbs_none():
    assert _orpheus_prefill_buckets({"prefill": [48, 16]}) == [16, 48]


def test_rungs_past_the_first_that_holds_the_step_budget_are_dropped():
    """Qwen3.5's ``prefill_text`` capture replays only ``prefill_text`` steps,
    which run inside ``mixed`` and are scheduled under its budget (4096 here),
    not the walk's own 2048. With chunking off nothing is dropped, nor where
    only the walk's is off: a prompt that cannot be cut runs whole."""
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    walks, combined = {"prefill_text", "decode"}, {("LLM", "prefill_text"): "mixed", ("LLM", "decode"): "mixed"}
    llm = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)))
    ladder = {"prefill_text": [1024, 2048, 3000, 4096, 8192]}

    def buckets(budgets, walk="prefill_text", override_mixed=()):
        engine = _budgeted_engine()
        engine.set_chunking_policies({
            ("LLM", "prefill_text"): ChunkingPolicy(walks=("prefill_text",), chunkable=True, max_batch_tokens=2048),
            ("LLM", "mixed"): ChunkingPolicy(walks=("mixed",), max_batch_tokens=2048),
        })
        engine.set_token_budgets(budgets, walks, combined, override_mixed=override_mixed)
        bs = llm.MIXED_CAPTURE_BATCH_SIZES[0] if walk == "mixed" else 1
        return _captured_buckets(llm, walk, {walk: ladder["prefill_text"]}, engine, bs=bs)

    assert buckets({"LLM": {"mixed": 4096}}) == [1024, 2048, 3000, 4096]
    assert buckets({"LLM": {"mixed": 2500}}) == [1024, 2048, 3000]
    assert buckets({"LLM": None}) == [1024, 2048, 3000, 4096, 8192]
    for walk in ("prefill_text", "mixed"):
        assert buckets({"LLM": {"prefill_text": None}}, walk, {"LLM"}) == [1024, 2048, 3000, 4096, 8192]


def test_a_mixed_ladder_keeps_the_rung_a_prompt_beside_decode_rows_runs_past_the_budget():
    """Where a mixed step's decode rows leave a prompt less than
    ``min_chunk_tokens`` of the budget, the prompt still runs that many beside
    them: 127 decode rows and a 2048-token chunk replay at 2176, which stays.
    At the default floor no step passes the budget, so 2176 is dropped. A
    ``prefill_text`` step carries no decode rows, so its ladder drops it too."""
    from mstar.model.qwen3_5.submodules import LLMSubmodule
    from mstar.worker.batch_builder import fill_token_budget_in_start_order

    walks, combined = {"prefill_text", "decode"}, {("LLM", "prefill_text"): "mixed", ("LLM", "decode"): "mixed"}
    llm = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)))
    rows = llm.MIXED_CAPTURE_BATCH_SIZES[-1]

    def buckets(min_chunk_tokens):
        engine = _budgeted_engine()
        engine.set_chunking_policies({
            ("LLM", "prefill_text"): ChunkingPolicy(walks=("prefill_text",), chunkable=True, max_batch_tokens=2048),
            ("LLM", "mixed"): ChunkingPolicy(walks=("mixed",), max_batch_tokens=2048),
        })
        engine.set_token_budgets({}, walks, combined, min_chunk_tokens=min_chunk_tokens)
        # the scheduler's step: the decode rows a speculation brought, then a long prompt
        take, _ = fill_token_budget_in_start_order(
            [0], {0: 5000}, {0}, 2048 - (rows - 1), engine.min_chunk_tokens(2048),
        )
        ladder = [1024, 2048, 2176, 4096]
        return (
            rows - 1 + take[0],
            _captured_buckets(llm, "mixed", {"mixed": ladder}, engine, bs=rows),
            _captured_buckets(llm, "prefill_text", {"prefill_text": ladder}, engine),
        )

    assert buckets(128) == (2048, [1024, 2048], [1024, 2048])
    assert buckets(2048) == (2175, [1024, 2048, 2176], [1024, 2048])


def test_a_walk_with_no_token_ladder_to_set_is_refused():
    # Orpheus's ``decode`` capture takes a batch size, one token a row (a walk whose steps
    # replay another's capture is refused in test_cosyvoice3_model)
    with pytest.raises(ValueError, match="batch size, not tokens"):
        _orpheus_prefill_buckets({"decode": [64]})


def test_a_walk_a_node_captures_nothing_for_at_runtime_is_skipped(caplog):
    """GLM-5.2 captures nothing under ``dsa_long_context``, and serves eagerly
    whatever the ladder."""
    import logging

    from mstar.model.glm52.config import Glm52ModelConfig
    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    glm = Glm52LLMSubmodule(language_model, Glm52ModelConfig(dsa_long_context=True, mla_absorb=True))
    engine = _budgeted_engine()
    with caplog.at_level(logging.WARNING):
        assert _captured_buckets(glm, "prefill", {"prefill": [64]}, engine) == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def _graph_model():
    """LLM runs prefill and decode, which it combines into mixed; ViT runs encode.
    Loading a node raises."""
    from mstar.testing_utils.model import make_model

    def unloadable(*args, **kwargs):
        raise AssertionError("loaded weights before checking the config")

    return make_model(
        get_graph_walk_graphs=lambda: {
            walk: SimpleNamespace(get_nodes=lambda nodes=nodes: nodes)
            for walk, nodes in {"prefill": ["LLM"], "decode": ["LLM"], "encode": ["ViT"]}.items()
        },
        get_combined_graph_walks=lambda: {"LLM": {"mixed": {"prefill", "decode"}}},
        get_submodule=unloadable,
    )


@pytest.mark.parametrize("model_config, match", [
    # the key's place before it moved, which every model's `**kwargs` would swallow
    ({"model_kwargs": {"prefill_token_buckets": [32, 64]}}, "moved to the top level"),
    ({"model_kwargs": {"prefill_batched_token_buckets": [256]}}, "moved to the top level"),
    ({"model_kwargs": {"backbone_token_buckets": [256]}}, "moved to the top level"),
    ({"model_kwargs": {"lm_prefill_token_buckets": [256]}}, "moved to the top level"),
    ({"prefill_token_buckets": [32, 64]}, "node -> capture walk -> ladder"),
    ({"prefill_token_buckets": {"LLM": []}}, "node -> capture walk -> ladder"),
    ({"prefill_token_buckets": {"LLM": {"prefill": []}}}, "distinct positive"),
    ({"prefill_token_buckets": {"LLM": {"prefill": [64, 64]}}}, "distinct positive"),
    ({"prefill_token_buckets": {"LLM": {"prefill": [0, 64]}}}, "distinct positive"),
    ({"prefill_token_buckets": {"LLM": {"prefill": 2048}}}, "distinct positive"),
    ({"prefill_token_buckets": {"LLM": {"prefill": {"1": [64]}}}}, "min batch size"),
    # a batch's ladder short of one request's would outgrow the static buffers its capture sized
    ({"prefill_token_buckets": {"LLM": {"prefill": {1: [64, 4096], 2: [64, 2048]}}}}, "at least as far"),
    # a walk of the model, but another node's
    ({"prefill_token_buckets": {"LLM": {"encode": [64]}}}, "does not run"),
    # no walk, so a piecewise region's, which takes one ladder
    ({"prefill_token_buckets": {"LLM": {"layers": {1: [64]}}}}, "piecewise region"),
])
def test_a_serving_config_ladder_that_cannot_apply_is_refused_before_loading(model_config, match):
    """EngineManager.build refuses it from the config and the model's graph, before any
    node's weights load."""
    import types

    import torch

    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.worker.engine_manager import EngineManager

    with pytest.raises(ValueError, match=match):
        EngineManager.build(
            node_names={"LLM"}, device=torch.device("cpu"),
            model_config={"node_groups": [{"node_names": ["LLM", "ViT"], "ranks": [0]}], **model_config},
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            transfer_engine_info=TransferEngineInfo(my_entity_id="t", my_session_id="t", transfer_engine=None),
            model=_graph_model(), graph_runtime=types.SimpleNamespace(),
        )


def test_a_name_that_is_no_walk_sets_the_piecewise_region_it_labels():
    """OmniVoice's backbone layer loop, by its label; a label it doesn't declare is refused."""
    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.submodules import BACKBONE_LAYERS, OmniVoiceBackboneSubmodule
    from mstar.testing_utils.engine import make_engine, make_mgmt

    backbone = OmniVoiceBackboneSubmodule(torch.nn.Module(), OmniVoiceConfig())

    def regions(ladders):
        mgmt = make_mgmt(backbone, joint_comm_group=SimpleNamespace(world_size=1), region_ladders=ladders)
        return make_engine(_submodules={"backbone": mgmt})._build_piecewise_runners("backbone", mgmt)

    (runner,) = regions({BACKBONE_LAYERS: [512, 256]}).values()
    assert runner._config.total_tokens == [256, 512]
    with pytest.raises(ValueError, match="neither"):
        regions({"backbone_layer": [256]})


def _tower_with_ladder(tower: str, ladder: list[int], walk: str | None = None):
    """A vision block-loop or AuT encoder node, its region captured on the serving
    config's ``ladder`` (or, with ``walk``, the ladder named for that walk instead)
    and its attention resource sized from the class's own top bucket, as the models
    declare it: the node and its region's runner."""
    from mstar.model.components.aut_encoder import AuTEncoderConfig
    from mstar.model.components.aut_submodule import AuTEncoderSubmodule
    from mstar.model.components.block_loop_encoder import VISION_BLOCK_LOOP, PackedBlockLoopEncoder
    from mstar.model.keys import VISION_ATTN
    from mstar.testing_utils.engine import make_engine, make_mgmt

    if tower == "vision":
        tower_class = type("Tower", (PackedBlockLoopEncoder,), {"prepare_inputs": None})
        node = tower_class(torch.nn.Module(), SimpleNamespace(hidden_size=8, head_dim=4), "out")
        node.node_resources = {VISION_ATTN: SimpleNamespace(max_segments_per_step=8)}
        label = VISION_BLOCK_LOOP
    else:
        config = AuTEncoderConfig()
        node = type("Encoder", (AuTEncoderSubmodule,), {"prepare_inputs": None})(
            SimpleNamespace(config=config), "audio_attn")
        node.node_resources = {"audio_attn": SimpleNamespace(
            max_segments_per_request=AuTEncoderSubmodule.max_windows_per_replay(config))}
        label = "aut_encode"
    named = {"token_ladders": {walk: ladder}} if walk else {"region_ladders": {label: ladder}}
    mgmt = make_mgmt(node, joint_comm_group=SimpleNamespace(world_size=1), **named)
    (runner,) = make_engine(_submodules={"node": mgmt})._build_piecewise_runners("node", mgmt).values()
    return node, runner


def test_a_vision_ladder_past_the_class_top_replays_and_batches_to_its_own_top():
    node, runner = _tower_with_ladder("vision", [256, 32768])
    rows = [SimpleNamespace(input_seq_len=10000, resource_step_info={"main": [10000]}) for _ in range(2)]
    assert runner.can_lease(rows, ["a", "b"]) and node.can_batch(None, rows)


def test_an_audio_clip_past_the_window_ceiling_runs_eagerly_however_long_the_ladder():
    """The resource's windows per row are sized from the class's top bucket; a clip
    with more than a replay's plan holds takes no lease, so its stack runs eagerly
    rather than overflowing the plan."""
    node, runner = _tower_with_ladder("audio", [64, 16384])
    windows = node.node_resources["audio_attn"].max_segments_per_request
    rows = [SimpleNamespace(input_seq_len=104 * n) for n in (windows, windows + 1)]
    assert runner.can_lease(rows[:1], ["a"])
    assert not runner.can_lease(rows[1:], ["b"])
    # a batch the longer ladder holds packs, past the class's top bucket
    assert node.can_batch(None, [SimpleNamespace(input_seq_len=3000)] * 2)
    # clips over the ceiling run one by one, not as one eager step of them all
    assert not node.can_batch(None, [SimpleNamespace(input_seq_len=5000)] * 2)


def test_a_walk_named_on_a_node_that_captures_no_walk_is_refused_naming_its_regions():
    """Qwen3.5's vision tower captures only its block-loop region, so a ladder named
    for one of its walks could never apply."""
    with pytest.raises(ValueError, match="vision_block_loop"):
        _tower_with_ladder("vision", [256, 1024], walk="prefill_vision")


@pytest.mark.parametrize("regions_off", [True, False])
def test_a_name_on_a_node_with_no_region_here_is_skipped_only_if_it_declares_some(regions_off, caplog):
    """OmniVoice's backbone declares its region but captures none with ``backbone_graphs``
    off, so the ladder does nothing; Orpheus's LLM declares none at all, so the name is a
    typo of a walk."""
    import logging

    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.submodules import OmniVoiceBackboneSubmodule
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
    from mstar.testing_utils.engine import make_engine, make_mgmt

    node = (
        OmniVoiceBackboneSubmodule(torch.nn.Module(), OmniVoiceConfig(backbone_graphs=False)) if regions_off
        else object.__new__(OrpheusLLMSubmodule)
    )
    mgmt = make_mgmt(node, joint_comm_group=SimpleNamespace(world_size=1), region_ladders={"prefil": [64]})
    engine = make_engine(_submodules={"node": mgmt})
    if regions_off:
        with caplog.at_level(logging.WARNING):
            assert engine._build_piecewise_runners("node", mgmt) == {}
        assert any(r.levelno == logging.WARNING for r in caplog.records)
    else:
        with pytest.raises(ValueError, match="neither"):
            engine._build_piecewise_runners("node", mgmt)


# ── sampler: a non-final chunk's token is not kept ────────────────────


def test_a_non_final_chunks_row_commits_no_sample():
    """The sampler keeps a row mid-prefill out of what its draw commits by
    itself, off the step context, so a model never has to know it chunks."""
    from mstar.engine.resources.sampler.config import SamplerStep
    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar.engine.resources.step import StepContext

    ctx = StepContext(request_ids=("d", "m", "l"), graph_walk="mixed", slot=0, capture=False)
    ctx.chunk_tokens_after = {"m": 6}  # "l" runs its final chunk, "d" a decode token

    assert SamplerResource._kept_rids_of(SamplerStep(), ctx) == {"d", "l"}
    narrowed = SamplerStep(kept_rids=frozenset({"m", "l"}))
    assert SamplerResource._kept_rids_of(narrowed, ctx) == {"l"}

    untouched = SamplerStep()
    plain = StepContext(request_ids=("d",), graph_walk="decode", slot=0, capture=False)
    assert SamplerResource._kept_rids_of(untouched, plain) is None, "every row, as before"


# ── load: the nodes the worker will not chunk ──────────────────────────


class _Chunker:
    """A submodule that reports row lengths, as a chunked walk must."""

    def get_input_sequence_len(self, graph_walk, request_info, inputs):
        del graph_walk, request_info, inputs


class _Unmeasured(_Chunker):
    """Chunks, but leaves row lengths to the base class, which reports none."""

    get_input_sequence_len = NodeSubmodule.get_input_sequence_len


def _chunking_model():
    return make_model(chunking_policy_of=lambda: {
        ("LLM", "prefill"): ChunkingPolicy(
            walks=("prefill",), chunkable=True, max_batch_tokens=512),
    })


def _groups(sp=1):
    """The worker's real groups, so the call site can't drift from their API."""
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups

    groups = WorkerParallelGroups(num_workers=sp, global_rank=0)
    if sp > 1:
        sp_group = CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(sp)))
        groups.add("LLM", sp_group, "sp")
        groups.node_to_instance_comm["LLM"] = sp_group
    return groups


def test_a_node_under_sp_has_its_chunking_turned_off(caplog):
    """Chunked prefill is not supported under SP yet. Refusing to start
    would take SP away from every model that chunks, so the node runs its
    prefills whole and says so."""
    from mstar.worker.engine_manager import _unchunkable_nodes

    with caplog.at_level("WARNING"):
        off = _unchunkable_nodes({"LLM": _Chunker()}, _chunking_model(), _groups(sp=2))

    assert off == frozenset({"LLM"})
    assert "chunked prefill is off for ['LLM']" in caplog.text


def test_a_node_on_one_rank_or_under_tp_keeps_its_chunking():
    """Under TP the leader cuts the chunks and the followers run them."""
    from mstar.worker.engine_manager import _unchunkable_nodes

    assert not _unchunkable_nodes({"LLM": _Chunker()}, _chunking_model(), _groups())


def test_a_chunked_walk_that_reports_no_row_length_is_refused():
    """The model's own declaration, so it is an error on one rank as well."""
    import pytest

    from mstar.worker.engine_manager import _unchunkable_nodes

    with pytest.raises(ValueError, match="get_input_sequence_len"):
        _unchunkable_nodes({"LLM": _Unmeasured()}, _chunking_model(), _groups())
