"""Chunked prefill: the token budget and chunk bookkeeping.

What a chunk's output means is the node's Loop's business, covered in
``test/rust/test_runtime_parity.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.worker.batch_builder import fill_token_budget

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


def test_decode_rows_that_fill_the_budget_leave_a_prompt_the_floor():
    decodes = _decodes(2047)
    take, dropped = fill_token_budget([*decodes, "p"], decodes | {"p": 1000}, {"p"}, 2048, 128)
    assert take["p"] == 128  # 127 over budget, rather than 1 token
    assert len(take) == 2048 and not dropped


def test_the_floor_takes_a_short_remainder_whole_not_padded():
    decodes = _decodes(2040)
    take, _ = fill_token_budget([*decodes, "p"], decodes | {"p": 50}, {"p"}, 2048, 128)
    assert take["p"] == 50


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


# ── engine: prepare once, cut per chunk ─────────────────────────────────


def _engine_with(submodule, policies=None):
    """``policies`` is the model's ``chunking_policy_of()``: (node, walk) -> policy."""
    from types import SimpleNamespace

    from mstar.engine.chunking import ChunkingManagement
    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine._submodules = {"LLM": SimpleNamespace(submodule=submodule, resources={})}
    engine._keyed_walks = {}
    engine._chunking = ChunkingManagement(engine._submodules)
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
        per_request_input_metadata={},
    )


def test_a_chunked_row_is_prepared_once_and_cut_per_chunk():
    sub = _Prompt(10)
    engine = _engine_with(sub)

    first = engine._prepare_chunk(_batch(), 0, "prefill", (0, 4, 10))
    last = engine._prepare_chunk(_batch(), 0, "prefill", (4, 10, 10))

    assert sub.prepares == 1
    assert first.input_ids.tolist() == [0, 1, 2, 3] and not first.is_final_chunk
    assert last.input_ids.tolist() == list(range(4, 10)) and last.is_final_chunk
    assert (first.chunk_start, first.chunk_total, last.chunk_start) == (0, 10, 4)


def test_the_whole_input_is_kept_until_the_last_chunk_lands():
    """A refused admit retries the last chunk: it must not prepare again."""
    sub = _Prompt(10)
    engine = _engine_with(sub)
    engine._prepare_chunk(_batch(), 0, "prefill", (0, 10, 10))
    engine._prepare_chunk(_batch(), 0, "prefill", (0, 10, 10))
    assert sub.prepares == 1

    engine.release_chunk_inputs(0, "LLM")
    assert not engine._chunking.prepared


def test_a_request_removed_mid_prompt_drops_its_prepared_input():
    """Cancelled or failed between chunks: nothing else would release it."""
    from types import SimpleNamespace

    class _Sub(_Prompt):
        def cleanup_request(self, request_id):
            pass

    engine = _engine_with(_Sub(10))
    engine._loopback_steps = {}
    engine._runner = SimpleNamespace(remove_request=lambda rid: None)
    engine._prepare_chunk(_batch(), 0, "prefill", (0, 4, 10))

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


def test_a_row_its_model_will_not_cut_is_measured_whole():
    engine = _keyed_engine(_KeyedPrompt(10, reuses=False), matched=4)

    info = engine.input_sequence_len("LLM", "prefill", None, {}, rid=0)

    assert (info.seq_len, info.cached_prefix) == (10, 0)


def test_the_first_chunk_applies_the_prefix_it_starts_past():
    engine = _keyed_engine(_KeyedPrompt(10), matched=4)

    first = engine._prepare_chunk(_batch(), 0, "prefill", (4, 6, 10))
    engine._prepare_chunk(_batch(), 0, "prefill", (6, 10, 10))

    assert engine._runner.applied == [(0, 10, 4)], "once, with the whole input"
    assert first.input_ids.tolist() == [4, 5] and first.chunk_start == 4


def test_a_row_with_no_cached_prefix_drops_any_lease():
    engine = _keyed_engine(_KeyedPrompt(10), matched=0)

    engine._prepare_chunk(_batch(), 0, "prefill", (0, 4, 10))

    assert engine._runner.applied == [(0, 10, 0)]


def test_a_row_chunked_from_the_start_writes_over_a_lease_it_declined():
    engine = _keyed_engine(_KeyedPrompt(10, reuses=False), matched=4)

    engine._prepare_chunk(_batch(), 0, "prefill", (0, 4, 10))

    assert engine._runner.applied == [(0, 10, None)]


def test_a_first_chunk_past_a_prefix_its_inputs_cannot_skip_fails():
    import pytest

    engine = _keyed_engine(_KeyedPrompt(10, guided=True), matched=4)

    with pytest.raises(RuntimeError, match="start past a 4-token cached prefix"):
        engine._prepare_chunk(_batch(), 0, "prefill", (4, 6, 10))


def test_a_row_prepared_at_another_length_than_it_was_cut_by_fails():
    import pytest

    engine = _engine_with(_Prompt(10))

    with pytest.raises(RuntimeError, match="measured at 12 tokens .* prepared inputs hold 10"):
        engine._prepare_chunk(_batch(), 0, "prefill", (0, 4, 12))
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
    engine._enable_nvtx = False
    engine._loopback_steps = {}
    batch = ExecutingBatch(
        node_name="LLM",
        per_request_info={1: 1, 2: 2},
        chunk_ranges={1: (0, 4, 12)},  # measured at 12, prepares 10
        step_context=StepContext(request_ids=(1, 2), graph_walk="prefill", slot=0, capture=False),
    )

    engine.prepare_inputs(batch)

    assert list(batch.failed_requests) == [1]
    assert batch.request_ids == [2]


def test_a_step_over_its_token_budget_is_reported(caplog):
    from types import SimpleNamespace

    from mstar.engine.chunking import ChunkingManagement
    from mstar.model.submodule_base import ChunkingPolicy

    chunking = ChunkingManagement({"LLM": SimpleNamespace(submodule=SimpleNamespace())})
    chunking.set_policies(
        {("LLM", "mixed"): ChunkingPolicy(walks=("mixed",), max_batch_tokens=4)},
    )
    batch = SimpleNamespace(node_name="LLM", step_context=SimpleNamespace(graph_walk="mixed"))
    # over the budget of 4 and the chunk floor's slack of 4 more
    over = [SimpleNamespace(input_seq_len=6), SimpleNamespace(input_seq_len=3)]

    with caplog.at_level("WARNING"):
        for _ in range(3):
            chunking.check_budget(batch, over)
        chunking.check_budget(batch, [SimpleNamespace(input_seq_len=8)])  # within the slack

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


def _budgeted_engine():
    from types import SimpleNamespace

    from mstar.engine.chunking import ChunkingManagement
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import ChunkingPolicy

    engine = Engine.__new__(Engine)
    engine._submodules = {"LLM": SimpleNamespace(submodule=SimpleNamespace())}
    engine._chunking = ChunkingManagement(engine._submodules)
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


# ── engine: budgets under a combined walk ──────────────────────────────

COMBINED = {("LLM", "prefill"): "mixed", ("LLM", "decode"): "mixed"}
WALKS = {"prefill", "decode"}


def _mixed_engine():

    from mstar.model.submodule_base import ChunkingPolicy

    engine = _budgeted_engine()
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


# ── sampler: a non-final chunk's token is not kept ────────────────────


def test_a_non_final_chunks_row_leaves_the_kept_rows():
    from mstar.engine.resources.sampler.config import SamplerStep, keep_final_chunk_samples
    from mstar.model.submodule_base import ARNodeInputs

    def row(start, total, n):
        inp = ARNodeInputs(input_seq_len=n)
        inp.chunk_start, inp.chunk_total = start, total
        return inp

    whole, mid, last = ARNodeInputs(input_seq_len=1), row(0, 10, 4), row(4, 10, 6)

    step = keep_final_chunk_samples(SamplerStep(), ["d", "m", "l"], [whole, mid, last])
    assert step.kept_rids == {"d", "l"}

    narrowed = keep_final_chunk_samples(SamplerStep(kept_rids=frozenset({"m"})), ["d", "m"], [whole, mid])
    assert narrowed.kept_rids == frozenset()

    untouched = SamplerStep()
    assert keep_final_chunk_samples(untouched, ["d"], [whole]) is untouched


# ── load: the nodes the worker will not chunk ──────────────────────────


class _Chunker:
    """A submodule that reports row lengths, as a chunked walk must."""

    def get_input_sequence_len(self, graph_walk, request_info, inputs):
        del graph_walk, request_info, inputs


class _Unmeasured(_Chunker):
    """Chunks, but leaves row lengths to the base class, which reports none."""

    get_input_sequence_len = NodeSubmodule.get_input_sequence_len


def _chunking_model():
    return SimpleNamespace(chunking_policy_of=lambda: {
        ("LLM", "prefill"): ChunkingPolicy(
            walks=("prefill",), chunkable=True, max_batch_tokens=512),
    })


def _groups(width):
    return SimpleNamespace(get_instance_world_size_for_node=lambda node: width)


def test_a_node_wider_than_one_rank_has_its_chunking_turned_off(caplog):
    """Chunked prefill is not supported under TP/SP yet. Refusing to start
    would take TP away from every model that chunks, so the node runs its
    prefills whole and says so."""
    from mstar.worker.engine_manager import _unchunkable_nodes

    with caplog.at_level("WARNING"):
        off = _unchunkable_nodes({"LLM": _Chunker()}, _chunking_model(), _groups(2))

    assert off == frozenset({"LLM"})
    assert "chunked prefill is off for ['LLM']" in caplog.text


def test_a_node_on_one_rank_keeps_its_chunking():
    from mstar.worker.engine_manager import _unchunkable_nodes

    assert not _unchunkable_nodes({"LLM": _Chunker()}, _chunking_model(), _groups(1))


def test_a_chunked_walk_that_reports_no_row_length_is_refused():
    """The model's own declaration, so it is an error on one rank as well."""
    import pytest

    from mstar.worker.engine_manager import _unchunkable_nodes

    with pytest.raises(ValueError, match="get_input_sequence_len"):
        _unchunkable_nodes({"LLM": _Unmeasured()}, _chunking_model(), _groups(1))
