"""The fairness peek and the yield-away key on the walk a chain speculates into.

After a mixed step the step's walk is the combined one, which no row has, so
keying on it would count the chain's own parked rows as other work.
"""

from types import SimpleNamespace

from mstar.worker.step_repeats import RepeatConfig, StepRepeats
from mstar.worker.worker import SpeculationTarget, Worker, _SpecFairness


class _Span:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _worker(target, other_ready=True, speculation=None, mid_prefill_walks=frozenset(), budgeted=True):
    peeked, excluded = [], []
    fake = SimpleNamespace(
        _can_speculate=lambda batch: True,
        chunk_loops=SimpleNamespace(speculated=mid_prefill_walks),
        _choose_spec_target=lambda batch: target,
        _try_speculate_next=lambda pending, target, owes_run=False: speculation,
        _step_repeats=StepRepeats({}, {}),
        maybe_send_zmq_to_tp_followers=lambda node_batch, **kw: -1,
        _tp_lead_needs_marker=lambda pending, speculation: False,
        _span=lambda name: _Span(),
        enable_nvtx=False,
        _phase_period=0,
        request_state=None,
        scheduler=SimpleNamespace(
            has_ready_excluding=lambda state, key: peeked.append(key) or other_ready,
            get_next_batch=lambda state, exclude_target: excluded.append(exclude_target),
            chunks=lambda node_name, graph_walk: budgeted,
        ),
    )
    fake._build_speculation = Worker._build_speculation.__get__(fake)
    fake._stamp_repeats = lambda *args: None
    return fake, peeked, excluded


def _pending(repeats=1, owed_run=False):
    return SimpleNamespace(
        batch=object(), node_name="LLM", graph_walk="mixed",
        node_batch=SimpleNamespace(
            admit_error=None, step_context=SimpleNamespace(repeats=repeats, owed_run=owed_run),
        ),
        tp_seq=-1,
    )


def test_ready_work_ends_a_run_in_flight_at_its_next_iteration():
    """The rule that yields a chain away, on the device: a run needn't reach its count."""
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    worker, _, _ = _worker(target)
    yielded = []
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(yield_runs=yielded.append))

    worker._build_speculation(_pending(), 1, _SpecFairness(peek=True, max_consecutive=8))
    assert yielded == []
    worker._build_speculation(_pending(repeats=4), 1, _SpecFairness(peek=True, max_consecutive=8))
    assert yielded == ["LLM"]


def test_ready_rows_wait_out_one_run_and_no_more():
    """With ``ready_waits`` a step of one iteration is followed by a run that
    ready work gets no look at and can't end. After the run it gets its look,
    and after a step narrowed to one too, so nothing keeps it out for good."""
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    worker, peeked, _ = _worker(target)
    worker._step_repeats = StepRepeats({"LLM": RepeatConfig(max=4, ready_waits=True)}, {})
    yielded = []
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(yield_runs=yielded.append))
    fair = _SpecFairness(peek=True, max_consecutive=8)

    worker._build_speculation(_pending(), 1, fair)
    assert peeked == []
    worker._build_speculation(_pending(repeats=4), 1, fair)
    assert peeked == [("LLM", "decode")] and yielded == []
    worker._build_speculation(_pending(owed_run=True), 1, fair)
    assert peeked == [("LLM", "decode")] * 2


def test_the_run_ready_rows_wait_for_is_at_most_ready_waits_long():
    """``ready_waits: k`` lets a prefill in after ``k`` iterations: the owed run
    stops there, and the next step takes the waiting rows."""
    from mstar.engine.resources.step import StepContext
    from mstar.worker.micro_scheduler import ScheduledBatch

    ctx = StepContext(request_ids=[0, 1], graph_walk="decode", slot=0, capture=False)
    batch = ScheduledBatch(node_name="LLM", graph_walk="decode", request_to_worker_graph={0: 0, 1: 0})
    worker = Worker.__new__(Worker)
    worker._walk_drivers = {}
    worker._graph_runtime = SimpleNamespace(loop_room=lambda *args: 100)
    worker._step_repeats = StepRepeats({"LLM": RepeatConfig(max=8, ready_waits=3)}, {("LLM", "decode"): None})
    worker._step_repeats._walks.add(("LLM", "decode"))
    speculation = SimpleNamespace(
        scheduled_batch=batch, node_batch=SimpleNamespace(step_context=ctx), loop_name="loop",
        continuing_rids={0, 1}, is_new_iter=True, is_same_node=True, consumed_streaming_edges={},
    )
    in_flight = SimpleNamespace(batch=SimpleNamespace(repeats=1))
    worker._stamp_repeats(speculation, in_flight, 1, owes_run=True)
    assert ctx.repeats == 3 and ctx.owed_run
    worker._stamp_repeats(speculation, in_flight, 1)
    assert ctx.repeats == 8


def test_after_a_mixed_step_the_chain_keys_on_its_target_walk():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    worker, peeked, excluded = _worker(target)

    _, yielded_from = worker._build_speculation(_pending(), 1, _SpecFairness(peek=True, max_consecutive=8))

    assert peeked == [("LLM", "decode")]
    assert yielded_from == ("LLM", "decode") and excluded == [("LLM", "decode")]


def test_with_no_target_the_chain_keys_on_the_step():
    worker, peeked, _ = _worker(None)

    worker._build_speculation(_pending(), 1, _SpecFairness(peek=True, max_consecutive=8))

    assert peeked == [("LLM", "mixed")]


_PROMPT = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="prefill"), [0])


def test_a_chains_first_step_that_cannot_continue_yields_to_ready_work():
    """A prompt-only chunk step is never speculated; the step after it goes to
    the decode rows waiting, not to another prompt's chunk."""
    worker, peeked, excluded = _worker(_PROMPT)

    _, yielded_from = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_chains_first_step_that_cannot_continue_keeps_the_worker_busy_alone():
    worker, _, excluded = _worker(_PROMPT, other_ready=False)

    _, yielded_from = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert yielded_from is None and excluded == [None]


def test_a_chains_first_step_that_continues_does_not_peek():
    worker, peeked, excluded = _worker(
        _PROMPT, speculation=SimpleNamespace(node_batch=None, tp_seq=-1, is_yield_away=False),
    )

    speculation, _ = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is not None and not peeked and not excluded


def test_a_prompt_chain_that_speculates_its_next_chunk_yields_from_its_first_step():
    """Speculating a row's next chunk must not hold decode rows behind a second
    prompt step, so the peek runs before the chain's first speculation."""
    chunk = SimpleNamespace(node_batch=None, tp_seq=-1, is_yield_away=False)
    worker, peeked, excluded = _worker(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, yielded_from = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is not chunk and peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_prompt_chain_alone_speculates_its_next_chunk():
    chunk = SimpleNamespace(node_batch=None, tp_seq=-1, is_yield_away=False)
    worker, _, excluded = _worker(
        _PROMPT, other_ready=False, speculation=chunk,
        mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, _ = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is chunk and not excluded


def test_without_the_peek_a_step_that_cannot_continue_does_not_yield():
    worker, peeked, excluded = _worker(_PROMPT)

    worker._build_speculation(_pending(), 0, _SpecFairness(peek=False, max_consecutive=8))

    assert not peeked and excluded == [None]


def test_a_step_with_no_token_budget_that_cannot_continue_keeps_mains_order():
    """Its prompts ran whole (``max_batch_tokens: null``, or a model that never
    chunks), so there is no chunk step to yield after."""
    worker, peeked, excluded = _worker(_PROMPT, budgeted=False)

    worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert not peeked and excluded == [None]


def test_a_prompt_chain_with_no_token_budget_does_not_yield_from_its_first_step():
    """Inside its chunk Loop but never cut: it speculates on as on main."""
    chunk = SimpleNamespace(node_batch=None, tp_seq=-1, is_yield_away=False)
    worker, peeked, excluded = _worker(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}), budgeted=False,
    )

    speculation, _ = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is chunk and not peeked and not excluded
