"""The fairness peek and the yield-away key on the walk a chain speculates into.

After a mixed step the step's walk is the combined one, which no row has, so
keying on it would count the chain's own parked rows as other work.
"""

from types import SimpleNamespace

from mstar.worker.worker import SpeculationTarget, Worker, _SpecFairness


class _Span:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _worker(target, other_ready=True, speculation=None, mid_prefill_walks=frozenset()):
    peeked, excluded = [], []
    fake = SimpleNamespace(
        _can_speculate=lambda batch: True,
        chunk_loops=SimpleNamespace(speculated=mid_prefill_walks),
        _choose_spec_target=lambda batch: target,
        _try_speculate_next=lambda pending, target: speculation,
        maybe_send_zmq_to_tp_followers=lambda node_batch, **kw: -1,
        _tp_lead_needs_marker=lambda pending, speculation: False,
        _span=lambda name: _Span(),
        enable_nvtx=False,
        _phase_period=0,
        request_state=None,
        scheduler=SimpleNamespace(
            has_ready_excluding=lambda state, key: peeked.append(key) or other_ready,
            get_next_batch=lambda state, exclude_target: excluded.append(exclude_target),
        ),
    )
    fake._build_speculation = Worker._build_speculation.__get__(fake)
    return fake, peeked, excluded


def _pending():
    return SimpleNamespace(
        batch=object(), node_name="LLM", graph_walk="mixed",
        node_batch=SimpleNamespace(admit_error=None), tp_seq=-1,
    )


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
    worker, peeked, excluded = _worker(_PROMPT, speculation=SimpleNamespace(node_batch=None, tp_seq=-1))

    speculation, _ = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is not None and not peeked and not excluded


def test_a_prompt_chain_that_speculates_its_next_chunk_yields_from_its_first_step():
    """Speculating a row's next chunk must not hold decode rows behind a second
    prompt step, so the peek runs before the chain's first speculation."""
    chunk = SimpleNamespace(node_batch=None, tp_seq=-1)
    worker, peeked, excluded = _worker(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, yielded_from = worker._build_speculation(_pending(), 0, _SpecFairness(peek=True, max_consecutive=8))

    assert speculation is not chunk and peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_prompt_chain_alone_speculates_its_next_chunk():
    chunk = SimpleNamespace(node_batch=None, tp_seq=-1)
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
