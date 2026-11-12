"""The fairness peek and the yield-away key on the walk a chain speculates into.

After a mixed step the step's walk is the combined one, which no row has, so
keying on it would count the chain's own parked rows as other work.
"""

from types import SimpleNamespace

from mstar.testing_utils.worker import make_worker
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.scheduling_policy import RoundRobin
from mstar.worker.speculation import SpeculationTarget


def _speculator(target, other_ready=True, speculation=None, mid_prefill_walks=frozenset(), budgeted=True, peek=True):
    peeked, excluded = [], []
    speculator = make_worker(
        chunk_loops=ChunkLoops(mid_prefill_walks),
        peek=peek,
        max_consecutive_spec=8,
        scheduler=SimpleNamespace(
            has_ready_excluding=lambda key, among=None: peeked.append(key) or other_ready,
            next_batch=lambda exclude_target: excluded.append(exclude_target),  # noqa: PLW0108
            chunks=lambda node_name, graph_walk: budgeted,
            policy=RoundRobin(),
        ),
    ).speculator
    scheduler = speculator.scheduler
    scheduler.chain_yields = MicroScheduler.chain_yields.__get__(scheduler)
    speculator.can_speculate = lambda batch: True
    speculator._choose_spec_target = lambda batch, repeats: target
    speculator._try_speculate_next = lambda pending, target, owes_run=False: speculation
    return speculator, peeked, excluded


def _build(speculator, consecutive):
    """``build``, for a chain that has run ``consecutive`` steps."""
    speculator._consecutive = consecutive
    return speculator.build(_pending())


def _chain_step():
    return SimpleNamespace(node_batch=SimpleNamespace(node_name="LLM"), tp_seq=-1, is_yield_away=False)


def _pending():
    return SimpleNamespace(
        batch=object(), node_name="LLM", graph_walk="mixed",
        node_batch=SimpleNamespace(admit_error=None, step_context=SimpleNamespace(repeats=1)), tp_seq=-1,
    )


def test_after_a_mixed_step_the_chain_keys_on_its_target_walk():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    speculator, peeked, excluded = _speculator(target)

    _, yielded_from = _build(speculator, 1)

    assert peeked == [("LLM", "decode")]
    assert yielded_from == ("LLM", "decode") and excluded == [("LLM", "decode")]


def test_with_no_target_the_chain_keys_on_the_step():
    speculator, peeked, _ = _speculator(None)

    _build(speculator, 1)

    assert peeked == [("LLM", "mixed")]


_PROMPT = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="prefill"), [0])


def test_a_chains_first_step_that_cannot_continue_yields_to_ready_work():
    """A prompt-only chunk step is never speculated; the step after it goes to
    the decode rows waiting, not to another prompt's chunk."""
    speculator, peeked, excluded = _speculator(_PROMPT)

    _, yielded_from = _build(speculator, 0)

    assert peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_chains_first_step_that_cannot_continue_keeps_the_worker_busy_alone():
    speculator, _, excluded = _speculator(_PROMPT, other_ready=False)

    _, yielded_from = _build(speculator, 0)

    assert yielded_from is None and excluded == [None]


def test_a_chains_first_step_that_continues_does_not_peek():
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=_chain_step(),
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is not None and not peeked and not excluded


def test_a_prompt_chain_that_speculates_its_next_chunk_yields_from_its_first_step():
    """Speculating a row's next chunk must not hold decode rows behind a second
    prompt step, so the peek runs before the chain's first speculation."""
    chunk = _chain_step()
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, yielded_from = _build(speculator, 0)

    assert speculation is not chunk and peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_prompt_chain_alone_speculates_its_next_chunk():
    chunk = _chain_step()
    speculator, _, excluded = _speculator(
        _PROMPT, other_ready=False, speculation=chunk,
        mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is chunk and not excluded


def test_without_the_peek_a_step_that_cannot_continue_does_not_yield():
    speculator, peeked, excluded = _speculator(_PROMPT, peek=False)

    _build(speculator, 0)

    assert not peeked and excluded == [None]


def test_a_step_with_no_token_budget_that_cannot_continue_keeps_mains_order():
    """Its prompts ran whole (``max_batch_tokens: null``, or a model that never
    chunks), so there is no chunk step to yield after."""
    speculator, peeked, excluded = _speculator(_PROMPT, budgeted=False)

    _build(speculator, 0)

    assert not peeked and excluded == [None]


def test_a_prompt_chain_with_no_token_budget_does_not_yield_from_its_first_step():
    """Inside its chunk Loop but never cut: it speculates on as on main."""
    chunk = _chain_step()
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}), budgeted=False,
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is chunk and not peeked and not excluded
