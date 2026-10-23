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


def _worker(target):
    peeked, excluded = [], []
    fake = SimpleNamespace(
        _can_speculate=lambda batch: True,
        _choose_spec_target=lambda batch: target,
        _try_speculate_next=lambda pending, target: None,
        _tp_lead_needs_marker=lambda pending, speculation: False,
        _span=lambda name: _Span(),
        enable_nvtx=False,
        _phase_period=0,
        request_state=None,
        _rebuild_rows={},
        _dropped={},
        scheduler=SimpleNamespace(
            has_ready_excluding=lambda state, key: peeked.append(key) or True,
            get_next_batch=lambda state, exclude_target: excluded.append(exclude_target),
        ),
    )
    fake._build_speculation = Worker._build_speculation.__get__(fake)
    return fake, peeked, excluded


def _pending():
    return SimpleNamespace(
        batch=SimpleNamespace(request_to_worker_graph={0: 0}), node_name="LLM", graph_walk="mixed",
        node_batch=SimpleNamespace(admit_error=None),
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
