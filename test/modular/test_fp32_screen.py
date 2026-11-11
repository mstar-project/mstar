"""The fp32 screen's decisions (test/integration/fp32_screen.py, shared by the served drafter tests) on
synthetic replies and scans: which positions count, the exact rate and tail tests' p-values against their
closed forms, and each tripwire (a truncated or empty reply among them)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("scipy")


def _load():
    # the integration tests are no package; the served tests load it under the same key
    key = "_integration_fp32_screen"
    if key not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "integration" / "fp32_screen.py"
        spec = importlib.util.spec_from_file_location(key, path)
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


fs = _load()
GATES = fs.Gates(p_fail=1e-3, trip_rank=5, trip_deficit=1.5, burst=(3, 16))


class _Ref:
    """Scans by reply: ``marks[ids][j] = (rank, deficit)`` at a departure, fp32's argmax elsewhere."""

    def __init__(self, marks: dict[tuple, dict[int, tuple[int, float]]]):
        self.marks = marks

    def scan(self, prompt: str, ids: list[int], image: bool = False):
        marks = self.marks.get(tuple(ids), {})
        ranks = [marks.get(j, (0, 0.0))[0] for j in range(len(ids))]
        deficits = [marks.get(j, (0, 0.0))[1] for j in range(len(ids))]
        return fs.Scan([0] * len(ids), [1.0] * len(ids), ranks, deficits)


def _reply(ids: list[int], rid: str | None = None) -> dict:
    """A reply streamed a token a chunk, token j received at time j + 1."""
    return {"rid": rid, "ids": list(ids), "chunks": [[j + 1.0, [t]] for j, t in enumerate(ids)]}


def _pair(plain: list[int], other: list[int], rid: str | None = None, label: str = "p") -> fs.Pair:
    return fs.Pair("prompt", False, _reply(plain), _reply(other, rid), label)


def _departures(n: int, k: int) -> fs.Departures:
    return fs.Departures(n, [0.1] * k, 0, [])


def test_a_shared_prefix_counts_once_and_private_departures_follow_the_split():
    plain = list(range(1, 11))
    spec = plain[:6] + [90, 91, 92, 93]
    later = plain[:8] + [70, 71]
    ref = _Ref({tuple(spec): {2: (1, 0.2), 7: (1, 0.3)}, tuple(later): {2: (1, 0.2), 8: (2, 0.4)}})
    # the second copy of spec is the same reply, one sample; later shares its first 6 positions with spec
    pairs = [_pair(plain, spec), _pair(plain, spec), _pair(plain, later)]
    got = fs.departures(pairs, ref, "other", GATES)
    assert got == fs.Departures(14, [0.2, 0.3, 0.4], 2, [])


def test_prompts_and_image_inputs_keep_their_own_samples():
    """The same reply ids to two prompts, or to one prompt with and without an image, share no positions: the
    dedup and the trie key on the prompt and the image as well as the ids."""
    plain, spec = [1, 2, 3, 4], [1, 9, 8, 7]
    ref = _Ref({tuple(spec): {1: (1, 0.2)}})
    pairs = [fs.Pair(prompt, image, _reply(plain), _reply(spec), f"{prompt} {image}")
             for prompt, image in (("a", False), ("b", False), ("a", True))]
    got = fs.departures(pairs, ref, "other", GATES)
    assert got == fs.Departures(12, [0.2] * 3, 3, [])


def test_the_rate_test_is_the_conditional_binomial():
    # equal positions: spec's 6 of the 8 departures under Binomial(8, 1/2)
    p, ratio, (lo, hi) = fs.rate_test(_departures(100, 6), _departures(100, 2))
    assert p == pytest.approx(sum(math.comb(8, k) for k in range(6, 9)) / 2 ** 8, rel=1e-12)
    assert ratio == pytest.approx(3.0) and lo < ratio < hi
    # spec a quarter of the positions: 3 of 4 under Binomial(4, 1/4)
    p, ratio, _ = fs.rate_test(_departures(50, 3), _departures(150, 1))
    assert p == pytest.approx(4 * 0.25 ** 3 * 0.75 + 0.25 ** 4, rel=1e-12)
    assert ratio == pytest.approx(9.0)
    assert fs.rate_test(_departures(50, 0), _departures(50, 0)) == (1.0, 1.0, (0.0, float("inf")))
    p, ratio, _ = fs.rate_test(_departures(100, 20), _departures(100, 0))
    assert p == pytest.approx(0.5 ** 20, rel=1e-9) and ratio == float("inf")


def test_the_tail_test_is_the_hypergeometric_draw_past_plains_q99():
    plain = fs.Departures(1000, [i / 100 for i in range(1, 101)], 0, [])
    spec = fs.Departures(1000, [0.5] * 7 + [2.0] * 3, 0, [])
    p, q, ks, kp = fs.tail_test(spec, plain)
    # torch's linear quantile: 0.99 + 0.01 * 0.01
    assert q == pytest.approx(0.9901, abs=1e-6) and (ks, kp) == (3, 1)
    # 3+ of the 4 deep departures in spec's 10 draws from the 110
    want = sum(math.comb(4, k) * math.comb(106, 10 - k) for k in (3, 4)) / math.comb(110, 10)
    assert p == pytest.approx(want, rel=1e-9)
    assert fs.tail_test(spec, fs.Departures(0, [], 0, [])) == (1.0, float("inf"), 0, 0)


@pytest.mark.parametrize("mark, gates, trips", [
    ((5, 0.2), GATES, 1),  # rank at the tripwire
    ((4, 1.49), GATES, 0),
    ((1, 1.5), GATES, 1),  # deficit at the tripwire
    ((1, 1.5), GATES._replace(trip_deficit=None), 0),  # no deficit tripwire
], ids=["rank", "below", "deficit", "rank-only"])
def test_a_served_token_trips_at_the_rank_or_deficit_wire(mark, gates, trips):
    plain = list(range(20))
    spec = plain[:3] + [50 + j for j in range(17)]
    got = fs.departures([_pair(plain, spec)], _Ref({tuple(spec): {4: mark}}), "other", gates)
    assert len(got.trips) == trips, got.trips


@pytest.mark.parametrize("departed, trips", [((10, 12, 25), 1), ((10, 12, 26), 0), ((9, 12, 25), 0)],
                         ids=["three-within", "third-past-window", "first-before-reload"])
def test_private_departures_right_after_a_reload_trip_the_burst_wire(departed, trips):
    plain = list(range(40))
    spec = plain[:5] + [100 + j for j in range(35)]
    ref = _Ref({tuple(spec): {j: (1, 0.1) for j in departed}})
    # reloaded once the client had 10 tokens (token 9 at t = 10); the server's rid carries a suffix
    events = [[10.5, "arm-c0#2"]]
    got = fs.departures([_pair(plain, spec, rid="arm-c0")], ref, "other", GATES, events)
    assert len(got.trips) == trips, got.trips
    if trips:
        assert got.trips == ["p: 3+ private departures within 16 tokens of a reload at 10"]


def test_departures_before_the_split_are_no_burst():
    plain = list(range(40))
    spec = plain[:30] + [100 + j for j in range(10)]
    ref = _Ref({tuple(spec): {j: (1, 0.1) for j in (10, 12, 14)}})
    got = fs.departures([_pair(plain, spec, rid="arm-c0")], ref, "other", GATES, [[10.5, "arm-c0"]])
    assert got.private == 0 and got.trips == []


STOP = 99


@pytest.mark.parametrize("spec, trips", [
    (list(range(12)), ["p: ended at token 12 on no stop token where its pair went on"]),
    ([], ["p: ended at token 0 on no stop token where its pair went on"]),
    # a stop token where plain went on is a token like any other, scored by the scan
    (list(range(12)) + [STOP], []),
], ids=["truncated", "empty", "stopped"])
def test_a_reply_that_ends_on_no_stop_token_where_plain_goes_on_trips(spec, trips):
    plain = list(range(30)) + [STOP]
    got = fs.departures([_pair(plain, spec)], _Ref({}), "other", GATES)
    assert got.trips == trips
    failed = fs.screen("arm", [([_pair(plain, spec)], None, None)], _Ref({}), GATES)
    assert failed == [f"arm tripwire: {t}" for t in trips]


def test_an_empty_scan_is_well_typed():
    scan = fs.scan_logprobs(torch.zeros(0, 4), [])
    assert scan == fs.Scan([], [], [], [])


def test_the_screen_fails_on_the_arms_gates_and_reports_plains_trips(capsys):
    plain = list(range(30))
    # a clean arm: its one departure is plain's too
    ref = _Ref({tuple(plain): {3: (1, 0.1)}})
    assert fs.screen("arm", [([_pair(plain, plain)], None, None)], ref, GATES) == []
    # a tripwire on plain's own reply is printed beside the arm, not counted against it
    ref = _Ref({tuple(plain): {3: (7, 0.1)}})
    spec = plain[:5] + [100 + j for j in range(25)]
    assert fs.screen("arm", [([_pair(plain, spec)], None, None)], ref, GATES) == []
    assert "plain tripwire (beside arm): p: token 3 at fp32 rank 7" in capsys.readouterr().out
    # the arm's: one tripwire
    spec = plain[:10] + [100 + j for j in range(20)]
    ref = _Ref({tuple(spec): {12: (6, 0.3)}})
    assert fs.screen("arm", [([_pair(plain, spec)], None, None)], ref, GATES) == [
        "arm tripwire: p: token 12 at fp32 rank 6, deficit 0.300"]
    out = capsys.readouterr().out
    assert "split arm p at 10: t_p 10 t_s 100" in out
    # departures at 20 of the arm's 40 positions, plain's none: the rate test fails, the tail test has no pool
    pairs, marks = [], {}
    for c in range(20):
        spec = [1000 * c + j for j in range(2)]
        marks[tuple(spec)] = {1: (1, 0.2)}
        pairs.append(fs.Pair(f"prompt {c}", False, _reply([1000 * c, 1000 * c + 7]), _reply(spec), f"c{c}"))
    failed = fs.screen("arm", [(pairs, None, None)], _Ref(marks), GATES)
    assert failed == [f"arm: departure rate infx plain's, p {0.5 ** 20:.2g}"]


def test_scan_logprobs_ranks_and_deficits_the_served_tokens():
    lp = torch.tensor([[0.0, -1.0, -2.0, -3.0], [-0.5, -0.25, -4.0, -0.75]])
    scan = fs.scan_logprobs(lp, [0, 3])
    assert scan.argmax == [0, 1] and scan.ranks == [0, 2]
    assert scan.gaps == pytest.approx([1.0, 0.25]) and scan.deficits == pytest.approx([0.0, 0.5])


def test_server_events_and_receive_positions():
    log = "\n".join([
        "2026-11-06 01:02:03,456 INFO worker: Reloaded request arm-r0-c3#1 on rank 0",
        "2026-11-06 01:02:04,000 INFO worker: KV short on node=LLM: moved arm-r1-c2 to the host, 4 pages freed for "
        "arm-r0-c3, preempted 1 times (0 earlier moves not logged)",
        "2026-11-06 01:02:05,000 INFO worker: something else",
    ])
    t0 = 1793926923.456  # 2026-11-06 01:02:03.456 UTC
    got = fs.events(log)
    assert [e[1:] for e in got] == [["arm-r0-c3#1"]], "only reloads are read"
    assert got[0][0] == pytest.approx(t0)
    reply = {"chunks": [[1.0, [5, 6]], [2.0, [7]], [3.5, [8, 9, 10]]]}
    assert [fs.token_at(reply, t) for t in (0.5, 1.0, 2.5, 9.0)] == [0, 2, 3, 6]
