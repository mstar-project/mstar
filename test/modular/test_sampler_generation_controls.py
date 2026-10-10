"""Generation-aware sampler features against a plain-torch reference of
upstream's (MiniCPM-o TTS) logits pipeline:

    temperature -> windowed frequency penalty -> HF top-p (min_tokens_to_keep)
    -> HF top-k -> stop ids floored until ``min_tokens`` -> softmax -> draw

The draw is random, so the comparison is on the distribution handed to
FlashInfer's ``sampling_from_probs`` (recorded by a spy), at every step of a
batched run whose rows sit at different generated counts, windows and
settings, eagerly and replayed from a captured CUDA graph. The reference keeps
its own history from the tokens M* actually drew.

The CPU tests cover the plumbing: ingest refusals, the master rows a config
writes, and the history's slot lifecycle and ring.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402

from mstar.engine.resources import (  # noqa: E402
    BucketKey,
    SamplerStep,
    SamplingReqConfig,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.sampler.config import SamplerSpec  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.engine.resources.sampler.utils import (  # noqa: E402
    FilterOrder,
    GenerationHistory,
    HistoryRows,
    SamplerBuffers,
    SamplingConfig,
    advance_history,
    apply_min_tokens_floor,
    filter_top_k_top_p,
    fused_temperature_softmax,
)

CPU = torch.device("cpu")
VOCAB = 6562
EOS = 6561
WINDOW = 16

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer sampling requires CUDA"
)


# ── reference ───────────────────────────────────────────────────────────


@dataclass
class RefRow:
    """One request's settings and what it has generated, in upstream's terms."""
    temperature: float = 0.8
    window: int = 0
    penalty: float = 1.0
    min_tokens: int = 0
    top_p: float = 1.0
    top_k: int = 0
    min_keep: int = 1
    top_p_first: bool = True
    generated: list[int] = field(default_factory=list)

    def config(self) -> SamplingReqConfig:
        return SamplingReqConfig(
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            repetition_penalty=self.penalty, repetition_window=self.window,
            min_tokens=self.min_tokens, top_p_first=self.top_p_first,
            top_p_min_keep=self.min_keep,
        )


def _hf_top_p(x: torch.Tensor, p: float, min_keep: int) -> torch.Tensor:
    """transformers.TopPLogitsWarper, one row."""
    sorted_logits, order = torch.sort(x, descending=False)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - p)
    remove[-min_keep:] = False
    return x.masked_fill(remove.scatter(0, order, remove), float("-inf"))


def _hf_top_k(x: torch.Tensor, k: int, min_keep: int) -> torch.Tensor:
    """transformers.TopKLogitsWarper, one row."""
    k = min(max(k, min_keep), x.shape[-1])
    return x.masked_fill(x < torch.topk(x, k)[0][-1], float("-inf"))


def reference_probs(logits: torch.Tensor, row: RefRow, stop_ids=(EOS,)) -> torch.Tensor:
    """Upstream's distribution before the draw, for one row of ``logits``."""
    x = logits.float().clone()
    if row.temperature == 0:
        # greedy: the processors that change the argmax, then a one-hot
        x = _penalise(x, row)
        if len(row.generated) < row.min_tokens:
            x[list(stop_ids)] = float("-inf")
        return torch.nn.functional.one_hot(x.argmax(), x.shape[-1]).float()
    x = x / row.temperature
    x = _penalise(x, row)
    if row.top_p_first:
        if row.top_p < 1:
            x = _hf_top_p(x, row.top_p, row.min_keep)
        if row.top_k > 0:
            x = _hf_top_k(x, row.top_k, row.min_keep)
    else:
        # FlashInfer's order: top-k, then top-p over the renormalised top-k
        if row.top_k > 0:
            x = _hf_top_k(x, row.top_k, 1)
        if row.top_p < 1:
            x = _hf_top_p(x, row.top_p, row.min_keep)
    if len(row.generated) < row.min_tokens:
        x[list(stop_ids)] = float("-inf")
    return x.softmax(dim=-1)


def _penalise(x: torch.Tensor, row: RefRow) -> torch.Tensor:
    """Upstream's CustomRepetitionPenaltyLogitsProcessorRepeat, one row."""
    recent = row.generated[-row.window:] if row.window else []
    if not recent:
        return x
    freq = torch.bincount(torch.tensor(recent), minlength=x.shape[-1]).to(x.device)
    alpha = torch.pow(torch.tensor(row.penalty, device=x.device), freq.float())
    return torch.where(x < 0, x * alpha, x / alpha)


def _logits(bs: int, step: int, device) -> torch.Tensor:
    """Peaked rows over a small favoured set, so tokens repeat inside the
    window, EOS competes, and top-p sometimes keeps fewer than 3 tokens."""
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randn(bs, VOCAB, generator=g)
    x[:, :24] += 4.0
    x[:, EOS] += 6.0
    scale = torch.tensor([1.0, 2.5, 1.5, 4.0, 1.0, 2.0, 1.0, 3.0])[:bs, None]
    return (x * scale).to(device)


# ── CPU: plumbing ───────────────────────────────────────────────────────


def _resource(**kwargs) -> SamplerResource:
    return SamplerResource(
        vocab_size=None, enable_repetion_penalty=True, device=CPU, **kwargs,
    )


@pytest.mark.parametrize("cfg, spec, match", [
    (dict(repetition_window=16, repetition_penalty=1.05), {}, "max_repetition_window=0"),
    (dict(repetition_window=32, repetition_penalty=1.05),
     dict(max_repetition_window=16), "max_repetition_window=16"),
    (dict(repetition_window=8, repetition_penalty=0.0),
     dict(max_repetition_window=16), "must be > 0"),
    (dict(min_tokens=5), {}, "no min_tokens_stop_ids"),
    (dict(top_p_first=True), {}, "enable_top_p_first=False"),
    (dict(top_p_min_keep=3), {}, "enable_top_p_first=False"),
    (dict(top_p_min_keep=0), dict(enable_top_p_first=True), ">= 1"),
    (dict(min_tokens=-1), dict(min_tokens_stop_ids=(EOS,)), ">= 0"),
])
def test_ingest_refuses_what_the_spec_cannot_do(cfg, spec, match):
    res = _resource(**spec)
    with pytest.raises(ValueError, match=match):
        res.ingest_request("r", SamplingReqConfig(**{"repetition_penalty": 1.2, **cfg}))
    assert "r" not in res._sampler._sampling_config
    assert "r" not in res._penalty_rids
    if res._history is not None:
        assert "r" not in res._history._rid_to_slot


def test_cpu_device_refuses_the_cuda_only_features():
    # stands in for XPU: the processors are FlashInfer-backed
    res = _resource(max_repetition_window=16)
    with pytest.raises(ValueError, match="CUDA-only"):
        res.ingest_request("r", SamplingReqConfig(repetition_window=16, repetition_penalty=1.05))
    res.ingest_request("plain", SamplingReqConfig(repetition_penalty=1.2))


def test_defaults_build_nothing_new():
    res = _resource()
    assert res._history is None and res._sampler.history is None
    res.ingest_request("r", SamplingReqConfig())
    cfg = res._sampler._sampling_config["r"]
    assert not cfg.uses_history and not cfg.uses_filter_order
    bufs = SamplerBuffers.allocate(max_batch_size=4, device=CPU)
    view = bufs.slice_for_bs(2)
    assert view["history"] is None and view["order"] is None


def test_the_filter_order_alone_keeps_no_history():
    res = _resource(enable_top_p_first=True)
    assert res._history is None


def test_spec_forwards_the_capabilities():
    info = SimpleNamespace(device=CPU, joint_comm_group=None)
    spec = SamplerSpec(
        resource_key="s", nodes={"tts"}, vocab_size=None,
        max_repetition_window=16, min_tokens_stop_ids=(EOS,), enable_top_p_first=True,
    )
    res = SamplerResource.build(spec, info)
    assert res._history.window == 16
    assert res._stop_ids.tolist() == [EOS]
    assert res._enable_top_p_first


def test_a_windowed_penalty_does_not_engage_the_presence_mask():
    res = _resource(max_repetition_window=16)
    res._sampler.device = torch.device("cuda")  # past the CUDA-only refusal
    res.ingest_request("w", SamplingReqConfig(repetition_window=16, repetition_penalty=1.05))
    res.ingest_request("p", SamplingReqConfig(repetition_penalty=1.2))
    assert res._penalty_rids == {"p"}

    bufs = SamplerBuffers.allocate(
        max_batch_size=4, device=CPU, history=GenerationHistory(16, CPU),
    )
    bufs._write_master_row(1, SamplingConfig(repetition_window=16, repetition_penalty=1.05))
    assert bufs.rep_penalty.master[1] == 1.0  # the mask's kernel is inert for it
    assert bufs.window.master[1] == 16
    assert bufs.window_penalty.master[1] == pytest.approx(1.05)
    bufs._write_master_row(2, SamplingConfig(repetition_penalty=1.2))
    assert bufs.rep_penalty.master[2] == pytest.approx(1.2)
    assert bufs.window_penalty.master[2] == 1.0


def test_history_slots_are_recycled_with_their_count_zeroed():
    hist = GenerationHistory(window=4, device=CPU, capacity=2)  # one usable slot + scratch
    hist.register("a")
    rows, tokens, count = hist.read(["a"])
    h = HistoryRows(count=count, tokens=tokens, window=torch.tensor([3], dtype=torch.int32))
    for tok in (5, 6, 7, 8):
        advance_history(h, torch.tensor([tok]))
    hist.write(rows, h.tokens, h.count)
    assert hist.read(["a"])[2].tolist() == [4]
    # a ring of the request's window (3) inside the node's (4): 8 overwrote 5
    assert hist.read(["a"])[1].tolist() == [[8, 6, 7, -1]]

    hist.register("b")  # grows past the initial capacity
    hist.unregister("a")
    hist.register("c")  # takes a's slot back
    assert hist._rid_to_slot["c"] == 1
    assert hist.read(["c", "b"])[2].tolist() == [0, 0]
    assert hist.read(["c"])[1].tolist() == [[-1, -1, -1, -1]]  # a's tokens are gone
    assert 0 not in hist._rid_to_slot.values()  # the scratch row stays unowned


@needs_cuda
@pytest.mark.parametrize("vocab", [6562, 40000])  # the single-block and the split-vocab kernels
def test_softmax_kernel_penalty_and_greedy_floor_match_upstream(vocab):
    """The windowed penalty and the greedy rows' floor inside
    ``fused_temperature_softmax``, against the reference, for rows with
    different windows inside one ring width, a ring that has wrapped, an empty
    history, and a greedy row under its floor whose argmax is a stop id."""
    dev = torch.device("cuda")
    rows = [
        RefRow(window=16, penalty=1.05, generated=[3, 3, 7] + [3] * 20),
        RefRow(window=4, penalty=1.3, generated=[1, 2, 2, 9, 9, 9]),
        RefRow(window=16, penalty=1.05, generated=[]),
        RefRow(window=0, penalty=1.0, generated=[1, 2]),
        RefRow(temperature=0.0, window=4, penalty=1.3, min_tokens=8, generated=[5, 5]),
    ]
    torch.manual_seed(0)
    logits = torch.randn(len(rows), vocab, device=dev) * 3
    logits[4, EOS % vocab] = 50.0  # the greedy row's argmax is a stop id it may not pick yet
    width = 16
    ring = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        for t, tok in enumerate(r.generated):
            if r.window:
                ring[i, t % r.window] = tok
    h = HistoryRows(
        count=torch.tensor([len(r.generated) for r in rows], dtype=torch.int32, device=dev),
        tokens=ring.to(dev),
        window=torch.tensor([r.window for r in rows], dtype=torch.int32, device=dev),
        window_penalty=torch.tensor([r.penalty for r in rows], device=dev),
        min_tokens=torch.tensor([r.min_tokens for r in rows], dtype=torch.int32, device=dev),
        stop_ids=torch.tensor([EOS % vocab], device=dev),
    )
    temperature = torch.tensor([r.temperature for r in rows], device=dev)
    got = fused_temperature_softmax(logits, temperature, include_greedy=True, history=h)
    for i, r in enumerate(rows):
        want = reference_probs(logits[i], dataclasses.replace(r, top_p=1.0, top_k=0), stop_ids=(EOS % vocab,))
        if r.temperature == 0:
            assert got[i].argmax() == want.argmax() != EOS % vocab
        else:
            torch.testing.assert_close(got[i], want, rtol=1e-4, atol=1e-7)


# ── GPU: against the reference, eager and captured ─────────────────────


def _tts_rows() -> dict[str, RefRow]:
    """MiniCPM-o's TTS settings, plus rows that differ from it in one way each."""
    tts = dict(temperature=0.8, window=WINDOW, penalty=1.05, min_tokens=50,
               top_p=0.85, top_k=25, min_keep=3)
    return {
        "tts": RefRow(**tts),
        "short_floor": RefRow(**{**tts, "min_tokens": 3, "window": 4, "penalty": 1.3}),
        "top_k_first": RefRow(**{**tts, "top_p_first": False, "min_keep": 1, "min_tokens": 0}),
        "greedy": RefRow(**{**tts, "temperature": 0.0, "min_tokens": 8}),
        "plain": RefRow(temperature=0.7, top_k=40, top_p=0.9),
    }


class _ProbsSpy:
    """Records the distribution each ``sampling_from_probs`` call draws from."""

    def __init__(self, monkeypatch):
        import flashinfer

        self.seen: list[torch.Tensor] = []
        real = flashinfer.sampling.sampling_from_probs

        def spy(probs, *args, **kwargs):
            self.seen.append(probs)
            return real(probs, *args, **kwargs)

        monkeypatch.setattr(flashinfer.sampling, "sampling_from_probs", spy)


def _capable(device) -> SamplerResource:
    return SamplerResource(
        vocab_size=None, enable_repetion_penalty=True, device=device,
        max_repetition_window=WINDOW, min_tokens_stop_ids=(EOS,),
        enable_top_p_first=True,
    )


def _check(probs: torch.Tensor, rids: list[str], rows: dict[str, RefRow], logits, step):
    for i, rid in enumerate(rids):
        want = reference_probs(logits[i], rows[rid])
        got = probs[i]
        assert torch.equal(got > 0, want > 0), (
            f"step {step}, {rid}: kept {int((got > 0).sum())} tokens, "
            f"upstream keeps {int((want > 0).sum())}"
        )
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def _ctx(rids, lease=None) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
        slot_lease=lease,
    )


STEPS = 60


@needs_cuda
def test_eager_matches_upstream_step_by_step(monkeypatch):
    pytest.importorskip("flashinfer")
    dev = torch.device("cuda")
    spy = _ProbsSpy(monkeypatch)
    res = _capable(dev)
    rows = _tts_rows()
    for rid, row in rows.items():
        res.ingest_request(rid, row.config())
    # rows join at different steps, so the batch spans generated counts
    joins = {"tts": 0, "short_floor": 0, "top_k_first": 5, "greedy": 12, "plain": 20}
    step_kind = SamplerStep(apply_penalty=False)
    for step in range(STEPS):
        rids = [r for r in rows if joins[r] <= step]
        logits = _logits(len(rids), step, dev)
        res.plan(step_kind, _ctx(rids))
        spy.seen.clear()
        tokens = res.sample(rids, logits).tolist()
        res.commit(step_kind, _ctx(rids))
        assert len(spy.seen) == 1
        _check(spy.seen[0], rids, rows, logits, step)
        for rid, tok in zip(rids, tokens, strict=True):
            rows[rid].generated.append(tok)
    # the floor was live, then lifted, for the 50-token rows
    assert EOS not in rows["tts"].generated[:50]


@needs_cuda
def test_captured_graph_matches_upstream_step_by_step(monkeypatch):
    """Buckets of 4 holding 3 real rows, double-buffered over two slots with
    each step pre-planned ahead and promoted, as the async worker runs it. A
    row is swapped out mid-run for a fresh request, which must start from an
    empty history in the reused slot. Each request's first token is drawn
    eagerly, as a prefill's would be."""
    pytest.importorskip("flashinfer")
    dev = torch.device("cuda")
    spy = _ProbsSpy(monkeypatch)
    res = _capable(dev)
    bs = 4
    res.build_cuda_graph_buffers(
        [SimpleNamespace(slot=0), SimpleNamespace(slot=1)], max_bs=bs, max_seq_len=1,
    )
    rows = _tts_rows()
    live = ["tts", "short_floor", "greedy"]
    for rid in live:
        res.ingest_request(rid, rows[rid].config())
    step_kind = SamplerStep(apply_penalty=False)
    bucket = BucketKey(graph_walk="decode", bs=bs, num_tokens=bs)
    leases = [SlotLease(slot=i, bucket=bucket) for i in range(2)]
    padded = live + ["__pad__"]

    def advance(rids, tokens):
        for rid, tok in zip(rids, tokens, strict=False):
            rows[rid].generated.append(tok)

    def eager_step(step):
        logits = _logits(len(live), step, dev)
        res.plan(step_kind, _ctx(live))
        spy.seen.clear()
        tokens = res.sample(live, logits).tolist()
        res.commit(step_kind, _ctx(live))
        _check(spy.seen[0], live, rows, logits, step)
        advance(live, tokens)

    eager_step(0)

    static = torch.zeros(bs, VOCAB, device=dev)
    graphs, outs, probs = [], [], []
    for lease in leases:
        res.plan(step_kind, _ctx(live, lease))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            res.sample(padded, static)  # warmup: autotune, lazy init
        torch.cuda.current_stream().wait_stream(side)
        # the warmup advanced the per-step rows; regather before capturing
        res.plan(step_kind, _ctx(live, lease))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append(res.sample(padded, static))
        graphs.append(graph)
        probs.append(spy.seen[-1])

    for step in range(1, STEPS):
        if step == 25:  # swap a row: the new request reuses the freed slot
            res.remove_request("short_floor")
            live[1] = "top_k_first"
            res.ingest_request("top_k_first", rows["top_k_first"].config())
            eager_step(step)
            continue
        i = step % 2
        ctx = _ctx(live, leases[i])
        res.plan(step_kind, StepContext(
            request_ids=tuple(live), graph_walk="decode", slot=0, capture=False,
            is_preplan=True, slot_lease=leases[i],
        ))
        assert res._preplanned
        res.plan(step_kind, ctx)  # promotes the pre-plan
        assert not res._preplanned and res._cg_sampler is not None
        logits = _logits(bs, step, dev)
        static.copy_(logits)
        graphs[i].replay()
        tokens = outs[i].tolist()
        res.commit(step_kind, ctx)
        _check(probs[i], live, rows, logits, step)
        advance(live, tokens)
    assert len(rows["top_k_first"].generated) == STEPS - 25
    counts = res._history.read(live)[2].tolist()
    assert counts == [len(rows[r].generated) for r in live]


@needs_cuda
def test_min_keep_changes_peaked_rows():
    """With top-p 0.85, a row whose top token holds >= 85% keeps only it
    unless min_keep raises the count: HF keeps 3."""
    pytest.importorskip("flashinfer")
    dev = torch.device("cuda")
    logits = torch.full((2, 64), -5.0, device=dev)
    logits[:, 0] = 5.0
    logits[:, 1] = 1.0
    logits[:, 2] = 0.5
    probs = logits.softmax(dim=-1)
    order = FilterOrder(
        top_p_first=torch.tensor([True, True], device=dev),
        min_keep=torch.tensor([1, 3], dtype=torch.int32, device=dev),
    )
    out = filter_top_k_top_p(
        probs, torch.full((2,), 25, dtype=torch.int32, device=dev),
        torch.full((2,), 0.85, device=dev), order,
    )
    assert (out > 0).sum(dim=-1).tolist() == [1, 3]


@needs_cuda
@pytest.mark.parametrize("captured", [False, True])
def test_unfiltered_steps_skip_top_k_and_top_p(monkeypatch, captured):
    """``SamplerStep(apply_filters=False)`` draws from the temperature/penalty
    distribution with only the min-tokens floor applied (upstream's TTS samples
    its first code so); the next, filtered steps are unaffected. Captured, the
    unfiltered step replays a graph captured with that setting, its rows loaded
    by a pre-plan the next plan promotes, as the async worker runs it."""


    pytest.importorskip("flashinfer")
    dev = torch.device("cuda")
    spy = _ProbsSpy(monkeypatch)
    res = _capable(dev)
    rows = _tts_rows()
    live = ["tts", "short_floor"]
    bs = len(live)
    if captured:
        # before ingest: ingesting registers a request's settings with the graph buffers
        res.build_cuda_graph_buffers([SimpleNamespace(slot=0)], max_bs=bs, max_seq_len=1)
    for rid in live:
        res.ingest_request(rid, rows[rid].config())
    unfiltered, filtered = SamplerStep(apply_penalty=False, apply_filters=False), SamplerStep(apply_penalty=False)

    def check_and_advance(probs, logits, tokens, step):
        want = {r: (dataclasses.replace(rows[r], top_k=0, top_p=1.0) if step == 0 else rows[r]) for r in live}
        _check(probs, live, want, logits, step)
        for rid, tok in zip(live, tokens, strict=True):
            assert len(rows[rid].generated) >= rows[rid].min_tokens or tok != EOS
            rows[rid].generated.append(tok)

    start = 0
    if captured:
        lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=bs, num_tokens=bs))
        static = torch.zeros(bs, VOCAB, device=dev)
        res.plan(unfiltered, _ctx(live, lease))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            res.sample(live, static)  # warmup
        torch.cuda.current_stream().wait_stream(side)
        res.plan(unfiltered, _ctx(live, lease))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = res.sample(live, static)
        probs = spy.seen[-1]  # the graph's buffer: after a replay, the replay's distribution
        res.plan(unfiltered, StepContext(
            request_ids=tuple(live), graph_walk="decode", slot=0, capture=False,
            is_preplan=True, slot_lease=lease,
        ))
        res.plan(unfiltered, _ctx(live, lease))  # promotes the pre-plan
        logits = _logits(bs, 0, dev)
        static.copy_(logits)
        graph.replay()
        tokens = out.tolist()
        res.commit(unfiltered, _ctx(live, lease))
        check_and_advance(probs, logits, tokens, 0)
        start = 1
    for step in range(start, 4):
        kind = unfiltered if step == 0 else filtered
        logits = _logits(len(live), step, dev)
        res.plan(kind, _ctx(live))
        spy.seen.clear()
        tokens = res.sample(live, logits).tolist()
        res.commit(kind, _ctx(live))
        check_and_advance(spy.seen[0], logits, tokens, step)


@needs_cuda
@pytest.mark.parametrize("vocab", [6562, 1000, 16384])
def test_fused_filter_matches_upstream(vocab):
    """The one-kernel filter against HF's warpers (``reference_probs``) row by row,
    over mixed orders, k, p (including 1), min_keep, peaked and flat rows, and
    the floor, including a row whose kept mass is all stop ids (which keeps its
    unfloored distribution). FlashInfer's renorm path is not the reference: at
    p = 1 its float cumulative sum reaches 1 early and drops a peaked row's tail."""
    pytest.importorskip("flashinfer")
    torch.manual_seed(0)
    bs, dev = 48, torch.device("cuda")
    logits = torch.randn(bs, vocab, device=dev) * torch.linspace(0.5, 12, bs, device=dev)[:, None]
    stop = (vocab - 1, 3)
    logits[-1, vocab - 1] = 60.0  # a row whose top-1 is a stop id
    probs = logits.softmax(-1)
    g = torch.Generator(device="cpu").manual_seed(1)
    top_k = torch.randint(0, 40, (bs,), generator=g)
    top_p = torch.rand(bs, generator=g).clamp(0.05, 1.0)
    top_p[::7] = 1.0
    count = torch.randint(0, 100, (bs,), generator=g)
    first = torch.rand(bs, generator=g) < 0.5
    keep = torch.randint(1, 4, (bs,), generator=g)
    order = FilterOrder(top_p_first=first.to(dev), min_keep=keep.to(dev, torch.int32))
    floor = HistoryRows(
        count=count.to(dev, torch.int32), min_tokens=torch.full((bs,), 50, device=dev, dtype=torch.int32),
        stop_ids=torch.tensor(stop, device=dev),
    )
    got = apply_min_tokens_floor(filter_top_k_top_p(probs, top_k.to(dev, torch.int32), top_p.to(dev), order), floor)
    for i in range(bs):
        row = RefRow(temperature=1.0, top_k=int(top_k[i]), top_p=float(top_p[i]), min_keep=int(keep[i]),
                     top_p_first=bool(first[i]), min_tokens=50, generated=[0] * int(count[i]))
        want = reference_probs(logits[i], row, stop_ids=stop)
        if not torch.isfinite(want).all() or want.sum() == 0:  # every kept token a stop id: no floor
            want = reference_probs(logits[i], dataclasses.replace(row, min_tokens=0), stop_ids=stop)
        assert torch.equal(got[i] > 0, want > 0), (i, int((got[i] > 0).sum()), int((want > 0).sum()))
        torch.testing.assert_close(got[i], want, rtol=1e-4, atol=1e-6)
