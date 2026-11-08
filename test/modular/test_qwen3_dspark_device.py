"""Qwen3 with a DSpark drafter on a real device: a tiny random-weight target and
drafter on the real Engine, with the device's own attention (FlashInfer, FA3 for
verify), RoPE and sampler, eager and captured, in synchronous and asynchronous
mode, and with torch.compile on and off. The standalone harness for what a
served 8B model would otherwise be the first to run: capture of the verify step
(the draft pass, target, verify and observe in one graph) and of the packed
prefill with taps, resident planning across the drafter's layers in the target's
pool, FA3's resident gather, and the take-back under all of them.

The drafts are read back through a probe on the sampler's ``agree`` (rank 0's
drafts, which every rank writes KV for): a copy into a static buffer, which a
captured graph records and replays, so a replayed step's drafts are checked as
an eager step's are.

Two models: random weights, whose greedy tokens and drafts must match a dense
float64 recomputation on the same weights (by the margin rule on a bf16 device:
where they first differ, the reference's top-2 margin is a near-tie); and a
"chain", whose target's next token is a fixed function of the current one and
whose drafter proposes token + 1, right except where the chain skips, so every
accepted count runs inside the graphs.

The CPU subset (eager, synchronous, float64, `qwen3_cpu`'s stand-ins) always
runs and checks the harness itself; the device cases need CUDA and FlashInfer,
and asynchronous mode FA3's paged kernel (Hopper).

    pytest test/modular/test_qwen3_dspark_device.py -q
"""
from __future__ import annotations

import collections

import pytest
import torch
from qwen3_cpu import (
    NODE,
    STOP,
    Driver,
    draft_block,
    engine,
    make_model,
    patch_cpu,
    reference_drafts,
    reference_greedy,
    write_checkpoints,
)

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.qwen3.submodules import Qwen3LMSubmodule

VOCAB = 512
# head size 128: what FlashInfer and FA3 run
DEVICE_SHAPES = dict(hidden_size=256, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                     intermediate_size=512, vocab_size=VOCAB)
PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4: [88, 90]}
MAX = 24
K = 3


def _cuda_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    return None


def _fa3_reason() -> str | None:
    from mstar.engine.resources.attn.fa3_verify import fa3_verify_unavailable_reason

    return _cuda_reason() or fa3_verify_unavailable_reason(torch.device("cuda"), 128, torch.bfloat16)


class Setup:
    """One case: where it runs, in what, and how close a recomputation must be."""

    def __init__(self, device: str, mode: str = "synchronous", captured: bool = False, compiled: bool = True):
        self.device, self.mode, self.captured, self.compiled = device, mode, captured, compiled
        cpu = device == "cpu"
        self.dtype = torch.float64 if cpu else torch.bfloat16
        # float64 agrees to the last bit; bf16 may flip a near-tie
        self.margin = 1e-9 if cpu else 0.1

    def __repr__(self):
        return f"{self.device}-{self.mode}-{'captured' if self.captured else 'eager'}-compile{int(self.compiled)}"


CPU = [Setup("cpu")]
DEVICE = [
    Setup("cuda"), Setup("cuda", captured=True), Setup("cuda", captured=True, compiled=False),
    Setup("cuda", "asynchronous", captured=True),
]


def _skip(setup: Setup):
    if setup.device == "cpu":
        return
    reason = _fa3_reason() if setup.mode == "asynchronous" else _cuda_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    patch_cpu(monkeypatch)
    # every take-back keeps a row's position at its KV length plus a fixed offset
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    # a few captures, not the production ladders; 8 covers the chain's 5 rows
    monkeypatch.setattr(Qwen3LMSubmodule, "DECODE_CAPTURE_BATCH_SIZES", [1, 2, 4, 8])
    monkeypatch.setattr(Qwen3LMSubmodule, "PREFILL_CAPTURE_BATCH_SIZES", [1, 2, 4, 8])


def _checkpoints(tmp_path, setup: Setup, **dspark):
    if setup.device == "cpu":
        return write_checkpoints(tmp_path, dspark=dspark)
    return write_checkpoints(
        tmp_path, target={**DEVICE_SHAPES, "torch_dtype": "bfloat16"},
        dspark={**DEVICE_SHAPES, "dtype": "bfloat16", **dspark},
    )


def _model(target, dspark, setup: Setup, draft: bool = True):
    return make_model(
        target, draft_block(dspark, K, setup.mode) if draft else None,
        accelerator_graph=setup.captured, prefill_token_buckets=[64, 128],
    )


class DraftProbe:
    """Rank 0's drafts of the last verify step, copied out by ``agree``."""

    def __init__(self, monkeypatch, device, rows: int = 8):
        self.buf = torch.zeros(rows, K, dtype=torch.long, device=device)
        real, probe = SamplerResource.agree, self

        @torch.compiler.disable
        def agree(self, values):
            out = real(self, values)
            if out.dim() == 2 and out.shape[1] == K:
                # the drafts ([n, k]); verify's result rides the same broadcast, wider
                probe.buf[: out.shape[0]].copy_(out)
            return out

        monkeypatch.setattr(SamplerResource, "agree", agree)


class ProbedDriver(Driver):
    """Keeps, per verify step and row, the anchor's position and the drafts."""

    def __init__(self, eng, model, probe: DraftProbe, prompts):
        super().__init__(eng, model)
        self.probe, self.prompts, self.drafts = probe, prompts, []
        # per verify step: whether it replayed a capture on resident tables
        self.resident: list[bool] = []
        self._leased = None
        real = eng.reserve_replay_slot

        def reserve(batch, real=real):
            lease = real(batch)
            if batch.step_context.graph_walk == "decode":
                self._leased = lease is not None and batch.step_context.resident is not None
            return lease

        eng.reserve_replay_slot = reserve

    def step(self, rids):
        anchors = {rid: len(self.prompts[rid]) + len(self.out[rid]) - 1 for rid in rids}
        self._leased = False
        stopped = super().step(rids)
        self.resident.append(self._leased)
        drafts = self.probe.buf[: len(rids)].cpu()
        self.drafts += [(rid, anchors[rid], drafts[i].tolist()) for i, rid in enumerate(rids)]
        return stopped


def _run(setup: Setup, model, prompts, monkeypatch, batched: bool = True, prepare=None):
    if not setup.compiled:
        monkeypatch.setattr(Qwen3LMSubmodule, "disable_torch_compile", True)
    probe = DraftProbe(monkeypatch, setup.device)
    # warmed up as a server is, captures or none: an eager verifying node builds its slot masters there
    eng = engine(model, device=setup.device, dtype=setup.dtype, warmup=setup.device != "cpu", prepare=prepare,
                 slot_masters=setup.device == "cpu")
    driver = ProbedDriver(eng, model, probe, prompts)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, MAX)
    for group in ([list(prompts)] if batched else [[rid] for rid in prompts]):
        driver.prefill(group, prompts)
        driver.run(group)
    return driver, eng._submodules[NODE].submodule


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    at = next((j for j in range(n) if a[j] != b[j]), None)
    return n if at is None and len(a) != len(b) else at


def _check_tokens(model, sub, prompts, out, setup: Setup) -> None:
    for rid, prompt in prompts.items():
        margins: list[float] = []
        ref = reference_greedy(model, prompt, MAX, like=sub, margins=margins)
        at = _first_divergence(ref, out[rid])
        assert at is None or margins[at] <= setup.margin, (rid, at, ref, out[rid])


def _check_drafts(model, sub, prompts, driver, setup: Setup) -> int:
    checked = 0
    for rid, at, drafts in driver.drafts:
        seq = prompts[rid] + driver.out[rid]
        if at + 1 > len(seq) - 1:
            continue  # the reply ended at this step's anchor
        margins: list[float] = []
        ref = reference_drafts(model, seq, at, K, like=sub, margins=margins)
        j = _first_divergence(ref, drafts)
        # past the first difference the Markov head runs from another draft
        assert j is None or margins[j] <= setup.margin, (rid, at, ref, drafts)
        checked += 1
    return checked


@pytest.mark.parametrize("setup", CPU + DEVICE, ids=repr)
def test_greedy_tokens_and_drafts_match_a_dense_recomputation(tmp_path, monkeypatch, setup):
    _skip(setup)
    target, dspark = _checkpoints(tmp_path, setup)
    model = _model(target, dspark, setup)
    driver, sub = _run(setup, model, PROMPTS, monkeypatch)
    _check_tokens(model, sub, PROMPTS, driver.out, setup)
    assert _check_drafts(model, sub, PROMPTS, driver, setup) >= 2 * len(PROMPTS)
    _check_replayed(driver, setup)


def _check_replayed(driver, setup: Setup) -> None:
    """A captured case replays every verify step on resident tables, or the
    accepted counts it checks ran eager."""
    if setup.captured:
        assert driver.resident and all(driver.resident), driver.resident


# the chain's prompts: [4]'s first anchor (5) is a multiple of 5, so its first block accepts nothing
CHAIN_PROMPTS = {**PROMPTS, 5: [4]}


@pytest.mark.parametrize("setup", CPU + DEVICE, ids=repr)
def test_a_chain_exercises_every_accepted_count(tmp_path, monkeypatch, setup):
    """The target's next token is u + 1, or u + 2 after a multiple of 5; the
    drafter proposes u + 1, so a block is accepted up to its first skip. The
    tokens and every step's accepted counts are the pure-Python chain's."""
    _skip(setup)
    target, dspark = _checkpoints(tmp_path, setup, markov_rank=VOCAB if setup.device != "cpu" else 97)
    model = _model(target, dspark, setup)
    vocab = model.config.vocab_size
    driver, sub = _run(setup, model, CHAIN_PROMPTS, monkeypatch, prepare=lambda sub: _chain(sub, vocab))
    want_out, want_accepted = _chain_run(CHAIN_PROMPTS, vocab)
    assert driver.out == want_out
    assert driver.accepted == want_accepted
    assert {0, 1, K} <= set(want_accepted), collections.Counter(want_accepted)
    _check_replayed(driver, setup)


def _chain_run(prompts: dict[int, list[int]], vocab: int) -> tuple[dict[int, list[int]], list[int]]:
    """The Driver's run of the chain, in Python: per step, each live row's
    accepted count (before any cut) in row order, and every row's reply."""
    out = {rid: [_next(prompt[-1], vocab)] for rid, prompt in prompts.items()}
    accepted = []
    live = [rid for rid in prompts if out[rid][-1] != STOP and len(out[rid]) < MAX]
    while live:
        for rid in live:
            prev = out[rid][-1]
            drafts = [(prev + 1 + j) % vocab for j in range(K)]
            kept = 0
            for draft in drafts:
                if draft != _next(prev, vocab):
                    break
                kept, prev = kept + 1, draft
            accepted.append(kept)
            run = [*drafts[:kept], _next(prev, vocab)]
            if STOP in run:
                run = run[: run.index(STOP) + 1]
            out[rid] += run[: MAX - len(out[rid])]
        live = [rid for rid in live if out[rid][-1] != STOP and len(out[rid]) < MAX]
    return out, accepted


def _next(u: int, vocab: int) -> int:
    return (u + 2) % vocab if u % 5 == 0 else (u + 1) % vocab


@torch.no_grad()
def _chain(sub, vocab: int) -> None:
    """Weights whose greedy continuation is `_next`, and a drafter that drafts u + 1:
    every layer's output projections zeroed, so each stream is its token's embedding."""
    for layer in [*sub.lm.layers, *sub.drafter.backbone.layers]:
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    embed = sub.lm.embed_tokens.weight
    unit = embed / embed.norm(dim=-1, keepdim=True)
    head = torch.zeros_like(sub.lm.lm_head.weight)
    for u in range(vocab):
        head[_next(u, vocab)] += 10 * unit[u]
    sub.lm.lm_head.weight.copy_(head)
    drafter = sub.drafter
    drafter.backbone.lm_head.weight.zero_()
    rank = drafter.markov_head.markov_w1.weight.shape[1]
    assert rank == vocab, "the chain drafter's Markov head needs rank = vocabulary"
    drafter.markov_head.markov_w1.weight.copy_(torch.eye(vocab))
    shift = torch.zeros(vocab, vocab)
    shift[(torch.arange(vocab) + 1) % vocab, torch.arange(vocab)] = 10.0
    drafter.markov_head.markov_w2.weight.copy_(shift)


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_plain_decode_on_the_device_matches_the_recomputation(tmp_path, monkeypatch, captured):
    setup = Setup("cuda", captured=captured)
    _skip(setup)
    target, dspark = _checkpoints(tmp_path, setup)
    model = _model(target, dspark, setup, draft=False)
    driver, sub = _run(setup, model, PROMPTS, monkeypatch)
    _check_tokens(model, sub, PROMPTS, driver.out, setup)


def test_synchronous_and_asynchronous_agree_exactly_one_request_at_a_time(tmp_path, monkeypatch):
    """Both on FA3, every step a batch of one: the same tokens and the same drafts."""
    runs = []
    for mode in ("synchronous", "asynchronous"):
        setup = Setup("cuda", mode, captured=True)
        _skip(setup)
        target, dspark = _checkpoints(tmp_path / mode, setup)
        driver, _ = _run(setup, _model(target, dspark, setup), PROMPTS, monkeypatch, batched=False)
        runs.append((driver.out, driver.drafts))
    assert runs[0] == runs[1]


def test_the_cpu_cases_run_here():
    """The harness itself is checked wherever the suite runs."""
    assert CPU[0].device == "cpu" and NODE == "LLM"
