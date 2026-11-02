"""The captured sampler's log-prob rows against fp32 ``log_softmax``, and
against the eager sampler: padding rows and rows that did not ask. Drives
``SamplerResource`` the way the engine does: plan (the gathers), the sample
captured once and replayed, commit.

    pytest test/sampling_test/test_logprobs_cudagraph.py

Requires a GPU + flashinfer.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)
flashinfer = pytest.importorskip("flashinfer")

from mstar.engine.resources.sampler.config import SAMPLED_LOGPROB, SamplerStep, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.engine.resources.sampler.utils import FilterOrder, processed_logprobs, token_logprobs  # noqa: E402
from mstar.engine.resources.step import SlotLease, StepContext  # noqa: E402

DEV = torch.device("cuda:0")
V = 4096
BS = 4


def _reference(logits, tokens, temperature):
    t = torch.tensor([x if x > 0 else 1.0 for x in temperature], device=DEV)
    lp = (logits.float() / t[:, None]).log_softmax(-1)
    return lp.gather(1, tokens.long()[:, None]).squeeze(1)


class _Captured:
    """One bucket of BS rows: its sample captured, then replayed per step."""

    def __init__(self, res: SamplerResource, rids):
        self.res = res
        self.logits = _logits(99)
        for _ in range(3):  # autotune outside the capture, re-planned as the runner does
            self._plan(rids)
            res.sample(list(rids), self.logits)
        ctx = self._plan(rids)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.tokens = res.sample(list(rids), self.logits)
        res.commit(SamplerStep(apply_penalty=False), ctx)

    def _plan(self, rids):
        # the padding rows past `rids` read the trash row
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
            slot_lease=SlotLease(slot=0, bucket=SimpleNamespace(bs=BS)),
        )
        self.res.plan(SamplerStep(apply_penalty=False), ctx)
        return ctx

    def step(self, rids, logits):
        ctx = self._plan(rids)
        self.logits.copy_(logits)
        self.graph.replay()
        rows = self.res.step_host_rows()[SAMPLED_LOGPROB].cpu()
        self.res.commit(SamplerStep(apply_penalty=False), ctx)
        torch.cuda.synchronize()
        return self.tokens.clone(), rows


def _processed_reference(logits, tokens, temperature, top_k, top_p, penalty=1.0, seen=None):
    """The drawn token under softmax(penalized logits / T) renormed by
    FlashInfer's own top-k, then top-p."""
    logits = logits.float()
    if seen is not None:
        penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
        logits = torch.where(seen, penalized, logits)
    probs = (logits / torch.tensor(temperature, device=DEV)[:, None]).softmax(-1)
    probs = flashinfer.sampling.top_k_renorm_probs(probs, torch.tensor(top_k, device=DEV))
    probs = flashinfer.sampling.top_p_renorm_probs(probs, torch.tensor(top_p, device=DEV))
    return probs.gather(1, tokens.long()[:, None]).squeeze(1).log()


def _resource(processed=False):
    res = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=DEV, enable_logprobs=True,
        enable_processed_logprobs=processed,
    )
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=BS, max_seq_len=1)
    return res


def _logits(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(BS, V, device=DEV, generator=gen).mul_(3).to(torch.bfloat16)


TEMPS = [0.7, 1.3, 0.0]
ASKS = [True, False, True]


@pytest.mark.parametrize("vocab", [4096, 100_003, 248_320])
def test_the_kernel_matches_log_softmax_whether_or_not_the_vocab_splits(vocab):
    gen = torch.Generator(device=DEV).manual_seed(vocab)
    logits = torch.randn(5, vocab, device=DEV, generator=gen).mul_(4).to(torch.bfloat16)
    tokens = torch.randint(vocab, (5,), device=DEV, generator=gen)
    temperature = torch.tensor([0.0, 0.3, 0.7, 1.0, 2.0], device=DEV)
    chosen = token_logprobs(logits, tokens, temperature)
    expected = _reference(logits, tokens, temperature.tolist())
    torch.testing.assert_close(chosen, expected, rtol=1e-5, atol=1e-5)


def test_a_near_certain_token_never_scores_above_zero():
    gen = torch.Generator(device=DEV).manual_seed(0)
    logits = torch.randn(256, 248_320, device=DEV, generator=gen).mul_(2).to(torch.bfloat16)
    logits[:, 7] = logits.max(dim=1).values + 12
    tokens = torch.full((256,), 7, device=DEV)
    assert token_logprobs(logits, tokens, torch.full((256,), 0.3, device=DEV)).max() <= 0
    logits = torch.randn(256, 248_320, device=DEV, generator=gen).mul_(2)
    logits[:, 7] = logits.max(dim=1).values + 4
    top_k = torch.zeros(256, device=DEV, dtype=torch.int32)
    top_p = torch.full((256,), 0.95, device=DEV)
    assert processed_logprobs((logits / 0.7).softmax(-1), tokens, top_k, top_p).max() <= 0


def test_replayed_logprobs_match_log_softmax_with_padding_and_quiet_rows():
    res = _resource()
    for rid, (t, ask) in enumerate(zip(TEMPS, ASKS, strict=True)):
        res.ingest_request(rid, SamplingReqConfig(temperature=t, return_logprobs=ask))
    cap = _Captured(res, [0, 1, 2])
    for seed in range(4):
        logits = _logits(seed)
        tokens, rows = cap.step([0, 1, 2], logits)
        expected = _reference(logits, tokens, TEMPS + [1.0]).cpu()
        torch.testing.assert_close(rows[[0, 2]], expected[[0, 2]], rtol=1e-5, atol=1e-5)
        assert rows[1].isnan(), "a row that did not ask"
        assert rows[3].isnan(), "a padding row"


def test_captured_and_eager_agree_on_a_greedy_row():
    """Greedy draws the same token on both paths, so the scores must match."""
    logits = _logits(7)[:2]
    eager = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=DEV, enable_logprobs=True)
    for rid in (0, 1):
        eager.ingest_request(rid, SamplingReqConfig(temperature=0.0, return_logprobs=True))
    step = SamplerStep(apply_penalty=False)
    ctx = StepContext(request_ids=(0, 1), graph_walk="decode", slot=0, capture=False)
    eager.plan(step, ctx)
    eager_tokens = eager.sample([0, 1], logits)
    eager_rows = eager.step_host_rows()[SAMPLED_LOGPROB].cpu()

    res = _resource()
    for rid in (0, 1):
        res.ingest_request(rid, SamplingReqConfig(temperature=0.0, return_logprobs=True))
    padded = torch.cat([logits, _logits(8)[:2]])
    tokens, rows = _Captured(res, [0, 1]).step([0, 1], padded)

    assert torch.equal(tokens[:2], eager_tokens)
    torch.testing.assert_close(rows[:2], eager_rows)
    torch.testing.assert_close(eager_rows, _reference(logits, eager_tokens, [0.0, 0.0]).cpu())


def _eager_step(res, rids, logits, apply_penalty=False):
    step = SamplerStep(apply_penalty=apply_penalty)
    ctx = StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False)
    res.plan(step, ctx)
    tokens = res.sample(list(rids), logits, apply_penalty=apply_penalty)
    rows = res.step_host_rows()[SAMPLED_LOGPROB].cpu()
    res.commit(step, ctx)
    return tokens, rows


# processed and filtered, raw with the same filters, processed and greedy
MIXED = [
    SamplingReqConfig(temperature=0.7, top_k=20, top_p=0.9, return_logprobs=True, logprobs_mode="processed"),
    SamplingReqConfig(temperature=0.7, top_k=20, top_p=0.9, return_logprobs=True),
    SamplingReqConfig(temperature=0.0, return_logprobs=True, logprobs_mode="processed"),
]


@pytest.mark.parametrize("graph", [False, True])
def test_processed_rows_score_the_distribution_the_token_was_drawn_from(graph):
    res = _resource(processed=True)
    for rid, cfg in enumerate(MIXED):
        res.ingest_request(rid, cfg)
    cap = _Captured(res, [0, 1, 2]) if graph else None
    for seed in range(4):
        logits = _logits(seed)
        if graph:
            tokens, rows = cap.step([0, 1, 2], logits)
        else:
            tokens, rows = _eager_step(res, [0, 1, 2], logits[:3])
        processed = _processed_reference(logits[:1], tokens[:1], [0.7], [20], [0.9]).cpu()
        torch.testing.assert_close(rows[:1], processed, rtol=1e-4, atol=1e-4)
        raw = _reference(logits[1:2], tokens[1:2], [0.7]).cpu()
        torch.testing.assert_close(rows[1:2], raw, rtol=1e-5, atol=1e-5)
        assert rows[2] == 0, "a greedy draw is certain"


@pytest.mark.parametrize("vocab", [4096, 248_320])
def test_the_two_renorm_shortcut_matches_the_full_filter(vocab):
    gen = torch.Generator(device=DEV).manual_seed(vocab)
    probs = (torch.randn(6, vocab, device=DEV, generator=gen) * 4).softmax(-1)
    top_k = torch.tensor([0, 0, 5, 50, 5, 0], device=DEV, dtype=torch.int32)
    top_p = torch.tensor([1.0, 0.9, 1.0, 0.8, 0.5, 0.3], device=DEV)
    tokens = probs.argmax(-1)
    every_row_flashinfer_order = FilterOrder(
        top_p_first=torch.zeros(6, dtype=torch.bool, device=DEV), min_keep=torch.ones(6, dtype=torch.int32, device=DEV),
    )
    torch.testing.assert_close(
        processed_logprobs(probs, tokens, top_k, top_p),
        processed_logprobs(probs, tokens, top_k, top_p, order=every_row_flashinfer_order),
        rtol=1e-4, atol=1e-4,
    )


def test_processed_rows_see_the_repetition_penalty():
    res = SamplerResource(
        vocab_size=V, enable_repetion_penalty=True, device=DEV,
        enable_logprobs=True, enable_processed_logprobs=True,
    )
    res.ingest_request(0, SamplingReqConfig(
        temperature=1.0, top_p=0.95, repetition_penalty=1.5, return_logprobs=True, logprobs_mode="processed",
    ))
    logits = _logits(11)[:1]
    seen = torch.zeros(1, V, dtype=torch.bool, device=DEV)
    for _ in range(4):
        tokens, rows = _eager_step(res, [0], logits, apply_penalty=True)
        expected = _processed_reference(logits, tokens, [1.0], [V], [0.95], penalty=1.5, seen=seen).cpu()
        torch.testing.assert_close(rows, expected, rtol=1e-4, atol=1e-4)
        seen[0, tokens] = True
