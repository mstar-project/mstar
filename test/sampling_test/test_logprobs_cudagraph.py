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
pytest.importorskip("flashinfer")

from mstar.engine.resources.sampler.config import SAMPLED_LOGPROB, SamplerStep, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.engine.resources.sampler.utils import token_logprobs  # noqa: E402
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
        ctx = self._plan(rids)
        for _ in range(3):  # autotune outside the capture
            res.sample(list(rids), self.logits)
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


def _resource():
    res = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=DEV, enable_logprobs=True,
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
