"""A sampler's presence state survives changes in settings and execution path."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import BucketKey, SamplerStep, SamplingReqConfig, SlotLease, StepContext
from mstar.engine.resources.base import CachedPrefix
from mstar.engine.resources.sampler import utils
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.submodule_base import ARNodeInputs


@pytest.mark.parametrize("ingraph", [False, True])
@pytest.mark.parametrize("vocab", [33, 65])
def test_presence_uses_one_owner_across_eager_captured_and_reused_rows(monkeypatch, ingraph, vocab):
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1" if ingraph else "0")
    monkeypatch.setenv("MSTAR_SAMPLER_SLOTS", "3")
    observed = []

    def draw(logits, *args, **kwargs):
        mask = kwargs.get("seen_token_mask", kwargs.get("seen_tokens"))
        observed.append(None if mask is None else mask.clone())
        return logits.argmax(-1)

    monkeypatch.setattr(utils, "sample_tokens", draw)
    monkeypatch.setattr(utils, "sample_accelerator_graphable_gpu", draw)
    res = SamplerResource(vocab, True, torch.device("cpu"))
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0), SimpleNamespace(slot=1)], 3, 1)
    res.seal_capture_buffers()
    for rid in (1, 2):
        res.ingest_request(rid, SamplingReqConfig(penalize_prompt=rid == 1))
        if ingraph:
            assert res.reserve(rid, "n", "decode", None) is None
    owner, bufs = res._rows, res._cg_buffers
    expected = {rid: torch.zeros(vocab, dtype=torch.bool) for rid in (1, 2)}

    def ctx(rids=(1, 2), slot=None, preplan=False):
        lease = None if slot is None else SlotLease(slot, BucketKey("decode", 3, 3))
        return StepContext(tuple(rids), "decode", 0, False, slot_lease=lease, is_preplan=preplan)

    def sample(rids, tokens, *, tracked=True, kept=None, check=False, captured=False):
        logits = torch.zeros(3 if captured else len(rids), vocab)
        logits.scatter_(1, torch.tensor([*tokens, *([31] if captured else [])]).unsqueeze(1), 1.)
        res.sample([*rids, *([-1] if captured else [])], logits)
        if check:
            torch.testing.assert_close(observed[-1][:len(rids)], torch.stack([expected[rid] for rid in rids]))
        if tracked:
            for rid, token in zip(rids, tokens, strict=True):
                if kept is None or rid in kept:
                    expected[rid][token] = True

    # Reset must drain before the first prompt OR. Prefix and prompt overlap
    # within one signed word and across the word/tail boundaries.
    prefix = torch.tensor([32, vocab - 1, 32])
    res.apply_cached_prefix(1, "n", "decode", ARNodeInputs(input_ids=prefix, input_seq_len=3),
                            CachedPrefix("main", 3, 3))
    prompt = torch.tensor([0, 31, 31])
    first = SamplerStep(apply_penalty=False, prefill_tracked_tokens={1: prompt, 2: prompt})
    res.plan(first, ctx())
    master = owner.seen_words
    expected[1][torch.cat((prefix, prompt))] = True
    sample((1, 2), (2, 2), tracked=False)
    res.commit(first, ctx())
    torch.testing.assert_close(owner.read_seen(owner.indices((1, 2))), torch.stack(list(expected.values())))

    # Inert parameters still retain kept draws for a later enabled penalty.
    res.plan(SamplerStep(), ctx())
    sample((1, 2), (1, 31))
    res.commit(SamplerStep(), ctx())
    for rid in (1, 2):
        res._sampler.set_config(rid, repetition_penalty=1.3)
        bufs.update_request_config(rid, res._sampler._sampling_config[rid])
    current, successor = ctx((2, 1), 0), ctx(slot=1)
    res.plan(SamplerStep(), current)
    res.plan(SamplerStep(), ctx(slot=1, preplan=True))
    sample((2, 1), (0, 5), check=True, captured=True)
    sample((2, 1), (32, vocab - 1), check=True, captured=True)
    res.commit(SamplerStep(), current)
    dropped = SamplerStep(kept_rids=frozenset({1}))
    res.plan(dropped, successor)
    sample((1, 2), (6, 7), kept={1}, check=True, captured=True)
    res.commit(dropped, successor)
    res.plan(SamplerStep(), ctx())
    sample((1, 2), (31, vocab - 1), check=True)
    res.commit(SamplerStep(), ctx())
    torch.testing.assert_close(owner.read_seen(owner.indices((1, 2))), torch.stack(list(expected.values())))
    assert bufs.sampler_for(3, 0).seen_tokens_buf.data_ptr() != bufs.sampler_for(3, 1).seen_tokens_buf.data_ptr()

    old_row = owner.slot_of(1)
    res.remove_request(1)
    res.ingest_request(3, SamplingReqConfig(repetition_penalty=1.3))
    if ingraph:
        assert res.reserve(3, "n", "decode", None) is None
    assert owner.slot_of(3) == old_row
    res.plan(SamplerStep(prefill_tracked_tokens={3: torch.tensor([31, 31])}), ctx((3, 2), 0))
    reused = owner.read_seen(owner.indices((3, 2)))
    assert reused[0].nonzero().flatten().tolist() == [31]
    torch.testing.assert_close(reused[1], expected[2])
    assert owner.seen_words is master
