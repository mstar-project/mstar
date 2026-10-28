"""On a GPU: verify steps of a node with a draft model, replayed from captured graphs.

The toy's decode walk (``draft_model_toy``: the drafter's k passes, the target's verify and the
verdict, one graph) is captured at buckets of 1, 2 and 4 rows of k + 1 tokens, with the
target and the drafter each on its own FlashInfer attention resource. Three requests replay
in the bucket of 4, so one row is padding, and keep replaying as they stop. The replayed
streams must equal the same steps run eager and plain greedy decode, the drafter must accept
what it accepts eager, and both KVs stay inside their layouts. At temperature > 0, with the
drafter's drafts sampled, the replayed streams equal the eager ones drawn from the same
seeds. Skips without CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flashinfer")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_model_toy as dm  # noqa: E402
import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
MAX_TOKENS = 20


def _run(k: int, captured: bool, drafter_seed: int | None = 3, prompts=PROMPTS, sample: bool | None = None):
    """``sample``: None decodes greedily; False or True verifies at temperature 0.8, top-k 8
    (each request its own seed), the drafter drafting its argmax or sampling."""
    sub = dm.ToyDraftModelSubmodule(k=k, drafter_seed=drafter_seed, sample=bool(sample))
    # an fp32 trunk over an fp16 KV, which FlashInfer needs: the toy's margins hold there
    eng = dm.engine(
        sub, GPU, attn="flashinfer", warmup=captured, dtype=torch.float32, kv_dtype=torch.float16,
        sampled_verify=sample is not None, argmax_prefill=sample is not None,
    )
    if captured:
        assert eng._submodules[toy.NODE].accelerator_graph_runner is not None, "the decode walk did not capture"
    drv = dm.Driver(eng, sub)
    sampling = {"temperature": 0.8, "top_k": 8} if sample is not None else {}
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, MAX_TOKENS, seed=900 + rid, **sampling)
    out = drv.run(list(range(len(prompts))))
    torch.cuda.synchronize()
    return out, eng, drv


def _check(out, eng, drv, k):
    decode_leases = drv.leases[len(PROMPTS):]
    assert decode_leases and all(lease is not None for lease in decode_leases), "a decode step ran eager"
    # three rows replay in the bucket of four, so one is padding
    assert decode_leases[0].bucket.bs == 4 and decode_leases[0].bucket.num_tokens == 4 * (k + 1)
    dv = eng._resources[toy.DV]
    assert all(rid >= 0 for rid in dv._pending) and all(rid >= 0 for rid in dv._settled)
    for key in (toy.KV, dm.DKV):
        kv = eng._resources[key]
        for rid in out:
            stream = kv._streams[rid]["main"]
            assert len(stream.page_indices) <= stream.footprint


@pytest.mark.parametrize(("k", "drafter_seed"), [(3, 3), (3, None), (2, 3)], ids=["k3", "self", "k2"])
def test_replayed_verify_steps_emit_what_plain_decode_emits(k, drafter_seed, monkeypatch):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    out, eng, drv = _run(k, captured=True, drafter_seed=drafter_seed)
    _check(out, eng, drv, k)
    margins = []
    assert out == {rid: toy.reference(p, MAX_TOKENS, margins=margins) for rid, p in enumerate(PROMPTS)}
    assert min(margins) > 0.01
    eager, eager_eng, _ = _run(k, captured=False, drafter_seed=drafter_seed)
    assert out == eager
    dv, eager_dv = eng._resources[toy.DV], eager_eng._resources[toy.DV]
    assert dv.accepted_hist.tolist() == eager_dv.accepted_hist.tolist()
    if drafter_seed is None:
        assert dv.drafts_accepted == k * dv.rows_verified, dv.stats()
    else:
        assert 0 < dv.drafts_accepted < k * dv.rows_verified, dv.stats()


@pytest.mark.parametrize("sample", [False, True], ids=["argmax_drafts", "sampled_drafts"])
def test_replayed_sampled_verify_steps_draw_what_eager_ones_draw(sample, monkeypatch):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    out, eng, drv = _run(3, captured=True, sample=sample)
    _check(out, eng, drv, 3)
    eager, eager_eng, _ = _run(3, captured=False, sample=sample)
    assert out == eager
    assert out != {rid: toy.reference(p, MAX_TOKENS) for rid, p in enumerate(PROMPTS)}
    dv = eng._resources[toy.DV]
    assert 0 < dv.drafts_accepted < 3 * dv.rows_verified
    assert dv.drafts_accepted == eager_eng._resources[toy.DV].drafts_accepted


@pytest.mark.parametrize("drafter_seed", [3, None], ids=["k3", "self"])
def test_one_request_at_a_time_replays_in_the_bucket_of_one(drafter_seed):
    """Requests one after another, each alone in its steps: every decode step replays the
    bucket of one, and each stream is plain greedy decode's."""
    sub = dm.ToyDraftModelSubmodule(k=3, drafter_seed=drafter_seed)
    eng = dm.engine(sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16)
    drv = dm.Driver(eng, sub)
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS)
        out = drv.run([rid])
        assert out[rid] == toy.reference(prompt, MAX_TOKENS)
    torch.cuda.synchronize()
    leases = [lease for lease in drv.leases if lease is not None]
    assert leases and all(lease.bucket.bs == 1 for lease in leases)
