"""On a GPU: verify steps pre-planned ahead of the previous step's verdict, as the worker's plan
thread runs them, then corrected for the rewind (``StepRunner.settle_preplan``).

The draft-verify toy (``draft_verify_toy``: a chained draft head, the target's verify and the
verdict, one graph) replays at buckets of 1, 2 and 4 rows, three requests in the bucket of 4.
Each decode step is pre-planned on a slot of its own from the lengths the step before it
committed, as if it kept its whole block; the runner then waits for that verdict, rewinds, and
the KV, positions and FlashInfer (an FA3 plan's row lengths rewritten, another backend's plan
made again) correct what they staged. Every KV row a request keeps, in every layer, must equal bitwise what the same
steps planned inline after the rewind wrote, the streams must be plain greedy decode's, and
with any one correction left out the cache differs. Skips without CUDA.
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

import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
MAX_TOKENS = 20
# the KV and attention resources a verify step plans: the target's
KV_KEYS, ATTN_KEYS = (toy.KV,), (toy.ATTN,)


def _run(k: int, plan_ahead: bool, corrupt: int = 2, sample: bool | None = None):
    sub = toy.ToySubmodule(k=k, corrupt=corrupt, sampled_drafts=bool(sample))
    eng = toy.engine(
        sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16,
        sampled_verify=sample is not None, argmax_prefill=sample is not None,
    )
    drv = toy.Driver(eng, sub, plan_ahead=plan_ahead)
    sampling = {"temperature": 0.8, "top_k": 8} if sample is not None else {}
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS, seed=900 + rid, **sampling)
    out = drv.run(list(range(len(PROMPTS))))
    torch.cuda.synchronize()
    return out, eng, drv


def _kept_rows(eng) -> dict:
    """Every KV row each request keeps, per cache and layer: ``[2, kept, heads, dim]``."""
    rows = {}
    for key in KV_KEYS:
        kv = eng._resources[key]
        page = kv.kv_cache.page_size
        for rid, streams in kv._streams.items():
            if not isinstance(rid, int) or rid < 0:
                continue
            stream = streams["main"]
            pages = torch.tensor(stream.page_indices[: -(-stream.stored_len // page)], device=GPU)
            for layer in range(toy.LAYERS):
                view = kv.layer_view(layer)[pages]  # [pages, 2, page, heads, dim]
                kept = view.transpose(0, 1).reshape(2, -1, *view.shape[3:])[:, : stream.stored_len]
                rows[(key, rid, layer)] = kept.clone()
    return rows


def _same_rows(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _decode_steps(drv) -> int:
    return len(drv.leases) - len(PROMPTS)


@pytest.fixture
def patched(monkeypatch):
    """Counts the FA3 length rewrites that took."""
    count = {"patched": 0}
    real = FlashInferPrefillWrapper.patch_kv_lens

    def counting(self, kv_lens):
        ok = real(self, kv_lens)
        count["patched"] += ok
        return ok

    monkeypatch.setattr(FlashInferPrefillWrapper, "patch_kv_lens", counting)
    return count


@pytest.mark.parametrize("cache", [0, 1], ids=["streams", "plan_cache"])
@pytest.mark.parametrize(("k", "corrupt"), [(3, 2), (3, 0), (2, 3)], ids=["k3", "accept_all", "k2"])
def test_steps_planned_ahead_keep_what_steps_planned_after_the_rewind_keep(
    k, corrupt, cache, monkeypatch, patched,
):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(kv_manager, "kv_plan_cache", lambda: cache)
    out, eng, drv = _run(k, plan_ahead=True, corrupt=corrupt)
    assert out == {rid: toy.reference(p, MAX_TOKENS) for rid, p in enumerate(PROMPTS)}
    # every decode step was pre-planned ahead of its verdict; FA3 plans (Hopper) are put right by
    # rewriting their lengths, other backends' planned again
    assert drv.planned_ahead == _decode_steps(drv) > 0
    fa3 = all(
        getattr(w.attn_wrapper, "_backend", None) == "fa3"
        for key in ATTN_KEYS for w in eng._resources[key]._cg_plan_states.values()
    )
    for key in KV_KEYS:
        kv = eng._resources[key]
        assert kv.plan_cache_stats["mismatch"] == 0
        assert (kv.plan_cache_stats["hit"] > 0) == bool(cache)
        for rid in out:
            stream = kv._streams[rid]["main"]
            assert len(stream.page_indices) <= stream.footprint
    behind, behind_eng, _ = _run(k, plan_ahead=False, corrupt=corrupt)
    assert out == behind
    assert _same_rows(_kept_rows(eng), _kept_rows(behind_eng))
    dv = eng._resources[toy.DV]
    assert dv.accepted_hist.tolist() == behind_eng._resources[toy.DV].accepted_hist.tolist()
    if corrupt:
        # rows rejected drafts, so the corrections had lengths to put right
        assert 0 < dv.drafts_accepted < k * dv.rows_verified, dv.stats()
        assert patched["patched"] > 0 or not fa3


@pytest.mark.parametrize("sample", [False, True], ids=["argmax_drafts", "sampled_drafts"])
def test_sampled_steps_planned_ahead_draw_what_steps_planned_after_the_rewind_draw(sample):
    out, eng, drv = _run(3, plan_ahead=True, sample=sample)
    assert drv.planned_ahead == _decode_steps(drv) > 0
    behind, behind_eng, _ = _run(3, plan_ahead=False, sample=sample)
    assert out == behind
    assert _same_rows(_kept_rows(eng), _kept_rows(behind_eng))


@pytest.mark.parametrize("left_out", ["kv", "positions", "attention_lengths"])
def test_each_correction_is_needed(left_out, monkeypatch):
    """With one correction skipped (its staged plan kept at the untrimmed lengths), the rows the
    cache keeps are no longer the ones steps planned after the rewind write."""
    behind, behind_eng, _ = _run(3, plan_ahead=False)
    if left_out == "kv":
        monkeypatch.setattr(KVManager, "correct_preplan", lambda self, verdicts, step, ctx: True)
    elif left_out == "positions":
        monkeypatch.setattr(RopeManager, "correct_preplan", lambda self, verdicts, step, ctx: True)
    else:
        monkeypatch.setattr(FlashInferPrefillWrapper, "patch_kv_lens", lambda self, kv_lens: True)
    out, eng, drv = _run(3, plan_ahead=True)
    assert drv.planned_ahead > 0
    rows, want = _kept_rows(eng), _kept_rows(behind_eng)
    assert not _same_rows(rows, want)
