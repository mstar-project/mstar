"""On a GPU: verify steps pre-planned ahead of the previous step's verdict, as the worker's plan
thread runs them, then corrected for the rewind (``StepRunner.settle_preplan``).

The draft-verify toy (``draft_verify_toy``: a chained draft head, the target's verify and the
verdict, one graph) and its draft-model variant (``draft_model_toy``: the drafter's k passes on
KV, positions and attention of its own come first) replay at buckets of 1, 2 and 4 rows, three
requests in the bucket of 4. Each decode step is pre-planned on a slot of its own from the
lengths the step before it committed, as if it kept its whole block; the runner then waits for
that verdict, rewinds, and the KV, positions and FlashInfer (FA3: each row's length rewritten
in the plan) correct what they staged. Every KV row a request keeps, in both caches of the
draft model and every layer, must equal bitwise what the same steps planned inline after the
rewind wrote, the streams must be plain greedy decode's, and with any one correction left out
the caches differ. Skips without CUDA.
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

from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
MAX_TOKENS = 20
# the KV and attention resources each toy's verify step plans
KV_KEYS = {"head": (toy.KV,), "draft_model": (toy.KV, dm.DKV)}
ATTN_KEYS = {"head": (toy.ATTN,), "draft_model": (toy.ATTN, dm.DATTN)}


def _run(k: int, plan_ahead: bool, corrupt: int = 2, sample: bool | None = None, kind: str = "head",
         drafter_seed: int | None = 3):
    spec = dict(sampled_verify=sample is not None, argmax_prefill=sample is not None)
    if kind == "head":
        sub = toy.ToySubmodule(k=k, corrupt=corrupt, sampled_drafts=bool(sample))
        eng = toy.engine(sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16,
                         **spec)
        drv = toy.Driver(eng, sub, plan_ahead=plan_ahead)
    else:
        sub = dm.ToyDraftModelSubmodule(k=k, drafter_seed=drafter_seed, sample=bool(sample))
        eng = dm.engine(sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16,
                        **spec)
        drv = dm.Driver(eng, sub, plan_ahead=plan_ahead)
    sampling = {"temperature": 0.8, "top_k": 8} if sample is not None else {}
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS, seed=900 + rid, **sampling)
    out = drv.run(list(range(len(PROMPTS))))
    torch.cuda.synchronize()
    return out, eng, drv


def _kept_rows(eng, kind: str = "head") -> dict:
    """Every KV row each request keeps, per cache and layer: ``[2, kept, heads, dim]``."""
    rows = {}
    for key in KV_KEYS[kind]:
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
@pytest.mark.parametrize(("kind", "k", "corrupt"), [
    ("head", 3, 2), ("head", 3, 0), ("head", 2, 3), ("draft_model", 3, 0), ("draft_model", 2, 0),
], ids=["k3", "accept_all", "k2", "drafter_k3", "drafter_k2"])
def test_steps_planned_ahead_keep_what_steps_planned_after_the_rewind_keep(
    kind, k, corrupt, cache, monkeypatch, patched,
):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(kv_manager, "kv_plan_cache", lambda: cache)
    out, eng, drv = _run(k, plan_ahead=True, corrupt=corrupt, kind=kind)
    assert out == {rid: toy.reference(p, MAX_TOKENS) for rid, p in enumerate(PROMPTS)}
    # every decode step was pre-planned ahead of its verdict, on FA3 plans
    assert drv.planned_ahead == _decode_steps(drv) > 0
    for key in ATTN_KEYS[kind]:
        assert all(w.attn_wrapper._backend == "fa3" for w in eng._resources[key]._cg_plan_states.values())
    for key in KV_KEYS[kind]:
        kv = eng._resources[key]
        assert kv.plan_cache_stats["mismatch"] == 0
        assert (kv.plan_cache_stats["hit"] > 0) == bool(cache)
        for rid in out:
            stream = kv._streams[rid]["main"]
            assert len(stream.page_indices) <= stream.footprint
    behind, behind_eng, _ = _run(k, plan_ahead=False, corrupt=corrupt, kind=kind)
    assert out == behind
    assert _same_rows(_kept_rows(eng, kind), _kept_rows(behind_eng, kind))
    dv = eng._resources[toy.DV]
    assert dv.accepted_hist.tolist() == behind_eng._resources[toy.DV].accepted_hist.tolist()
    if corrupt or kind == "draft_model":
        # rows rejected drafts, so the corrections had lengths to put right
        assert 0 < dv.drafts_accepted < k * dv.rows_verified, dv.stats()
        assert patched["patched"] > 0


@pytest.mark.parametrize("kind", ["head", "draft_model"])
def test_planned_ahead_with_the_cache_checked_against_full_plans(kind, monkeypatch):
    monkeypatch.setattr(kv_manager, "kv_plan_cache", lambda: 2)
    out, eng, drv = _run(3, plan_ahead=True, kind=kind)
    assert out == {rid: toy.reference(p, MAX_TOKENS) for rid, p in enumerate(PROMPTS)}
    for key in KV_KEYS[kind]:
        stats = eng._resources[key].plan_cache_stats
        assert stats["mismatch"] == 0 and stats["hit"] > 0, stats


@pytest.mark.parametrize("kind", ["head", "draft_model"])
@pytest.mark.parametrize("sample", [False, True], ids=["argmax_drafts", "sampled_drafts"])
def test_sampled_steps_planned_ahead_draw_what_steps_planned_after_the_rewind_draw(sample, kind):
    out, eng, drv = _run(3, plan_ahead=True, sample=sample, kind=kind)
    assert drv.planned_ahead == _decode_steps(drv) > 0
    behind, behind_eng, _ = _run(3, plan_ahead=False, sample=sample, kind=kind)
    assert out == behind
    assert _same_rows(_kept_rows(eng, kind), _kept_rows(behind_eng, kind))


def test_the_step_after_a_stop_planned_ahead_stays_in_the_footprint(monkeypatch, caplog):
    """The async worker's wasted step after a ``max_tokens`` stop, pre-admitted from the length
    its predecessor committed (before that step's rejected rows are taken back), reserves up
    to ``2k + 1`` slots past the kept tokens, as many as it writes after the rewind: the
    policy's lookahead holds it. An oracle drafter keeps every draft, so the block that stops
    the reply starts at ``max_tokens - 1`` emitted tokens, the tightest case."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    prompt = PROMPTS[0]
    sub = toy.ToySubmodule(k=3)
    # on the device before capture: a replay reads it in place
    sub.oracle = torch.tensor(prompt + toy.reference(prompt, 64, ignore_eos=True), device=GPU)
    eng = toy.engine(sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16)
    drv = toy.Driver(eng, sub, plan_ahead=True)
    drv.add(0, prompt, 6, ignore_eos=True)
    drv.run([0])
    drv.decode([0])
    torch.cuda.synchronize()
    assert drv.planned_ahead == len(drv.leases) - 1
    assert eng._resources[toy.DV].drafts_accepted == eng._resources[toy.DV].rows_verified * 3
    assert "against a footprint" not in caplog.text
    for key in KV_KEYS["head"]:
        stream = eng._resources[key]._streams[0]["main"]
        assert len(stream.page_indices) <= stream.footprint, key


def test_a_drafter_that_keeps_every_draft_stays_in_both_footprints_planned_ahead(monkeypatch, caplog):
    """The same tightest case for a draft model that is the target (it keeps every draft),
    three requests: the drafter's KV, one token behind, has its own footprint to stay in."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    sub = dm.ToyDraftModelSubmodule(k=3, drafter_seed=None)
    eng = dm.engine(sub, GPU, attn="flashinfer", warmup=True, dtype=torch.float32, kv_dtype=torch.float16)
    drv = dm.Driver(eng, sub, plan_ahead=True)
    rids = list(range(len(PROMPTS)))
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, 6, ignore_eos=True)
    drv.run(rids)
    drv.decode(rids)
    torch.cuda.synchronize()
    assert drv.planned_ahead == _decode_steps(drv)
    assert "against a footprint" not in caplog.text
    for key in KV_KEYS["draft_model"]:
        kv = eng._resources[key]
        for rid in rids:
            stream = kv._streams[rid]["main"]
            assert len(stream.page_indices) <= stream.footprint, (key, rid)


@pytest.mark.parametrize("kind", ["head", "draft_model"])
@pytest.mark.parametrize("left_out", ["kv", "positions", "fa3_lengths"])
def test_each_correction_is_needed(left_out, kind, monkeypatch):
    """With one correction skipped (its staged plan kept at the untrimmed lengths), the rows the
    caches keep are no longer the ones steps planned after the rewind write."""
    behind, behind_eng, _ = _run(3, plan_ahead=False, kind=kind)
    if left_out == "kv":
        monkeypatch.setattr(KVManager, "correct_preplan", lambda self, verdicts, step, ctx: True)
    elif left_out == "positions":
        monkeypatch.setattr(RopeManager, "correct_preplan", lambda self, verdicts, step, ctx: True)
    else:
        monkeypatch.setattr(FlashInferPrefillWrapper, "patch_kv_lens", lambda self, kv_lens: True)
    out, eng, drv = _run(3, plan_ahead=True, kind=kind)
    assert drv.planned_ahead > 0
    rows, want = _kept_rows(eng, kind), _kept_rows(behind_eng, kind)
    assert not _same_rows(rows, want)
