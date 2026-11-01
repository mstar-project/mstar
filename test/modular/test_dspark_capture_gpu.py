"""On a GPU: the DSpark toy (``dspark_toy``) with FlashInfer for the target and the drafter (the
drafter's block non-causal, its last row hidden), captured at buckets of 1, 2 and 4 rows,
three requests replaying in the bucket of 4 with a padding row. The drafter's pass, the heads,
the target's verify, the verdict and the drafter's context write replay as one graph per
bucket. Captured must equal eager, both plain greedy decode; steps planned ahead of their
verdicts keep, in both caches, what steps planned after the rewind keep. The same with each
verdict taken back on the device in the step after it (``device_rewind``: the KV lengths of
FA2's or paged FA3's plan, the KV writes and the positions moved in the replay), which on FA2
must keep, bitwise, what steps planned after the host's rewind keep. Skips without CUDA, and
paged FA3's cases without its wheel on an SM90 GPU."""

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
import dspark_toy as ds  # noqa: E402

from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper, fa3_paged_unavailable_reason  # noqa: E402
from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = {0: [3, 9, 12, 40], 1: [5, 11], 2: [20, 21, 22, 23, 24, 25]}
MAX_TOKENS = 20
# the most a kept K/V element may differ between paged FA3 and FA2 runs (fp16 cache)
FA3_VS_FA2_KV = 0.05


def _run(k: int, captured: bool, plan_ahead: bool = False, fi_backend: str = "auto", rewind_kernel: str = "auto",
         **sub_kwargs):
    sub = ds.ToyDSparkSubmodule(k=k, rope_dtype=None, **sub_kwargs)
    eng = ds.engine(sub, GPU, attn="flashinfer", warmup=captured, dtype=torch.float32, kv_dtype=torch.float16,
                    fi_backend=fi_backend, rewind_kernel=rewind_kernel)
    drv = ds.Driver(eng, sub, plan_ahead=plan_ahead)
    drv.add_batch(PROMPTS, MAX_TOKENS)
    out = drv.run(list(PROMPTS))
    torch.cuda.synchronize()
    return out, eng, drv


def _need(rewind_kernel: str) -> None:
    if rewind_kernel == "fa3" and (reason := fa3_paged_unavailable_reason(GPU, toy.HEAD_DIM, torch.float16)):
        pytest.skip(reason)


def _reference():
    return {rid: toy.reference(p, MAX_TOKENS) for rid, p in PROMPTS.items()}


def _kept_rows(eng) -> dict:
    """Every KV row each request keeps, in both caches, every layer."""
    rows = {}
    for key, layers in ((toy.KV, toy.LAYERS), (ds.DKV, 2)):
        kv = eng._resources[key]
        page = kv.kv_cache.page_size
        for rid, streams in kv._streams.items():
            if not isinstance(rid, int) or rid < 0:
                continue
            stream = streams["main"]
            pages = torch.tensor(stream.page_indices[: -(-stream.stored_len // page)], device=GPU)
            for layer in range(layers):
                view = kv.layer_view(layer)[pages]
                kept = view.transpose(0, 1).reshape(2, -1, *view.shape[3:])[:, : stream.stored_len]
                rows[(key, rid, layer)] = kept.clone()
    return rows


@pytest.mark.parametrize("k", [3, 7])
def test_captured_steps_emit_what_eager_steps_and_plain_decode_emit(k):
    captured, ceng, cdrv = _run(k, captured=True)
    eager, eeng, _ = _run(k, captured=False)
    assert captured == eager == _reference()
    # every decode step replayed a graph
    assert all(lease is not None and lease.bucket is not None for lease in cdrv.leases[1:])
    cdv, edv = ceng._resources[toy.DV], eeng._resources[toy.DV]
    assert cdv.accepted_hist.tolist() == edv.accepted_hist.tolist()
    attn = ceng._resources[ds.DATTN]
    assert attn._cg_plan_states and all(isinstance(w, FlashInferPrefillWrapper) for w in attn._cg_plan_states.values())


def test_steps_planned_ahead_keep_what_steps_planned_after_the_rewind_keep(monkeypatch):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    ahead, aeng, adrv = _run(3, captured=True, plan_ahead=True)
    behind, beng, _ = _run(3, captured=True)
    assert ahead == behind == _reference()
    assert adrv.planned_ahead == len(adrv.leases) - 1 > 0
    rows_a, rows_b = _kept_rows(aeng), _kept_rows(beng)
    assert rows_a.keys() == rows_b.keys() and all(torch.equal(rows_a[key], rows_b[key]) for key in rows_a)


@pytest.mark.parametrize("rewind_kernel", ["fa2", "fa3"])
@pytest.mark.parametrize("k", [3, 7])
def test_verdicts_taken_back_on_the_device_replay_what_eager_and_plain_decode_emit(k, rewind_kernel):
    _need(rewind_kernel)
    captured, ceng, cdrv = _run(k, captured=True, device_rewind=True, rewind_kernel=rewind_kernel)
    eager, _, _ = _run(k, captured=False, device_rewind=True, rewind_kernel=rewind_kernel)
    assert captured == eager == _reference()
    assert all(lease is not None and lease.bucket is not None for lease in cdrv.leases[1:])
    dv = ceng._resources[toy.DV]
    # rows rejected drafts, so steps took them back on the device
    assert dv.device_rewind and dv.rows_verified and dv.drafts_accepted < k * dv.rows_verified
    for key in (toy.ATTN, ds.DATTN):
        on_fa3 = [isinstance(w, FA3PagedWrapper) for w in ceng._resources[key]._cg_plan_states.values()]
        assert any(on_fa3) == (rewind_kernel == "fa3")


def test_steps_planned_before_their_verdicts_keep_what_steps_planned_after_them_keep(monkeypatch):
    """Every decode step pre-planned while the step before it is unsettled, its last verdict
    taken back in its replay: both caches keep, bitwise, what steps planned after the host's
    rewind keep (on the same kernel, FA2), inside the footprints."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    device, deng, ddrv = _run(3, captured=True, plan_ahead=True, device_rewind=True, rewind_kernel="fa2")
    host, heng, _ = _run(3, captured=True, fi_backend="fa2")
    assert device == host == _reference()
    assert ddrv.planned_ahead == len(ddrv.leases) - 1 > 0
    for eng in (deng, heng):
        ds.settle_all(eng, list(PROMPTS))
    rows_d, rows_h = _kept_rows(deng), _kept_rows(heng)
    assert rows_d.keys() == rows_h.keys() and all(torch.equal(rows_d[key], rows_h[key]) for key in rows_d)


def test_steps_on_paged_fa3_planned_before_their_verdicts_keep_what_steps_planned_at_launch_keep(monkeypatch):
    """Paged FA3 under device rewind, each decode step pre-planned before the step ahead of it
    settles: both caches keep, bitwise, what the same steps planned when launched keep, and
    within fp16 rounding what FA2 steps planned after the host's rewind keep."""
    _need("fa3")
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    ahead, aeng, adrv = _run(3, captured=True, plan_ahead=True, device_rewind=True, rewind_kernel="fa3")
    inline, ieng, _ = _run(3, captured=True, device_rewind=True, rewind_kernel="fa3")
    host, heng, _ = _run(3, captured=True, fi_backend="fa2")
    assert ahead == inline == host == _reference()
    assert adrv.planned_ahead == len(adrv.leases) - 1 > 0
    for eng in (aeng, ieng, heng):
        ds.settle_all(eng, list(PROMPTS))
    rows_a, rows_i, rows_h = _kept_rows(aeng), _kept_rows(ieng), _kept_rows(heng)
    assert rows_a.keys() == rows_i.keys() == rows_h.keys()
    assert all(torch.equal(rows_a[key], rows_i[key]) for key in rows_a)
    worst = max((rows_a[key].float() - rows_h[key].float()).abs().max().item() for key in rows_a)
    assert worst < FA3_VS_FA2_KV
