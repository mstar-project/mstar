"""On a GPU: the DSpark toy (``dspark_toy``) with FlashInfer for the target and the drafter (the
drafter's block non-causal, its last row hidden), captured at buckets of 1, 2 and 4 rows,
three requests replaying in the bucket of 4 with a padding row. The drafter's pass, the heads,
the target's verify, the verdict and the drafter's context write replay as one graph per
bucket. Captured must equal eager, both plain greedy decode; steps planned ahead of their
verdicts keep, in both caches, what steps planned after the rewind keep. Skips without CUDA."""

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

from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = {0: [3, 9, 12, 40], 1: [5, 11], 2: [20, 21, 22, 23, 24, 25]}
MAX_TOKENS = 20


def _run(k: int, captured: bool, plan_ahead: bool = False, **sub_kwargs):
    sub = ds.ToyDSparkSubmodule(k=k, rope_dtype=None, **sub_kwargs)
    eng = ds.engine(sub, GPU, attn="flashinfer", warmup=captured, dtype=torch.float32, kv_dtype=torch.float16)
    drv = ds.Driver(eng, sub, plan_ahead=plan_ahead)
    drv.add_batch(PROMPTS, MAX_TOKENS)
    out = drv.run(list(PROMPTS))
    torch.cuda.synchronize()
    return out, eng, drv


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
