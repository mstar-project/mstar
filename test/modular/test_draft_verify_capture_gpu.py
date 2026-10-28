"""On a GPU: draft-verify steps replayed from captured graphs, with padding rows.

The toy's decode walk is captured at buckets of 1, 2 and 4 rows of k + 1 tokens, against the
FlashInfer attention resource. Three requests replay in the bucket of 4, so one row is
padding, and they keep replaying as they stop and the batch shrinks. The replayed streams
must equal the same steps run eager and plain greedy decode, the padding rows must leave no
verdict behind, and the KV must stay inside its layout. Skips without CUDA.
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

from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402

GPU = torch.device("cuda", 0)
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
MAX_TOKENS = 20


def _run(k: int, corrupt: int, captured: bool, prompts=PROMPTS, batched=True):
    sub = toy.ToySubmodule(k=k, corrupt=corrupt)
    # an fp32 trunk over an fp16 KV, which FlashInfer needs: the toy's margins hold there
    eng = toy.engine(sub, GPU, attn="flashinfer", warmup=captured, dtype=torch.float32, kv_dtype=torch.float16)
    if captured:
        assert eng._submodules[toy.NODE].accelerator_graph_runner is not None, "the decode walk did not capture"
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, MAX_TOKENS)
    out = drv.run(list(range(len(prompts))), batched=batched)
    torch.cuda.synchronize()
    return out, eng, drv


@pytest.fixture(scope="module")
def plain():
    out, _, _ = _run(0, 0, captured=False)
    # the float64 reference, whose top two logits stay apart at every step here
    for rid, prompt in enumerate(PROMPTS):
        margins = []
        assert out[rid] == toy.reference(prompt, MAX_TOKENS, margins=margins)
        assert min(margins) > 0.01
    return out


@pytest.mark.parametrize(("k", "corrupt"), [(3, 0), (3, 2), (2, 1)], ids=["accept", "mixed", "reject_all"])
def test_replayed_verify_steps_emit_what_plain_decode_emits(k, corrupt, plain, monkeypatch):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    out, eng, drv = _run(k, corrupt, captured=True)
    decode_leases = drv.leases[len(PROMPTS):]
    assert decode_leases and all(lease is not None for lease in decode_leases), "a decode step ran eager"
    # three rows replay in the bucket of four, so one is padding
    assert decode_leases[0].bucket.bs == 4 and decode_leases[0].bucket.num_tokens == 4 * (k + 1)
    assert out == plain
    eager, _, _ = _run(k, corrupt, captured=False)
    assert out == eager
    dv = eng._resources[toy.DV]
    assert dv.rows_verified > 0
    # no padding row's verdict was kept
    assert all(rid >= 0 for rid in dv._pending) and all(rid >= 0 for rid in dv._settled)
    kv = eng._resources[toy.KV]
    for rid in out:
        stream = kv._streams[rid]["main"]
        assert len(stream.page_indices) <= stream.footprint


def test_one_request_replays_in_the_bucket_of_one(plain):
    out, _, drv = _run(3, 2, captured=True, prompts=PROMPTS[:1])
    assert all(lease is not None and lease.bucket.bs == 1 for lease in drv.leases[1:])
    assert out[0] == plain[0]
