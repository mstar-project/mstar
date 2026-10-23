"""A captured decode wrapper's later plans call FlashInfer's planner
directly; they must leave the wrapper exactly as FlashInfer's own plan does."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from mstar.engine.resources.attn import wrappers  # noqa: E402
from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper  # noqa: E402

PAGE, BS, QO, KV, HD, PAGES = 16, 8, 8, 2, 128, 512


def _wrapper():
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return FlashInferDecodeWrapper(
        ws, num_qo_heads=QO, num_kv_heads=KV, head_dim=HD, page_size=PAGE,
        batch_size=BS, max_num_pages=PAGES, device=torch.device("cuda"), accelerator_graph=True,
    )


def _batch(rng):
    lens = rng.integers(1, 300, BS)
    npages = (lens + PAGE - 1) // PAGE
    indptr = np.concatenate(([0], np.cumsum(npages))).astype(np.int32)
    indices = rng.permutation(PAGES)[: int(npages.sum())].astype(np.int32)
    last = ((lens - 1) % PAGE + 1).astype(np.int32)
    t = torch.from_numpy
    return dict(paged_kv_indptr=t(indptr), paged_kv_indices=t(indices),
                paged_kv_last_page_len=t(last), kv_lens=t(lens.astype(np.int32)))


def test_direct_plan_matches_flashinfer_plan(monkeypatch):
    rng = np.random.default_rng(0)
    full, direct = _wrapper(), _wrapper()
    cache = torch.randn(PAGES, 2, PAGE, KV, HD, dtype=torch.bfloat16, device="cuda")
    for step in range(12):
        batch = _batch(rng)
        monkeypatch.setattr(wrappers, "_EXP_SLIM_FI_PLAN", False)
        full.plan(dtype=torch.bfloat16, **batch)
        monkeypatch.setattr(wrappers, "_EXP_SLIM_FI_PLAN", True)
        direct.plan(dtype=torch.bfloat16, **batch)
        if step == 0:
            assert direct._planner_args is not None, "this configuration should take the direct path"
        a, b = full.attn_wrapper, direct.attn_wrapper
        assert list(a._plan_info) == list(b._plan_info)
        for name in ("_paged_kv_indptr_buf", "_paged_kv_last_page_len_buf"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        n = batch["paged_kv_indices"].shape[0]
        assert torch.equal(a._paged_kv_indices_buf[:n], b._paged_kv_indices_buf[:n])
        q = torch.randn(BS, QO, HD, dtype=torch.bfloat16, device="cuda")
        assert torch.equal(full.run(q, cache), direct.run(q, cache))


def _prefill_wrapper():
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return wrappers.FlashInferPrefillWrapper(
        ws, num_qo_heads=QO, num_kv_heads=KV, head_dim=HD, page_size=PAGE,
        batch_size=BS, max_total_tokens=TOKENS, max_num_pages=PAGES,
        device=torch.device("cuda"), accelerator_graph=True,
    )


TOKENS = 256


def _prefill_batch(rng):
    spans = rng.multinomial(TOKENS - BS, np.ones(BS) / BS) + 1
    lens = spans + rng.integers(0, 200, BS)
    npages = (lens + PAGE - 1) // PAGE
    t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.int32))  # noqa: E731
    return dict(
        qo_indptr=t(np.concatenate(([0], np.cumsum(spans)))),
        paged_kv_indptr=t(np.concatenate(([0], np.cumsum(npages)))),
        paged_kv_indices=t(rng.permutation(PAGES)[: int(npages.sum())]),
        paged_kv_last_page_len=t((lens - 1) % PAGE + 1), kv_lens=t(lens),
    )


def test_prefill_direct_plan_matches_flashinfer_plan(monkeypatch):
    rng = np.random.default_rng(1)
    full, direct = _prefill_wrapper(), _prefill_wrapper()
    cache = torch.randn(PAGES, 2, PAGE, KV, HD, dtype=torch.bfloat16, device="cuda")
    for step in range(12):
        batch = _prefill_batch(rng)
        monkeypatch.setattr(wrappers, "_EXP_SLIM_FI_PLAN", False)
        full.plan(causal=True, dtype=torch.bfloat16, **batch)
        monkeypatch.setattr(wrappers, "_EXP_SLIM_FI_PLAN", True)
        direct.plan(causal=True, dtype=torch.bfloat16, **batch)
        if step == 0:
            assert direct._planner_args is not None, "this configuration should take the direct path"
        a, b = full.attn_wrapper, direct.attn_wrapper
        assert list(a._plan_info) == list(b._plan_info)
        for name in ("_qo_indptr_buf", "_paged_kv_indptr_buf", "_paged_kv_last_page_len_buf"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        n = batch["paged_kv_indices"].shape[0]
        assert torch.equal(a._paged_kv_indices_buf[:n], b._paged_kv_indices_buf[:n])
        q = torch.randn(TOKENS, QO, HD, dtype=torch.bfloat16, device="cuda")
        assert torch.equal(full.run(q, cache), direct.run(q, cache))
