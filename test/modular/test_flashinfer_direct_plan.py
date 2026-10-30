"""A captured wrapper's later plans call FlashInfer's planner directly; they
must leave the wrapper as FlashInfer's own plan does and attend the same.

These are what checks a FlashInfer version for the direct path: run them on
a GPU before adding a version to ``DIRECT_PLAN_FLASHINFER_VERSIONS``."""

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


def test_direct_plan_matches_flashinfer_plan():
    rng = np.random.default_rng(0)
    full, direct = _wrapper(), _wrapper()
    cache = torch.randn(PAGES, 2, PAGE, KV, HD, dtype=torch.bfloat16, device="cuda")
    for step in range(12):
        batch = _batch(rng)
        full._planner_args = None  # forget the direct planner: a full FlashInfer plan
        full.plan(dtype=torch.bfloat16, **batch)
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


def _prefill_wrapper(backend):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return wrappers.FlashInferPrefillWrapper(
        ws, num_qo_heads=QO, num_kv_heads=KV, head_dim=HD, page_size=PAGE,
        batch_size=BS, max_total_tokens=TOKENS, max_num_pages=PAGES,
        device=torch.device("cuda"), accelerator_graph=True, backend=backend,
    )


TOKENS = 256


SINK = PAGES - 1


def _prefill_batch(rng, short=False):
    """A packed step over every captured query row, or a ``short`` one: fewer
    rows, with the last ones padding, as a captured step shorter than its
    bucket is (zero span, one sink page each)."""
    n_pad = int(rng.integers(1, BS // 2 + 1)) if short else 0
    live = BS - n_pad
    total = int(rng.integers(live, TOKENS - n_pad)) if short else TOKENS
    spans = np.concatenate((rng.multinomial(total - live, np.ones(live) / live) + 1, np.zeros(n_pad, int)))
    lens = np.concatenate((spans[:live] + rng.integers(0, 200, live), np.ones(n_pad, int)))
    npages = (lens + PAGE - 1) // PAGE
    indices = np.concatenate((rng.permutation(SINK)[: int(npages[:live].sum())], np.full(n_pad, SINK)))
    t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.int32))  # noqa: E731
    return dict(
        qo_indptr=t(np.concatenate(([0], np.cumsum(spans)))),
        paged_kv_indptr=t(np.concatenate(([0], np.cumsum(npages)))),
        paged_kv_indices=t(indices),
        paged_kv_last_page_len=t((lens - 1) % PAGE + 1), kv_lens=t(lens),
    )


# fa2 is what "auto" resolves to before Hopper; it takes four more planner arguments than fa3
@pytest.mark.parametrize("backend", ["auto", "fa2"])
def test_prefill_direct_plan_matches_flashinfer_plan(backend):
    rng = np.random.default_rng(1)
    full, direct = _prefill_wrapper(backend), _prefill_wrapper(backend)
    cache = torch.randn(PAGES, 2, PAGE, KV, HD, dtype=torch.bfloat16, device="cuda")
    for step in range(12):
        batch = _prefill_batch(rng, short=step % 2 == 1)
        full._planner_key = None  # forget the direct planner: a full FlashInfer plan
        full.plan(causal=True, dtype=torch.bfloat16, **batch)
        direct.plan(causal=True, dtype=torch.bfloat16, **batch)
        if step == 0:
            assert direct._planner_args is not None, "this configuration should take the direct path"
        a, b = full.attn_wrapper, direct.attn_wrapper
        assert list(a._plan_info) == list(b._plan_info)
        for name in ("_qo_indptr_buf", "_paged_kv_indptr_buf", "_paged_kv_last_page_len_buf"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        for name in ("_qo_indptr_last", "_max_q_len", "_max_kv_len"):
            assert getattr(a, name) == getattr(b, name), name
        n = batch["paged_kv_indices"].shape[0]
        assert torch.equal(a._paged_kv_indices_buf[:n], b._paged_kv_indices_buf[:n])
        # eager run takes exactly the step's rows (a replay runs the bucket's)
        q = torch.randn(int(batch["qo_indptr"][-1]), QO, HD, dtype=torch.bfloat16, device="cuda")
        assert torch.equal(full.run(q, cache), direct.run(q, cache))


def test_an_unchecked_flashinfer_version_plans_with_flashinfer(monkeypatch):
    monkeypatch.setattr(wrappers, "DIRECT_PLAN_FLASHINFER_VERSIONS", frozenset({"0.0.0"}))
    wrappers._direct_plan_supported.cache_clear()
    try:
        w = _wrapper()
        for batch in (_batch(np.random.default_rng(s)) for s in range(3)):
            w.plan(dtype=torch.bfloat16, **batch)
        assert w._planner_args is None
    finally:
        wrappers._direct_plan_supported.cache_clear()


def test_direct_plan_only_takes_host_int32_lengths(monkeypatch):
    """The planner reads ``kv_lens`` by raw pointer as host int32; any other
    lengths go through ``plan()``, which converts them. Any FlashInfer version:
    the direct planner is stubbed, only the gate is under test."""
    wrapper = _wrapper()
    batch = _batch(np.random.default_rng(0))
    wrapper.plan(dtype=torch.bfloat16, **batch)  # sets the dtype the gate compares
    calls = []
    monkeypatch.setattr(wrapper, "_plan_direct", lambda *a: calls.append(a[-1]))
    lens = batch["kv_lens"]
    for kv_lens, direct in (
        (lens, True), (lens.long(), False), (lens.cuda(), False),
    ):
        calls.clear()
        wrapper._planner_args = []  # as if the first plan resolved to the direct path
        wrapper.plan(dtype=torch.bfloat16, **{**batch, "kv_lens": kv_lens})
        assert bool(calls) == direct, (kv_lens.dtype, kv_lens.device)
