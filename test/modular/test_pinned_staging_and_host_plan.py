"""CPU tests for the sync-free planning pieces (2026-08-19).

Host-side rewrites of work that used to run on the device with
stream-draining syncs. Each is pinned against the tensor arithmetic it
replaces — bit-identical results, no CUDA needed:

1. ``pinned()`` — staging tensor semantics (shape, dtype, values, and the
   pageable fallback when there is no CUDA).
2. ``paged_scatter_map_host`` (the MLA attention resource's per-token
   scatter map) vs the device-side scatter arithmetic it replaced,
   evaluated on CPU tensors, over random paged batches (variable lengths,
   zero-length padding rows, page boundaries); and ``build_host_plan``
   feeding it from the KV plan's ``SequenceView``s.
3. ``mtp_greedy_verify_host`` vs ``mtp_greedy_verify`` on random draft /
   target pairs, including the "emitted == target[:n_acc+1]" identity the
   decode step relies on.
"""
import random
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()

from mstar.engine.resources.attn.mla import (  # noqa: E402
    MlaSubPlan,
    build_host_plan,
    paged_scatter_map_host,
)
from mstar.engine.resources.kv.plan import SequenceView  # noqa: E402
from mstar.model.glm52.components.mtp import (  # noqa: E402
    mtp_greedy_verify,
    mtp_greedy_verify_host,
)
from mstar.utils.pinned_staging import pinned, to_device_async  # noqa: E402

# ---------------------------------------------------------------- pinned()

def test_pinned_preserves_shape_dtype_values():
    t = pinned([[3, 4], [5, 6], [7, 8]], torch.long)
    assert t.shape == (3, 2) and t.dtype == torch.long
    assert t.tolist() == [[3, 4], [5, 6], [7, 8]]
    u = pinned([0, 4, 8], torch.int32)
    assert u.shape == (3,) and u.dtype == torch.int32 and u.tolist() == [0, 4, 8]
    e = pinned([], torch.int32)
    assert e.shape == (0,)
    if torch.cuda.is_available():
        assert t.is_pinned() and u.is_pinned()


def test_pinned_accepts_cpu_tensor_and_rejects_device():
    src = torch.tensor([1, 2, 3], dtype=torch.int64)
    t = pinned(src, torch.int32)
    assert t.dtype == torch.int32 and t.tolist() == [1, 2, 3]
    if torch.cuda.is_available():
        dev = torch.tensor([1], device="cuda")
        try:
            pinned(dev)
        except ValueError:
            pass
        else:
            raise AssertionError("pinned() must reject device tensors")


def test_to_device_async_cpu_roundtrip():
    t = to_device_async([5, 6, 7], torch.long, torch.device("cpu"))
    assert t.tolist() == [5, 6, 7] and t.dtype == torch.long


# ------------------------------------------------- MLA scatter map on host

def _device_style_scatter(qo_indptr, kv_indptr, kv_indices, kv_len_arr, page_size):
    """The device-side scatter arithmetic the host map replaced (the old
    ``FlashInferMLAWrapper._plan_scatter_device``), on CPU tensors."""
    n_req = qo_indptr.shape[0] - 1
    starts = qo_indptr[:-1].to(torch.int32)
    lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
    total_tokens = int(lens.sum().item())
    seg = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int32), lens)
    intra = torch.arange(total_tokens, dtype=torch.int32) - torch.repeat_interleave(starts, lens)
    start_new = kv_len_arr[seg] - lens[seg]
    g = start_new + intra
    page_off = torch.div(g, page_size, rounding_mode="floor").to(torch.int32)
    off_in_page = (g - page_off * page_size).to(torch.int32)
    abs_page_ptr = kv_indptr[:-1][seg] + page_off
    return kv_indices[abs_page_ptr].to(torch.long).tolist(), off_in_page.to(torch.long).tolist()


def _random_paged_batch(rng, page_size, max_bs=6, max_ctx=300, max_new=40):
    bs = rng.randint(1, max_bs)
    qo = [0]
    kvp = [0]
    kvi = []
    kvl = []
    next_page = 1  # page 0 is the reserved null page in production
    for _ in range(bs):
        # Zero-length rows are the runner's padding slots.
        new = 0 if rng.random() < 0.15 else rng.randint(1, max_new)
        old = rng.randint(0, max_ctx)
        total = old + new
        n_pages = -(-total // page_size) if total else 0
        pages = list(range(next_page, next_page + n_pages))
        rng.shuffle(pages)
        next_page += n_pages
        qo.append(qo[-1] + new)
        kvi.extend(pages)
        kvp.append(kvp[-1] + n_pages)
        kvl.append(total)
    return qo, kvp, kvi, kvl


def test_paged_scatter_map_host_matches_device_arithmetic():
    rng = random.Random(1234)
    for page_size in (1, 16, 64, 128):
        for _ in range(60):
            qo, kvp, kvi, kvl = _random_paged_batch(rng, page_size)
            host = paged_scatter_map_host(qo, kvp, kvi, kvl, page_size)
            ref = _device_style_scatter(
                torch.tensor(qo, dtype=torch.int32),
                torch.tensor(kvp, dtype=torch.int32),
                torch.tensor(kvi, dtype=torch.int32),
                torch.tensor(kvl, dtype=torch.int32),
                page_size,
            )
            assert host == ref, (page_size, qo, kvp, kvi, kvl)
            assert len(host[0]) == qo[-1]


def test_paged_scatter_map_host_decode_row_lands_on_last_slot():
    # One new token per request: page = last page, offset = (len-1) % ps.
    ps = 64
    qo = [0, 1, 2]
    kvp = [0, 2, 3]
    kvi = [7, 9, 4]
    kvl = [65, 3]
    t2p, t2c = paged_scatter_map_host(qo, kvp, kvi, kvl, ps)
    assert t2p == [9, 4] and t2c == [0, 2]


def _random_views(rng, page_size, max_bs=6, max_ctx=300, max_new=40):
    """What the KV plan hands the attention resource: per request, the
    stream's pages (sliced to the declared length) and how many of those
    tokens are new."""
    views = []
    next_page = 1
    for i in range(rng.randint(1, max_bs)):
        new = 0 if rng.random() < 0.15 else rng.randint(1, max_new)
        old = rng.randint(0, max_ctx)
        total = old + new
        n_pages = -(-total // page_size) if total else 0
        pages = list(range(next_page, next_page + n_pages))
        rng.shuffle(pages)
        next_page += n_pages
        views.append(SequenceView(f"r{i}", "main", page_idxs=pages, length=total, to_compute=new))
    return views


def test_build_host_plan_then_scatter_lands_every_new_token_in_its_stream():
    """The plan the attention resource builds from the KV views, fed to the
    scatter map: token j of request i lands on that request's page
    ``(length - new + j) // page_size`` at offset ``% page_size`` — never
    another request's page, never a slot below the stored prefix."""
    rng = random.Random(99)
    for page_size in (1, 8, 64):
        for _ in range(40):
            views = _random_views(rng, page_size)
            default = MlaSubPlan(
                q_lens=tuple(v.to_compute for v in views),
                kv_lens=tuple(v.length for v in views),
            )
            plan = build_host_plan(views, default, page_size)
            assert plan.total_tokens == sum(v.to_compute for v in views)
            assert plan.kv_len_arr == [v.length for v in views]
            assert plan.page_tables == [list(v.page_idxs) for v in views]
            t2p, t2c = paged_scatter_map_host(
                plan.qo_indptr, plan.kv_indptr, plan.kv_indices, plan.kv_len_arr, page_size)
            row = 0
            for v in views:
                base = v.length - v.to_compute
                for j in range(v.to_compute):
                    g = base + j
                    assert t2p[row] == v.page_idxs[g // page_size]
                    assert t2c[row] == g % page_size
                    row += 1
            assert row == len(t2p) == len(t2c)


# ------------------------------------------------------ host greedy verify

def test_host_verify_matches_tensor_verify():
    rng = random.Random(7)
    for _ in range(500):
        k = rng.randint(1, 5)
        vocab = 5
        target = [rng.randrange(vocab) for _ in range(k + 1)]
        # Bias drafts toward matching so every acceptance count is exercised.
        drafts = [t if rng.random() < 0.6 else rng.randrange(vocab)
                  for t in target[:k]]
        n_ref, bonus_ref = mtp_greedy_verify(
            torch.tensor(drafts), torch.tensor(target))
        n = mtp_greedy_verify_host(drafts, target)
        assert n == n_ref
        # The identity the decode step uses: emitted == target[:n_acc+1].
        emitted_ref = drafts[:n_ref] + [int(bonus_ref)]
        assert emitted_ref == target[:n + 1]


def test_host_verify_shape_check():
    try:
        mtp_greedy_verify_host([1, 2], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("k+1 target rows required")
