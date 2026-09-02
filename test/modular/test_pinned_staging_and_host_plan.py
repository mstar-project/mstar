"""CPU tests for the sync-free planning pieces (2026-08-19).

Three host-side rewrites of work that used to run on the device with
stream-draining syncs. Each is pinned against the tensor arithmetic it
replaces — bit-identical results, no CUDA needed:

1. ``pinned()`` — staging tensor semantics (shape, dtype, values, and the
   pageable fallback when there is no CUDA).
2. ``mla_scatter_map_host`` vs the device-side scatter arithmetic that
   ``FlashInferMLAWrapper._plan_scatter_device`` runs, evaluated on CPU
   tensors, over random paged batches (variable lengths, zero-length padding
   rows, page boundaries).
3. ``mtp_greedy_verify_host`` vs ``mtp_greedy_verify`` on random draft /
   target pairs, including the "emitted == target[:n_acc+1]" identity the
   decode step now relies on.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from mstar.model.glm52.components.mtp import (  # noqa: E402
    mtp_greedy_verify,
    mtp_greedy_verify_host,
)
from mstar.utils.flashinfer_utils import mla_scatter_map_host  # noqa: E402
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
    """The arithmetic ``FlashInferMLAWrapper._plan_scatter_device`` runs, on
    CPU tensors (verbatim except for the device)."""
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


def test_mla_scatter_map_host_matches_device_arithmetic():
    rng = random.Random(1234)
    for page_size in (1, 16, 64, 128):
        for _ in range(60):
            qo, kvp, kvi, kvl = _random_paged_batch(rng, page_size)
            host = mla_scatter_map_host(qo, kvp, kvi, kvl, page_size)
            ref = _device_style_scatter(
                torch.tensor(qo, dtype=torch.int32),
                torch.tensor(kvp, dtype=torch.int32),
                torch.tensor(kvi, dtype=torch.int32),
                torch.tensor(kvl, dtype=torch.int32),
                page_size,
            )
            assert host == ref, (page_size, qo, kvp, kvi, kvl)
            assert len(host[0]) == qo[-1]


def test_mla_scatter_map_host_decode_row_lands_on_last_slot():
    # One new token per request: page = last page, offset = (len-1) % ps.
    ps = 64
    qo = [0, 1, 2]
    kvp = [0, 2, 3]
    kvi = [7, 9, 4]
    kvl = [65, 3]
    t2p, t2c = mla_scatter_map_host(qo, kvp, kvi, kvl, ps)
    assert t2p == [9, 4] and t2c == [0, 2]


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
