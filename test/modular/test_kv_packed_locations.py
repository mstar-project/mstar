"""A packed step's KV write addressing, computed on the host from the plan's
index arrays (``packed_token_locations``), against the arrays' definition: row
``i`` holds ``kv_indptr[i + 1] - kv_indptr[i]`` pages, the last of them filled
to ``last_page_len[i]``, and its ``qo_indptr[i + 1] - qo_indptr[i]`` new tokens
are the last ones of that length."""
import random

import numpy as np
import pytest

from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, packed_token_locations


def _reference(indptrs, page_size):
    qo, kv = indptrs.qo_indptr.tolist(), indptrs.paged_kv_indptr.tolist()
    last, pages = indptrs.paged_kv_last_page_len.tolist(), indptrs.paged_kv_indices.tolist()
    out_pages, out_offsets = [], []
    for i in range(len(qo) - 1):
        length = (kv[i + 1] - kv[i] - 1) * page_size + last[i]
        new = qo[i + 1] - qo[i]
        for pos in range(length - new, length):
            out_pages.append(pages[kv[i] + pos // page_size])
            out_offsets.append(pos % page_size)
    return out_pages, out_offsets


@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("seed", range(20))
def test_packed_locations_match_the_index_arrays(page_size, seed):
    rng = random.Random(seed)
    views, next_page = [], 1
    for r in range(rng.randint(1, 9)):
        start = rng.choice([0, 0, rng.randrange(page_size)])
        length = rng.randint(1, 300)
        # prefill rows, decode rows and rows with nothing new beside each other
        to_compute = rng.choice([length, 1, 0, rng.randint(0, length)])
        n_pages = -(-(start + length) // page_size)
        page_idxs = list(range(next_page, next_page + n_pages))
        rng.shuffle(page_idxs)
        next_page += n_pages
        views.append(SequenceView(f"r{r}", "main", page_idxs, length, to_compute, start=start))
    indptrs = build_paged_indptrs(views, page_size)
    pages, offsets = packed_token_locations(indptrs, page_size)
    want_pages, want_offsets = _reference(indptrs, page_size)
    assert pages.dtype == np.int64 and offsets.dtype == np.int64
    assert pages.tolist() == want_pages
    assert offsets.tolist() == want_offsets
    assert len(pages) == sum(v.to_compute for v in views)
